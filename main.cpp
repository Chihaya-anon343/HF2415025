#include "utils/DualCornerRecoveryDiagnostics.hpp"
#include "utils/PoseAxisDisplay.hpp"
#include "common/VisualObservationPolicy.hpp"
#include "utils/VisualObservationIO.hpp"
#include "utils/GravityPoseIO.hpp"
#include "common/GravityActivation.hpp"
#include "input/FlightImuSource.hpp"
/**
 * @file main.cpp
 * @brief GPNP 双目视觉跟踪器入口 —— YOLO检测 → 策略选择 → 位姿解算
 *
 * 运行: ./GPNP [config_path]
 * 配置: config/tracker_config.json
 */

#include "tracker/StereoTracker.hpp"
#include "tracker/MonoTracker.hpp"
#include "detection/YoloRoiProvider.hpp"
#include "common/GeometryUtils.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/TargetModel.hpp"
#include "common/MonoDualRoiPolicy.hpp"
#include "common/StereoRobustness.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "utils/AsyncImageSaver.hpp"
#include "utils/LiveMjpegServer.hpp"

// 输入系统 (Phase 1)
#include "input/InputProvider.hpp"
#include "input/InputConfig.hpp"
#include "input/VideoSource.hpp"
#include "input/FileStereoSource.hpp"
#include "input/SimulatedSensors.hpp"

// ESKF 多源融合 (可选)
#include "fusion/EskfFusionManager.hpp"
#include "fusion/ConditionedFlightFusion.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>

namespace {
// Phase 3.2: 优雅退出 — Ctrl+C (SIGINT) / kill (SIGTERM) 置位, 帧循环检测后收尾
volatile std::sig_atomic_t g_stop_requested = 0;
void handleSignal(int) {
    g_stop_requested = 1;
}

// ---- 实时预览 (live_view) 叠加绘制 ----
// 位姿成功帧: X红/Y绿/朝上法向-Z蓝 (仅显示，轴长随距离自适应) + 左上角状态行
void drawLiveOverlay(cv::Mat& img, const Eigen::Matrix3d& K,
                     const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                     const std::string& status) {
    const double len_d = gpnp::utils::poseAxisDisplayLength(t.norm());
    const float L = static_cast<float>(len_d);
    const std::vector<cv::Point3f> axes = {
        cv::Point3f(0, 0, 0), cv::Point3f(L, 0, 0),
        cv::Point3f(0, L, 0), cv::Point3f(0, 0, gpnp::utils::upNormalZ(L, t.z()))};

    cv::Mat R_cv(3, 3, CV_64F);
    cv::Matx33d K_cv;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_cv.at<double>(i, j) = R(i, j);
            K_cv(i, j) = K(i, j);
        }
    }
    cv::Mat rvec;
    cv::Rodrigues(R_cv, rvec);

    std::vector<cv::Point2f> pts;
    cv::projectPoints(axes, rvec, cv::Mat(cv::Vec3d(t.x(), t.y(), t.z())), K_cv,
                      cv::noArray(), pts);
    const cv::Point origin(pts[0]);
    const cv::Scalar axis_colors[3] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}};
    for (int i = 0; i < 2; ++i)
        cv::line(img, origin, cv::Point(pts[i + 1]), axis_colors[i], 2, cv::LINE_AA);
    gpnp::utils::drawUpNormal(img, origin, pts[3]);

    if (!status.empty())
        cv::putText(img, status, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
}

// Tracker output observations are already in full-image coordinates, including
// DualRoi class1 fallbacks. Draw observations, never projected model corners.
void drawLiveCorners(cv::Mat& image, const gpnp::PipelineResult& result) {
    for (const auto& p : result.pts_left_match) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            p.x < 0 || p.y < 0 || p.x >= image.cols || p.y >= image.rows)
            continue;
        const cv::Point center(cvRound(p.x), cvRound(p.y));
        // Dark outline keeps the measured corner visible on a white target.
        cv::circle(image, center, 5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::circle(image, center, 5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::drawMarker(image, center, cv::Scalar(0, 255, 255),
                       cv::MARKER_CROSS, 7, 1, cv::LINE_AA);
    }
}

// ---- 配置读取辅助 (ESKF 节) ----
double readDouble(const cv::FileNode& n, const char* key, double def) {
    cv::FileNode v = n[key];
    return v.empty() ? def : static_cast<double>(v);
}

/// 读取 3×3 矩阵 [[r00,r01,r02],...]; 失败 (缺省/尺寸不符) 保持 out 原值并返回 false
bool readMat3(const cv::FileNode& node, Eigen::Matrix3d& out) {
    if (node.empty() || node.size() != 3) return false;
    for (int i = 0; i < 3; ++i) {
        cv::FileNode row = node[i];
        if (row.empty() || row.size() != 3) return false;
        for (int j = 0; j < 3; ++j) {
            cv::FileNode e = row[j];
            if (e.empty()) return false;
            out(i, j) = static_cast<double>(e);
        }
    }
    return true;
}

/// 读取 3 维向量 [x,y,z]; 失败保持 out 原值并返回 false
bool readVec3(const cv::FileNode& node, Eigen::Vector3d& out) {
    if (node.empty() || node.size() != 3) return false;
    for (int i = 0; i < 3; ++i) {
        cv::FileNode e = node[i];
        if (e.empty()) return false;
        out(i) = static_cast<double>(e);
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    using namespace gpnp;

    // ========================================================================
    // ① 读取配置文件
    // ========================================================================
    std::string config_path = "config/tracker_config.json";
    if (argc >= 2) config_path = argv[1];

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开配置文件: " << config_path << std::endl;
        return 1;
    }

    // 相机内参 + 基线
    double fx = fs["camera"]["fx"], fy = fs["camera"]["fy"];
    double cx = fs["camera"]["cx"], cy = fs["camera"]["cy"];
    double baseline = fs["camera"]["baseline_mm"];

    // 板模板参数（Dual-ROI BC 模板派生与物理尺寸来源）
    std::string template_path = fs["strategies"]["board_template"]["template_path"];
    double tw = fs["strategies"]["board_template"]["template_real_width_mm"];
    double th = fs["strategies"]["board_template"]["template_real_height_mm"];
    int min_pts = fs["strategies"]["board_template"]["gpnp_min_pts"];
    bool use_init_pnp = static_cast<int>(fs["strategies"]["board_template"]["use_initial_pnp"]) != 0;
    // 单目模式下 stereo_frontend (视差链) 配置整体不读取、不校验、不生效
    bool mono_mode = false;
    cv::FileNode mono_node = fs["mono_mode"];
    if (!mono_node.empty()) {
        mono_mode = static_cast<int>(mono_node) != 0;
    }

    StereoFlowConfig stereo_flow_cfg;
    TemplateGeometryConfig template_geometry_cfg;
    bool unified_target_model = false;
    MonoPosePriorConfig mono_pose_prior_cfg;
    GravityPoseConfig gravity_pose_cfg;
    RadarPoseConfig radar_pose_cfg;
    RoiSelectionConfig roi_selection_cfg;
    try {
        if (!mono_mode) stereo_flow_cfg = readStereoFlowConfig(fs["stereo_frontend"]);
        unified_target_model = readUnifiedTargetModelEnabled(fs.root());
        if (unified_target_model) template_geometry_cfg.enabled = true;
        else template_geometry_cfg = readTemplateGeometryConfig(fs["template_geometry"]);
        mono_pose_prior_cfg = readMonoPosePriorConfig(fs["mono_pose_prior"]);
        gravity_pose_cfg = readGravityPoseConfig(fs["gravity_pose"]);
        radar_pose_cfg = readRadarPoseConfig(fs["radar_pose"]);
        roi_selection_cfg = readRoiSelectionConfig(fs["roi_selection"]);
    }
    catch (const std::exception& e) {
        std::cerr << "跟踪配置错误: " << e.what() << std::endl;
        return 1;
    }
    const std::string output_override = fs["output"]["directory"].empty()
        ? "" : static_cast<std::string>(fs["output"]["directory"]);

    // BinaryCorner 策略参数
    BinaryCornerExtractor::Config binary_cfg;
    binary_cfg.corners      = fs["strategies"]["binary_corner"]["corners"];
    binary_cfg.kernel_size   = fs["strategies"]["binary_corner"]["kernel_size"];
    binary_cfg.scale         = fs["strategies"]["binary_corner"]["corner_scale"];
    binary_cfg.target_size   = cv::Size(
        fs["strategies"]["binary_corner"]["target_width"],
        fs["strategies"]["binary_corner"]["target_height"]);
    binary_cfg.pixel_to_meter_scale_class0 = fs["strategies"]["binary_corner"]["pixel_to_meter_scale_class0"];
    binary_cfg.pixel_to_meter_scale_class1 = fs["strategies"]["binary_corner"]["pixel_to_meter_scale_class1"];
    binary_cfg.roi_pad_pixels = fs["strategies"]["binary_corner"]["roi_pad_pixels"];
    binary_cfg.mono_identity = readMonoCornerIdentityConfig(fs["strategies"]["binary_corner"]["mono_identity"]);
    {
        auto n=fs["strategies"]["binary_corner"]["class1_corner_recovery"];
        auto& c=binary_cfg.class1_corner_recovery;
        if(!n.empty()) {
            if(!n["enabled"].empty()) c.enabled=(int)n["enabled"]!=0;
            if(!n["repair_enabled"].empty()) c.repair_enabled=(int)n["repair_enabled"]!=0;
            if(!n["min_reliable_points"].empty()) c.min_reliable_points=(int)n["min_reliable_points"];
            if(!n["min_pose_inliers"].empty()) c.min_pose_inliers=(int)n["min_pose_inliers"];
            if(!n["min_model_coverage"].empty()) c.min_model_coverage=(double)n["min_model_coverage"];
            if(!n["residual_ratio"].empty()) c.residual_ratio=(double)n["residual_ratio"];
            if(!n["residual_mad_scale"].empty()) c.residual_mad_scale=(double)n["residual_mad_scale"];
            if(!n["edge_canny_low"].empty()) c.edge_canny_low=(double)n["edge_canny_low"];
            if(!n["edge_canny_high"].empty()) c.edge_canny_high=(double)n["edge_canny_high"];
            if(!n["edge_band_ratio"].empty()) c.edge_band_ratio=(double)n["edge_band_ratio"];
            if(!n["edge_radius_ratio"].empty()) c.edge_radius_ratio=(double)n["edge_radius_ratio"];
            if(!n["edge_max_rms_ratio"].empty()) c.edge_max_rms_ratio=(double)n["edge_max_rms_ratio"];
        }
        if(c.enabled && (!validClass1CornerRecoveryConfig(c) || c.min_reliable_points>binary_cfg.corners ||
                         (binary_cfg.mono_identity.enabled && binary_cfg.mono_identity.class1_enabled)))
            throw std::runtime_error("Invalid class1_corner_recovery settings or conflicting class1 mono_identity");
    }
    std::string binary_template_dir = fs["strategies"]["binary_corner"]["template_dir"];

    // TinyTarget 策略参数
    TinyTargetExtractor::Config tiny_cfg;
    tiny_cfg.target_size   = cv::Size(
        fs["strategies"]["tiny_target"]["target_width"],
        fs["strategies"]["tiny_target"]["target_height"]);
    tiny_cfg.scale_factor  = fs["strategies"]["tiny_target"]["scale_factor"];
    tiny_cfg.upscale_max_roi_area = readTinyTargetUpscaleMaxRoiArea(
        fs["strategies"]["tiny_target"]["upscale_max_roi_area"]);
    tiny_cfg.square_size_m_class0 = fs["strategies"]["tiny_target"]["square_size_m_class0"];
    tiny_cfg.square_size_m_class1 = fs["strategies"]["tiny_target"]["square_size_m_class1"];
    tiny_cfg.roi_pad_pixels = fs["strategies"]["tiny_target"]["roi_pad_pixels"];
    tiny_cfg.robustness = readTinyTargetRobustnessConfig(
        fs["strategies"]["tiny_target"]["robustness"]);
    tiny_cfg.robustness.diagnostics = tiny_cfg.robustness.diagnostics ||
                                     roi_selection_cfg.tt_verification.enabled;
    std::string tiny_template_dir = fs["strategies"]["tiny_target"]["template_dir"];

    // YOLO 检测参数
    std::string model_path = fs["yolo"]["model_path"];
    float conf = fs["yolo"]["conf_threshold"];
    int target_cls = fs["yolo"]["target_class_id"];
    float expand = fs["yolo"]["roi_expand_ratio"];
    int min_roi = fs["yolo"]["roi_min_size"];

    // 策略选择阈值（从 JSON 读取，替代原硬编码常量）
    int tiny_max_area  = fs["strategies"]["tiny_max_area"];
    int dual_trigger_area = fs["strategies"]["dual_trigger_area"];
    bool stereo_mono_fallback = static_cast<int>(fs["strategies"]["stereo_mono_fallback"]) != 0;

    // 近距离回退配置（class0 丢失时用 class1）
    RoiGenerator::CloseRangeConfig close_range_cfg;
    int tiny_max_area_class1  = 0;
    cv::FileNode cr_node = fs["strategies"]["close_range"];
    if (!cr_node.empty()) {
        close_range_cfg.enabled          = static_cast<int>(cr_node["enabled"]) != 0;
        close_range_cfg.class1_min_area  = cr_node["class1_min_area"];
        close_range_cfg.roi_expand_ratio = cr_node["roi_expand_ratio"];
        close_range_cfg.min_expand_pixels = cr_node["min_expand_pixels"];
        tiny_max_area_class1  = static_cast<int>(cr_node["tiny_max_area"]);
        close_range_cfg.tiny_max_area  = tiny_max_area_class1;
    }

    // 双 ROI 配置（class 1 ROI 拓展像素 + class1 三级退化链开关 + 巨型 primary 短路）
    int dual_expand = 10;
    bool dual_class1_fallback = true;
    double dual_primary_span_ratio = 0.99;
    bool dual_tier1_only = false;
    std::string dual_c0_points_path;
    std::string dual_c1_points_path;
    double dual_joint_gate_px = 12.0;
    cv::FileNode dual_node = fs["strategies"]["dual_roi"];
    const auto mono_dual_roi_cfg = readMonoDualRoiConfig(dual_node["mono_robustness"]);
    const auto stereo_robustness_cfg=readStereoRobustnessConfig(fs["stereo_frontend"]["robustness"]);
    if (!dual_node.empty()) {
        dual_expand = dual_node["secondary_expand_pixels"];
        {
            cv::FileNode c1fb = dual_node["class1_fallback_enabled"];
            if (!c1fb.empty()) dual_class1_fallback = static_cast<int>(c1fb) != 0;
        }
        {
            cv::FileNode p0 = dual_node["class0_points_path"];
            if (!p0.empty()) dual_c0_points_path = static_cast<std::string>(p0);
        }
        {
            cv::FileNode p1 = dual_node["class1_points_path"];
            if (!p1.empty()) dual_c1_points_path = static_cast<std::string>(p1);
        }
        {
            cv::FileNode gate = dual_node["joint_match_gate_px"];
            if (!gate.empty()) dual_joint_gate_px = static_cast<double>(gate);
        }
        {
            cv::FileNode span = dual_node["primary_span_ratio"];
            if (!span.empty()) dual_primary_span_ratio = static_cast<double>(span);
        }
        {
            cv::FileNode t1only = dual_node["tier1_only"];
            if (!t1only.empty()) dual_tier1_only = static_cast<int>(t1only) != 0;
        }
    }

    // ========================================================================
    // 运行模式 — normal (InputProvider+简洁) / debug (旧路径+详细)
    // ========================================================================
    std::string run_mode = "normal";
    {
        cv::FileNode mn = fs["mode"];
        if (!mn.empty()) run_mode = static_cast<std::string>(mn);
    }
    bool verbose_console  = (run_mode == "debug");
    bool visualize_detailed = verbose_console;  ///< Debug 模式生成 per-strategy 面板，与日志捕获独立
    bool use_input_system = (run_mode == "normal");

    // 输入输出
    std::string left_path  = fs["input"]["left"];
    std::string right_path = fs["input"]["right"];
    bool visualize = static_cast<int>(fs["output"]["visualize"]) != 0;
    bool log_file   = false;
    bool pose_csv_enabled = false;  // Optional full-precision experiment diagnostics.
    bool feature_points_csv_enabled = false;
    {
        cv::FileNode lf = fs["output"]["log_file"];
        if (!lf.empty()) log_file = static_cast<int>(lf) != 0;
        cv::FileNode pc = fs["output"]["pose_csv"];
        if (!pc.empty()) pose_csv_enabled = static_cast<int>(pc) != 0;
        cv::FileNode fp = fs["output"]["feature_points_csv"];
        if (!fp.empty()) feature_points_csv_enabled = static_cast<int>(fp) != 0;
    }
    // The frame table also records skipped/failed frames with no point rows.
    if (feature_points_csv_enabled) pose_csv_enabled = true;

    // 实时预览配置 (output.live_view, 可选): 内嵌 MJPEG 服务器, 浏览器实时观看
    LiveMjpegServer::Config live_cfg;
    bool live_enabled = false;
    {
        cv::FileNode lv = fs["output"]["live_view"];
        if (!lv.empty()) {
            live_enabled = static_cast<int>(lv["enabled"]) != 0;
            cv::FileNode p = lv["port"];
            if (!p.empty()) live_cfg.port = static_cast<int>(p);
            cv::FileNode q = lv["jpeg_quality"];
            if (!q.empty()) live_cfg.jpeg_quality = static_cast<int>(q);
            cv::FileNode f = lv["max_fps"];
            if (!f.empty()) live_cfg.max_fps = static_cast<double>(f);
        }
    }

    // 手动 ROI 配置 — 仅 debug 模式生效
    bool use_manual_roi = false;
    RoiRect manual_rl, manual_rr;
    if (run_mode == "debug") {
        cv::FileNode manual_node = fs["manual_roi"];
        if (!manual_node.empty()) {
            int enabled = manual_node["enabled"];
            if (enabled) {
                use_manual_roi = true;
                manual_rl.x      = manual_node["left"]["x"];
                manual_rl.y      = manual_node["left"]["y"];
                manual_rl.width  = manual_node["left"]["width"];
                manual_rl.height = manual_node["left"]["height"];
                manual_rr.x      = manual_node["right"]["x"];
                manual_rr.y      = manual_node["right"]["y"];
                manual_rr.width  = manual_node["right"]["width"];
                manual_rr.height = manual_node["right"]["height"];
            }
        }
    }

    // ========================================================================
    // ①-b 解析 input_system 配置（normal 模式用）
    // ========================================================================
    input::InputSystemConfig input_sys_cfg;
    int max_frames = (run_mode == "debug") ? 2 : 0;

    cv::FileNode input_sys_node = fs["input_system"];
    if (!input_sys_node.empty()) {
        // Phase 3.1: 线程化采集配置 (Camera 类型自动启用, 可不配置)
        input_sys_cfg.use_threaded_capture =
            static_cast<int>(input_sys_node["use_threaded_capture"]) != 0;
        {
            cv::FileNode rc = input_sys_node["ring_capacity"];
            if (!rc.empty()) input_sys_cfg.ring_capacity = static_cast<int>(rc);
        }

        cv::FileNode img_node = input_sys_node["image"];
        if (!img_node.empty()) {
            std::string img_type = img_node["type"];
            if (img_type == "file") {
                input_sys_cfg.image.type = input::ImageSourceType::File;
                input_sys_cfg.image.left_path = static_cast<std::string>(img_node["left_path"]);
                input_sys_cfg.image.right_path = static_cast<std::string>(img_node["right_path"]);
            } else if (img_type == "directory") {
                input_sys_cfg.image.type = input::ImageSourceType::Directory;
                input_sys_cfg.image.directory_path = static_cast<std::string>(img_node["directory_path"]);
                input_sys_cfg.image.left_pattern = static_cast<std::string>(img_node["left_pattern"]);
                input_sys_cfg.image.right_pattern = static_cast<std::string>(img_node["right_pattern"]);
                if (input_sys_cfg.image.left_pattern.empty()) input_sys_cfg.image.left_pattern = "left";
                if (input_sys_cfg.image.right_pattern.empty()) input_sys_cfg.image.right_pattern = "right";
            } else if (img_type == "sequence") {
                input_sys_cfg.image.type = input::ImageSourceType::Sequence;
                input_sys_cfg.image.directory_path = static_cast<std::string>(img_node["directory_path"]);
                input_sys_cfg.image.sequence_pattern = static_cast<std::string>(img_node["sequence_pattern"]);
                if (input_sys_cfg.image.sequence_pattern.empty()) input_sys_cfg.image.sequence_pattern = "frame";
            } else if (img_type == "video") {
                try {input_sys_cfg.image=input::readVideoInputConfig(img_node);}
                catch(const std::exception& e) {std::cerr<<"视频配置错误: "<<e.what()<<std::endl;return 1;}
            } else if (img_type == "camera") {
                input_sys_cfg.image.type = input::ImageSourceType::Camera;
                input_sys_cfg.image.camera_devices = static_cast<std::string>(img_node["camera_devices"]);
                if (input_sys_cfg.image.camera_devices.empty()) input_sys_cfg.image.camera_devices = "0";
                input_sys_cfg.image.target_fps = 0.0;
                {
                    cv::FileNode tf = img_node["target_fps"];
                    if (!tf.empty()) input_sys_cfg.image.target_fps = static_cast<double>(tf);
                }
            }
        }
        // Preserve the requested IMU state separately from source readiness.
        // Phase-1 InputProvider still has no operational IMU source.
        cv::FileNode imu_node=input_sys_node["imu"];
        if(!imu_node.empty()) {
            try {input_sys_cfg.imu=input::readImuInputConfig(imu_node);}
            catch(const std::exception& e){std::cerr<<"IMU configuration error: "<<e.what()<<'\n';return 1;}
        }
        try {input_sys_cfg.altimeter=input::readAltimeterInputConfig(input_sys_node["altimeter"]);}
        catch(const std::exception& e){std::cerr<<"Radar configuration error: "<<e.what()<<"\n";return 1;}
        if (use_input_system) {
            max_frames = static_cast<int>(input_sys_node["max_frames"]);
            if (max_frames <= 0) max_frames = 999999;
        }
    }

    // ========================================================================
    // ①-c ESKF 融合配置 (可选, 默认关闭) + 合成 IMU/雷达源配置
    // ========================================================================
    const bool video_input=use_input_system && input_sys_cfg.image.type==input::ImageSourceType::Video;
    if(video_input && !mono_mode) {std::cerr<<"视频文件输入当前仅支持mono_mode=true"<<std::endl;return 1;}
    fusion::EskfFusionConfig eskf_cfg;
    input::SimulatedImuConfig sim_imu_cfg;
    input::SimulatedRadarConfig sim_radar_cfg;

    cv::FileNode ek = fs["eskf"];
    if (!ek.empty()) {
        eskf_cfg.enabled = static_cast<int>(ek["enabled"]) != 0;
        if (eskf_cfg.enabled) {
            eskf_cfg.imu_rate_hz   = readDouble(ek, "imu_rate_hz", 200.0);
            eskf_cfg.radar_rate_hz = readDouble(ek, "radar_rate_hz", 20.0);
            eskf_cfg.max_imu_gap_s = readDouble(ek, "max_imu_gap_s", 0.1);
            eskf_cfg.max_cam_gap_s = readDouble(ek, "max_cam_gap_s", 1.0);

            // 反向传播 (延迟测量) 参数
            eskf_cfg.backprop_window_s = readDouble(ek, "backprop_window_s", 0.2);
            eskf_cfg.state_hist_hz = static_cast<int>(readDouble(ek, "state_hist_hz", 100.0));
            eskf_cfg.max_output_age_s = readDouble(ek, "max_output_age_s", 0.5);
            {
                cv::FileNode lf = ek["latency_fallback"];
                std::string lf_str = lf.empty() ? "inflate" : static_cast<std::string>(lf);
                eskf_cfg.latency_fallback =
                    (lf_str == "reject") ? fusion::LatencyFallback::Reject
                                         : fusion::LatencyFallback::Inflate;
            }
            eskf_cfg.threaded = static_cast<int>(ek["threaded"]) != 0;
            const auto vo=ek["visual_observation"];
            if(!vo.empty()) {
                eskf_cfg.visual_nis_gate=readDouble(vo,"nis_gate",22.46);
                eskf_cfg.visual_score_gap=readDouble(vo,"score_gap",4.0);
                eskf_cfg.visual_init_frames=static_cast<int>(readDouble(vo,"init_frames",2));
                eskf_cfg.visual_init_rotation_deg=readDouble(vo,"init_rotation_deg",10);
                eskf_cfg.visual_init_translation_m=readDouble(vo,"init_translation_m",0.2);
                eskf_cfg.visual_init_max_position_std_m=readDouble(vo,"init_max_position_std_m",0.25);
                eskf_cfg.visual_init_max_rotation_std_deg=readDouble(vo,"init_max_rotation_std_deg",15);
                eskf_cfg.visual_position_floor_m=readDouble(vo,"position_floor_m",0.005);
                eskf_cfg.visual_rotation_floor_deg=readDouble(vo,"rotation_floor_deg",1);
            }

            cv::FileNode nz = ek["noise"];
            if (!nz.empty()) {
                eskf_cfg.params.sigma_acc       = readDouble(nz, "sigma_acc", 0.3);
                eskf_cfg.params.sigma_gyro      = readDouble(nz, "sigma_gyro", 0.02);
                eskf_cfg.params.sigma_acc_bias  = readDouble(nz, "sigma_acc_bias", 0.01);
                eskf_cfg.params.sigma_gyro_bias = readDouble(nz, "sigma_gyro_bias", 0.001);
                eskf_cfg.params.sigma_pos_rw    = readDouble(nz, "sigma_pos_rw", 0.5);
                eskf_cfg.params.cam_pos_noise   = readDouble(nz, "cam_pos_noise", 0.1);
                eskf_cfg.params.cam_rot_noise   = readDouble(nz, "cam_rot_noise", 1.0);
                eskf_cfg.params.use_cam_z_noise_decoupling =
                    static_cast<int>(nz["use_cam_z_noise_decoupling"]) != 0;
                eskf_cfg.params.cam_pos_z_noise = readDouble(nz, "cam_pos_z_noise", 5.0);
                eskf_cfg.params.radar_alt_noise = readDouble(nz, "radar_alt_noise", 0.30);
            }
            cv::FileNode ng = ek["gravity"];
            if (!ng.empty() && ng.size() == 3) {
                Eigen::Vector3d g;
                bool ok = true;
                for (int i = 0; i < 3; ++i) {
                    if (ng[i].empty()) { ok = false; break; }
                    g(i) = static_cast<double>(ng[i]);
                }
                if (ok) eskf_cfg.params.gravity = g;
            }
            readMat3(ek["R_imu_cam"], eskf_cfg.params.R_imu_cam);      // 失败保持默认 (I)
            readVec3(ek["p_imu_in_cam"], eskf_cfg.params.p_imu_in_cam); // 失败保持默认 (0)
            readMat3(ek["R_template_world"], eskf_cfg.R_template_world);

            cv::FileNode ni = ek["init_std"];
            if (!ni.empty()) {
                eskf_cfg.init_std_p  = readDouble(ni, "p", 1.0);
                eskf_cfg.init_std_v  = readDouble(ni, "v", 1.0);
                eskf_cfg.init_std_q  = readDouble(ni, "q", 0.1);
                eskf_cfg.init_std_ba = readDouble(ni, "ba", 0.1);
                eskf_cfg.init_std_bg = readDouble(ni, "bg", 0.01);
            }
        }
    }

    // 合成数据源: input_system.imu.type == "simulated" / altimeter.type == "simulated"
    if (!input_sys_node.empty()) {
        cv::FileNode inode = input_sys_node["imu"];
        if (!inode.empty()
            && static_cast<std::string>(inode["type"]) == "simulated") {
            sim_imu_cfg.enabled    = true;
            sim_imu_cfg.rate_hz    = readDouble(inode, "rate_hz", 200.0);
            sim_imu_cfg.sigma_acc  = readDouble(inode, "sigma_acc", 0.1);
            sim_imu_cfg.sigma_gyro = readDouble(inode, "sigma_gyro", 0.005);
            readVec3(inode["bias_acc"], sim_imu_cfg.bias_acc);
            readVec3(inode["bias_gyro"], sim_imu_cfg.bias_gyro);
        }
        cv::FileNode anode = input_sys_node["altimeter"];
        if (!anode.empty()
            && static_cast<std::string>(anode["type"]) == "simulated") {
            sim_radar_cfg.enabled = true;
            sim_radar_cfg.rate_hz = readDouble(anode, "rate_hz", 20.0);
            sim_radar_cfg.noise_m = readDouble(anode, "noise_m", 0.30);
            sim_radar_cfg.initial_height =
                readDouble(anode, "initial_height", 10.0);
            sim_radar_cfg.inject_jump_every_s =
                readDouble(anode, "inject_jump_every_s", 0.0);
            sim_radar_cfg.jump_m = readDouble(anode, "jump_m", 50.0);
        }
    }

    // 时序连贯性配置 (temporal, 可选; 缺省值见 TemporalConfig 定义)
    TemporalConfig temporal_cfg;
    {
        cv::FileNode tn = fs["temporal"];
        if (!tn.empty()) {
            temporal_cfg.final_acceptance = readFinalPoseAcceptanceConfig(tn["final_acceptance"]);
            cv::FileNode en = tn["enabled"];
            if (!en.empty()) temporal_cfg.enabled = static_cast<int>(en) != 0;

            cv::FileNode pcn = tn["pose_chain"];
            if (!pcn.empty()) {
                cv::FileNode v = pcn["max_cache_age_frames"];
                if (!v.empty()) temporal_cfg.max_cache_age_frames = static_cast<int>(v);
                v = pcn["tie_epsilon_px"];
                if (!v.empty()) temporal_cfg.tie_epsilon_px = static_cast<double>(v);
            }

            cv::FileNode mg = tn["motion_gate"];
            if (!mg.empty()) {
                cv::FileNode v = mg["max_trans_ratio"];
                if (!v.empty()) temporal_cfg.max_trans_ratio = static_cast<double>(v);
                v = mg["max_rot_deg"];
                if (!v.empty()) temporal_cfg.max_rot_deg = static_cast<double>(v);
                v = mg["max_scale_ratio"];
                if (!v.empty()) temporal_cfg.max_scale_ratio = static_cast<double>(v);
                v = mg["switch_margin"];
                if (!v.empty()) temporal_cfg.switch_margin = static_cast<double>(v);
            }

            cv::FileNode stn = tn["stickiness"];
            if (!stn.empty()) {
                cv::FileNode v = stn["hold_frames"];
                if (!v.empty()) temporal_cfg.hold_frames = static_cast<int>(v);
                v = stn["hysteresis_up"];
                if (!v.empty()) temporal_cfg.hysteresis_up = static_cast<double>(v);
                v = stn["hysteresis_down"];
                if (!v.empty()) temporal_cfg.hysteresis_down = static_cast<double>(v);
                v = stn["locked_fail_limit"];
                if (!v.empty()) temporal_cfg.locked_fail_limit = static_cast<int>(v);
            }

            cv::FileNode v = tn["area_ema_alpha"];
            if (!v.empty()) temporal_cfg.area_ema_alpha = static_cast<double>(v);
        }
    }

    const VisualOutputConfig visual_output=readVisualOutputConfig(fs["vision_output"]);
    if(visual_output.fusion_mode && !mono_mode)
        throw std::invalid_argument("vision_output.mode=fusion currently requires mono_mode=true");
    const bool conditioned=eskf_cfg.enabled && std::string(ek["pose_source"])=="imu_radar_pnp";
    if(conditioned && (!video_input || !mono_mode || !gravity_pose_cfg.enabled || !radar_pose_cfg.enabled ||
        !input_sys_cfg.imu.enabled || input_sys_cfg.imu.type!="flight_csv" ||
        !input_sys_cfg.altimeter.enabled || input_sys_cfg.altimeter.type!="flight_csv" ||
        visual_output.fusion_mode || eskf_cfg.threaded || input_sys_cfg.use_threaded_capture))
        throw std::invalid_argument("imu_radar_pnp requires synchronous mono flight video, standalone C branch and flight_csv sensors");
    if(conditioned) {
        Eigen::Matrix3d extrinsic;
        if(!readMat3(ek["R_imu_cam"],extrinsic))throw std::invalid_argument("missing ESKF installation rotation");
        for(int i=0;i<9;++i)if(std::abs(extrinsic(i/3,i%3)-input_sys_cfg.imu.imu_to_camera_rotation[i])>1e-8)
            throw std::invalid_argument("AHRS and ESKF installation rotations differ");
        input_sys_cfg.imu.full_rate_replay=input_sys_cfg.altimeter.full_rate_replay=true;
    }
    if(input_sys_cfg.imu.enabled&&input_sys_cfg.imu.type=="flight_csv"&&eskf_cfg.enabled&&!conditioned) {
        std::cerr<<"flight_csv currently supplies gravity and one aligned sample per frame; full-rate ESKF replay is not enabled\n";return 1;
    }
    if(radar_pose_cfg.enabled && (!use_input_system || !mono_mode || !input_sys_cfg.altimeter.enabled || input_sys_cfg.altimeter.type!="flight_csv" || !gravity_pose_cfg.enabled || !input_sys_cfg.imu.enabled || (eskf_cfg.enabled&&!conditioned))) {std::cerr<<"incompatible radar_pose inputs or ESKF route\n";return 1;}
    const auto gravity_config_gate=resolveGravityActivation(gravity_pose_cfg.enabled,
        use_input_system && input_sys_cfg.imu.enabled && !sim_imu_cfg.enabled,true,mono_mode);
    if(gravity_pose_cfg.enabled && !gravity_config_gate.enabled)
        std::cerr<<"[GravityPose] disabled: "<<gravity_config_gate.reason<<" (requires independent IMU input)\n";
    gravity_pose_cfg.enabled=gravity_config_gate.enabled;
    if(visual_output.fusion_mode && eskf_cfg.enabled && sim_imu_cfg.enabled)
        throw std::invalid_argument("Fusion hypotheses require independent IMU; PnP-driven simulated IMU is a plumbing-only source");
    fs.release();

    // ========================================================================
    // 日志文件捕获 — normal 模式 + log_file 时重定向 cout 到字符串流
    // ========================================================================
    std::ostringstream log_body;
    std::streambuf* saved_cout = nullptr;
    bool capture_log = log_file && !verbose_console;   // 仅 normal 模式（非 verbose）
    if (capture_log) {
        saved_cout = std::cout.rdbuf(log_body.rdbuf());  // cout → 日志；保存原始终端缓冲区
        verbose_console = true;                        // 强制 verbose 以便捕获详细处理消息
    }

    // ========================================================================
    // ② 构造输出目录
    // ========================================================================
    namespace fsp = std::filesystem;
    std::string output_dir;
    if (use_input_system) {
        // 从实际输入源目录名派生
        std::string src_dir = input_sys_cfg.image.directory_path;
        if(video_input)src_dir=fsp::path(input_sys_cfg.image.video_path).stem().string();
        if (!src_dir.empty()) {
            std::string folder = fsp::path(src_dir).filename().string();
            if (folder.empty()) folder = fsp::path(src_dir).parent_path().filename().string();
            output_dir = "output/" + (folder.empty() ? "sequence" : folder);
        } else {
            output_dir = "output/sequence";
        }
    } else {
        std::string img_name = fsp::path(left_path).stem().string();
        {
            auto pos = img_name.find(" - ");
            if (pos != std::string::npos) img_name = img_name.substr(0, pos);
        }
        output_dir = "output/" + img_name;
    }
    if (!output_override.empty()) output_dir = output_override;
    if (video_input || visualize || log_file || pose_csv_enabled || feature_points_csv_enabled || stereo_flow_cfg.disparity_only ||
        visual_output.export_candidates || visual_output.fusion_mode) fsp::create_directories(output_dir);
    if (feature_points_csv_enabled && !mono_mode)
        throw std::runtime_error("feature_points_csv currently supports mono mode only");
    std::ofstream feature_csv;
    std::ofstream input_frames_csv;
    if(video_input) {
        input_frames_csv.open(output_dir+"/input_frames.csv");
        if(!input_frames_csv)throw std::runtime_error("Cannot open video input ledger");
        input_frames_csv<<"frame,source_frame_index,logical_timestamp_us,source_timestamp,clock_segment,timestamp_quality,decoded_frames,skipped_frames,decode_ms,visualization_file,pose_success,feature_points,failure,seek_method,startup_ms,seek_bypassed_frames\n"<<std::setprecision(17);
    }
    std::ofstream recovery_csv;
    std::ofstream class0_recovery_csv;
    std::ofstream dual_recovery_csv;
    if(feature_points_csv_enabled && mono_dual_roi_cfg.corner_recovery.enabled) {
        dual_recovery_csv.open(output_dir+"/dual_corner_recovery.csv");
        if(!dual_recovery_csv)throw std::runtime_error("Cannot open DualROI corner recovery CSV");
        gpnp::dualCornerRecoveryHeader(dual_recovery_csv);
    }
    if(feature_points_csv_enabled && binary_cfg.mono_identity.class0_recovery.enabled) {
        class0_recovery_csv.open(output_dir+"/class0_corner_recovery.csv");
        if(!class0_recovery_csv) throw std::runtime_error("Cannot open class0 corner recovery CSV");
        class0_recovery_csv<<"frame,strategy,pose_success,recovery_status,id,original_x,original_y,predicted_x,predicted_y,final_x,final_y,residual,status,threshold_px\n"<<std::setprecision(17);
    }
    if(feature_points_csv_enabled && binary_cfg.class1_corner_recovery.enabled) {
        recovery_csv.open(output_dir + "/class1_corner_recovery.csv");
        if(!recovery_csv) throw std::runtime_error("Cannot open class1 corner recovery CSV");
        recovery_csv << "frame,strategy,pose_success,recovery_status,id,original_x,original_y,predicted_x,predicted_y,final_x,final_y,residual,status,threshold_px\n" << std::setprecision(17);
    }
    if (feature_points_csv_enabled) {
        feature_csv.open(output_dir + "/feature_points.csv");
        if (!feature_csv) throw std::runtime_error("Cannot open feature point CSV in " + output_dir);
        feature_csv << "frame,strategy,success,point_index,class_id,model_id,model_table_id,x,y,used_in_pose\n"
                    << std::setprecision(17);
    }
    std::ofstream pose_csv;
    std::ofstream visual_json,feedback_json;
    std::ofstream gravity_json,radar_json,radar_alignment_csv;
    std::ofstream imu_alignment_csv;
    if(visual_output.export_candidates || visual_output.fusion_mode) {
        visual_json.open(output_dir+"/visual_observations.jsonl");
        feedback_json.open(output_dir+"/visual_feedback.jsonl");
        if(!visual_json || !feedback_json)throw std::runtime_error("Cannot open visual observation output");
    }
    if (pose_csv_enabled) {
        pose_csv.open(output_dir + "/poses.csv");
        if (!pose_csv) throw std::runtime_error("Cannot open pose CSV in " + output_dir);
        pose_csv << "frame,success,strategy,is_class1,model_frame,n_matched,warm_start,gate_status,seed_age,fallback,extract_ms,pnp_ms,tx,ty,tz";
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) pose_csv << ",R" << i << j;
        pose_csv << ",failure_reason,roi_candidates,roi_verified,roi_selected,"
                    "roi_containment,roi_reason,roi_verify_ms,bc_detected,bc_"
                    "rejected,pose_points,tt_status,tt_legacy_iou,tt_direction_iou,"
                    "tt_direction_gap,tt_quadrant,tt_refined_count,tt_max_refine_shift_px,roi_verification_details,"
                    "continuity_status,continuity_rotation_deg,continuity_normal_deg,continuity_translation_ratio,"
                    "continuity_scale_ratio,continuity_age,continuity_pending,continuity_rejected_candidates,tt_scale_factor_used,visual_status,visual_candidates,visual_ambiguous,visual_error_ratio,visual_error_gap_px\n"
                 << std::setprecision(17);
    }
    auto exportPose = [&](int frame, const PipelineResult& result) {
        if(gravity_json)writeGravityPose(gravity_json,frame,result.gravity_pose);
        if(radar_json)writeGravityPose(radar_json,frame,result.radar_pose,&result.radar_pose);
        if(visual_json)writeVisualObservation(visual_json,result.visual_observation);
        if(dual_recovery_csv)gpnp::writeDualCornerRecovery(dual_recovery_csv,frame,result);
        if(class0_recovery_csv) {
            for(const auto& o:result.class0_corner_observations) {
                class0_recovery_csv<<frame<<','<<result.strategy_name<<','<<result.success<<','<<result.class0_corner_recovery_status
                    <<','<<o.id<<','<<o.original.x<<','<<o.original.y<<',';
                if(o.has_prediction)class0_recovery_csv<<o.predicted.x<<','<<o.predicted.y;
                else class0_recovery_csv<<',';
                class0_recovery_csv<<','<<o.final_point.x<<','<<o.final_point.y<<','<<o.residual<<','<<o.status<<','<<o.threshold_px<<'\n';
            }
            if(!result.class0_corner_recovery_status.empty() && result.class0_corner_observations.empty()) {
                class0_recovery_csv<<frame<<','<<result.strategy_name<<','<<result.success<<','<<result.class0_corner_recovery_status<<",-1";
                for(int i=0;i<6;++i)class0_recovery_csv<<',';
                class0_recovery_csv<<",-1,not_observed,0\n";
            }
        }
        if(recovery_csv) for(const auto& o:result.class1_corner_observations) {
            recovery_csv << frame << ',' << result.strategy_name << ',' << result.success << ','
                << result.class1_corner_recovery_status << ',' << o.id << ',' << o.original.x << ',' << o.original.y << ',';
            if(o.has_prediction) recovery_csv << o.predicted.x << ',' << o.predicted.y;
            else recovery_csv << ',';
            recovery_csv << ',' << o.final_point.x << ',' << o.final_point.y << ',' << o.residual << ',' << o.status << ',' << o.threshold_px << '\n';
        }
        if (feature_points_csv_enabled) {
            // Mono points are full-image coordinates. The joint model table
            // appends class1 after the explicit class0 prefix.
            for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                int id = i < result.good_matches.size() ? result.good_matches[i].trainIdx : -1;
                int cls = result.is_class1 ? 1 : 0;
                int local_id = id;
                if (result.strategy_name == "DualRoi_BCJ" &&
                    result.pose_model_class0_count > 0 && id >= result.pose_model_class0_count) {
                    cls = 1;
                    local_id -= result.pose_model_class0_count;
                }
                const bool used = result.success && id >= 0 &&
                    id < static_cast<int>(result.pose_model_points.size());
                feature_csv << frame << ',' << result.strategy_name << ',' << result.success
                    << ',' << i << ',' << cls << ',' << local_id << ',' << id << ','
                    << result.pts_left_match[i].x << ',' << result.pts_left_match[i].y
                    << ',' << used << '\n';
            }
        }
        if (!pose_csv_enabled) return;
        auto csvString = [&](const std::string& value) {
            pose_csv << '"';
            for (char c : value) { if (c == '"') pose_csv << '"'; pose_csv << c; }
            pose_csv << '"';
        };
        const bool valid = result.success && !result.disparity_only;
        pose_csv << frame << ',' << valid << ',';
        csvString(result.strategy_name);
        pose_csv << ',' << result.is_class1 << ',';
        csvString(result.model_frame);
        pose_csv << ',' << result.n_matched << ',' << result.warm_start_used
                 << ',' << static_cast<int>(result.gate_status) << ',' << result.seed_age
                 << ',' << result.fallback_used << ',' << result.extract_ms << ',' << result.pnp_ms;
        for (int i = 0; i < 3; ++i) {
            pose_csv << ',';
            if (valid) pose_csv << result.t(i);
        }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            pose_csv << ',';
            if (valid) pose_csv << result.R(i, j);
        }
        pose_csv << ',';
        csvString(result.failure_reason);
        pose_csv << ',' << result.roi_candidates << ',' << result.roi_verified
                 << ',' << result.roi_selected << ',' << result.roi_containment
                 << ',';
        csvString(result.roi_selection_reason);
        auto roi_timing = result.timing.find("roi_verify");
        pose_csv << ','
                 << (roi_timing == result.timing.end() ? 0 : roi_timing->second)
                 << ',' << result.bc_detected_count << ','
                 << result.bc_rejected_count << ',' << result.gpnp_n_pts << ',';
        csvString(result.tt_status);
        pose_csv << ',' << result.tt_legacy_iou << ',' << result.tt_direction_iou
                 << ',' << result.tt_direction_gap << ',' << result.tt_quadrant
                 << ',' << result.tt_refined_count << ',' << result.tt_max_refine_shift_px << ',';
        csvString(result.roi_verification_details);
        pose_csv << ',';csvString(result.continuity_status);
        pose_csv << ',' << result.continuity_rotation_deg << ',' << result.continuity_normal_deg
                 << ',' << result.continuity_translation_ratio << ',' << result.continuity_scale_ratio
                 << ',' << result.continuity_age << ',' << result.continuity_pending
                 << ',' << result.continuity_rejected_candidates << ',' << result.tt_scale_factor_used << ',';
        csvString(result.visual_observation.status);
        pose_csv << ',' << result.visual_observation.candidates.size() << ',' << result.visual_observation.ambiguous
                 << ',' << result.visual_observation.error_ratio << ',' << result.visual_observation.error_gap_px << '\n';
    };
    std::cout << "输出目录: " << output_dir << std::endl;

    // ========================================================================
    // ③ 初始化相机 + tracker
    // ========================================================================
    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;
    Eigen::Matrix3d R_rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_rl(baseline, 0.0, 0.0);

    TrackerConfig tracker_cfg = makeTrackerConfig(min_pts, use_init_pnp, tw, th,
                                                  tiny_max_area, tiny_max_area_class1,
                                                  dual_expand);
    tracker_cfg.dual_roi_class1_fallback = dual_class1_fallback;
    tracker_cfg.dual_roi_primary_span_ratio = dual_primary_span_ratio;
    tracker_cfg.dual_roi_tier1_only = dual_tier1_only;
    tracker_cfg.dual_roi_class0_points_path = dual_c0_points_path;
    tracker_cfg.dual_roi_class1_points_path = dual_c1_points_path;
    tracker_cfg.dual_roi_joint_match_gate_px = dual_joint_gate_px;
    tracker_cfg.mono_dual_roi = mono_dual_roi_cfg;
    tracker_cfg.stereo_robustness=stereo_robustness_cfg;
    tracker_cfg.stereo_flow = stereo_flow_cfg;
    tracker_cfg.template_geometry = template_geometry_cfg;
    tracker_cfg.unified_target_model = unified_target_model;
    tracker_cfg.mono_pose_prior = mono_pose_prior_cfg;
    tracker_cfg.gravity_pose = gravity_pose_cfg;
    tracker_cfg.radar_pose = radar_pose_cfg;
    tracker_cfg.roi_selection = roi_selection_cfg;
    tracker_cfg.temporal = temporal_cfg;
    tracker_cfg.visual_output = visual_output;

    try {
        // ====================================================================
        // 图像加载 —— 新版 InputProvider 或 旧版 cv::imread
        // ====================================================================
        input::InputProvider input_provider;
        cv::Mat left_img, right_img;  // 旧版路径使用

        if (use_input_system) {
            std::cout << "[输入系统] 初始化 InputProvider..." << std::endl;
            if (!input_provider.initialize(input_sys_cfg))
                throw std::runtime_error("InputProvider 初始化失败");
            std::cout << "[输入系统] 就绪, 最大帧数: " << max_frames << std::endl;
        } else {
            // 旧版路径：直接从路径加载图像
            std::cout << "加载图像: " << left_path << " / " << right_path << std::endl;
            left_img  = cv::imread(left_path,  cv::IMREAD_COLOR);
            right_img = cv::imread(right_path, cv::IMREAD_COLOR);
            if (left_img.empty() || right_img.empty())
                throw std::runtime_error("无法读取输入图像");
        }

        // ====================================================================
        // 初始化 YOLO — 手动 ROI 时跳过以节省加载时间
        // ====================================================================
        tracker_cfg.imu_input_enabled=use_input_system && input_sys_cfg.imu.enabled && input_provider.hasImuInput();
        const auto gravity_runtime_gate=resolveGravityActivation(gravity_pose_cfg.enabled,
            use_input_system && input_sys_cfg.imu.enabled,tracker_cfg.imu_input_enabled,mono_mode);
        if(gravity_pose_cfg.enabled && !gravity_runtime_gate.enabled)
            std::cerr<<"[GravityPose] disabled: "<<gravity_runtime_gate.reason<<"\n";
        gravity_pose_cfg.enabled=gravity_runtime_gate.enabled;
        tracker_cfg.gravity_pose=gravity_pose_cfg;
        if(radar_pose_cfg.enabled && (!gravity_pose_cfg.enabled || !input_provider.hasRadarInput()))throw std::runtime_error("radar_pose inputs unavailable");
        tracker_cfg.radar_pose=radar_pose_cfg;
        if(radar_pose_cfg.enabled) {
            radar_json.open(output_dir+"/radar_poses.jsonl");radar_alignment_csv.open(output_dir+"/radar_alignment.csv");
            if(!radar_json||!radar_alignment_csv)throw std::runtime_error("Cannot open radar ledgers");
            radar_alignment_csv<<"frame,source_frame_index,video_timestamp,radar_timestamp,sample_id,raw_row,merged_rows,age_ms,height_m,valid,status\n"<<std::setprecision(17);
            if(visualize)fsp::create_directories(output_dir+"/radar_visualization");
        }
        if(input_provider.hasImuInput()&&input_sys_cfg.imu.type=="flight_csv") {
            imu_alignment_csv.open(output_dir+"/imu_alignment.csv");
            if(!imu_alignment_csv)throw std::runtime_error("Cannot open IMU alignment ledger");
            imu_alignment_csv<<"frame,source_frame_index,video_timestamp,imu_timestamp,unique_index,raw_row,age_ms,valid,status,skipped_prefix,samples_since_frame,raw_rows,unique_samples,duplicate_rows,gx_rad_s,gy_rad_s,gz_rad_s,ax_m_s2,ay_m_s2,az_m_s2,roll_deg,pitch_deg,nx,ny,nz\n"<<std::setprecision(17);
        }
        if(gravity_pose_cfg.enabled) {
            fsp::create_directories(output_dir);
            gravity_json.open(output_dir+"/gravity_poses.jsonl");
            if(!gravity_json)throw std::runtime_error("Cannot open gravity pose output");
            if(visualize)fsp::create_directories(output_dir+"/gravity_visualization");
        }
        YoloRoiProvider yolo;
        bool yolo_ok = false;

        if (!use_manual_roi) {
            YoloConfig yolo_cfg = makeYoloConfig(model_path, DeviceType::CPU, conf);
            yolo_cfg.target_class_id = target_cls;
            yolo_cfg.roi_expand_ratio = expand;
            RoiGenerator::Config roi_cfg{target_cls, expand, min_roi,
                                         dual_trigger_area, roi_selection_cfg};
            yolo_ok = yolo.initialize(yolo_cfg, roi_cfg);
            if (yolo_ok) yolo.setCloseRangeConfig(close_range_cfg);
        }

        // ====================================================================
        // 初始化 Tracker（单目/双目 分支）
        // ====================================================================

        std::unique_ptr<TrackerBase> tracker;

        if (mono_mode) {
            std::cout << "初始化 MonoTracker（单目模式）..." << std::endl;
            auto t = std::make_unique<MonoTracker>(K, template_path, tracker_cfg,
                                                    binary_cfg, binary_template_dir,
                                                    tiny_cfg, tiny_template_dir);
            t->setOutputDir(output_dir);
            t->setVerboseConsole(verbose_console);
            t->setVisualizeDetailed(visualize_detailed);
            tracker = std::move(t);
            std::cout << "单目模式已启用（仅左图）" << std::endl;
        } else {
            std::cout << "初始化 StereoTracker（双目模式）..." << std::endl;
            auto t = std::make_unique<StereoTracker>(K, R_rl, t_rl, template_path, tracker_cfg,
                                                      binary_cfg, binary_template_dir,
                                                      tiny_cfg, tiny_template_dir);
            t->setOutputDir(output_dir);
            t->setVerboseConsole(verbose_console);
            t->setVisualizeDetailed(visualize_detailed);
            tracker = std::move(t);
        }

        // ====================================================================
        // ③-b ESKF 融合 + 合成数据源 (仅 normal 模式; debug 固定2帧无时间序列)
        // ====================================================================
        if (tracker->targetModel()) {
            fsp::create_directories(output_dir);
            writeTargetModel(output_dir+"/target_model.json",*tracker->targetModel());
        }
        fusion::EskfFusionManager fusion(eskf_cfg);
        input::SimulatedImu   sim_imu(sim_imu_cfg);
        input::SimulatedRadar sim_radar(sim_radar_cfg);
        std::unique_ptr<fusion::ConditionedFlightFusion> flight_filter;
        if(conditioned) {
            cv::FileStorage settings(config_path,cv::FileStorage::READ);
            flight_filter=std::make_unique<fusion::ConditionedFlightFusion>(settings["eskf"],output_dir);
        }
        bool fusion_active = eskf_cfg.enabled && use_input_system && !conditioned;
        auto receiveVisualFeedback=[&](PipelineResult* current) {
            for(const auto& feedback:fusion.takeVisualFeedback()) {
                if(feedback_json)writeVisualFeedback(feedback_json,feedback);
                if(mono_mode)static_cast<MonoTracker*>(tracker.get())->applyVisualFeedback(feedback);
                if(current && feedback.frame_id==current->visual_observation.frame_id) {
                    current->continuity_status="fusion_"+feedback.status;
                    if(feedback.accepted) {
                        const auto& c=feedback.observation.candidates[feedback.candidate_index];
                        current->R=c.R;current->t=c.t_mm;current->success=current->gpnp_success=true;
                        const Eigen::Quaterniond q(c.R);
                        current->quat_w=q.w();current->quat_x=q.x();
                        current->quat_y=q.y();current->quat_z=q.z();
                        current->gpnp_n_pts=c.support;current->failure_reason.clear();
                    }else current->failure_reason="fusion_"+feedback.status;
                }
            }
        };
        if (eskf_cfg.enabled && !use_input_system) {
            std::cerr << "[ESKF] debug 模式无时间序列, 融合已禁用 (eskf.enabled 忽略)"
                      << std::endl;
        }
        if (fusion_active && eskf_cfg.threaded) {
            fusion.start();
            std::cout << "[ESKF] 融合工作线程已启动 (异步消费)" << std::endl;
        }
        // 合成源真值参考 (最近一帧相机位姿)
        Eigen::Matrix3d last_R_cam_w = Eigen::Matrix3d::Identity();
        double last_cam_height = sim_radar_cfg.initial_height;

        // ====================================================================
        // ③-d 实时预览服务器 (live_view, 仅 normal 模式)
        // ====================================================================
        LiveMjpegServer live_server;
        if (live_enabled && use_input_system) {
            if (live_server.start(live_cfg)) {
                std::cerr << "[LiveView] 浏览器打开 http://<运行主机IP>:" << live_cfg.port
                          << "/ 实时观看 (黄色角点 + 成功帧三维轴; 失败帧标注PnP FAIL; 网页断流自动重连)"
                          << std::endl;
            } else {
                std::cerr << "[LiveView] 端口 " << live_cfg.port
                          << " 监听失败, 继续无预览运行" << std::endl;
            }
        }

        // ====================================================================
        // ④ 逐帧处理
        // ====================================================================
        // Phase 3.2: 位姿成功帧计数 (汇总统计用)
        int processed_ok = 0;
        int disparity_ok = 0;

        // 阶段计时累加 (运行汇总: YOLO / 单帧跟踪 / 特征提取 / 位姿解算平均用时)
        double yolo_ms_total = 0.0;  int yolo_calls = 0;
        double pose_ms_total = 0.0;  int pose_calls = 0;
        double extract_ms_total = 0.0;  int extract_calls = 0;
        double pnp_solve_ms_total = 0.0;  int pnp_solve_calls = 0;
        auto steadyNow = []() { return std::chrono::steady_clock::now(); };
        auto elapsedMs = [](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        };

        // 终端每帧一行输出。capture_log 时 cout 被重定向到日志，这里直写原始终端缓冲区
        auto termLine = [&](const std::string& s) {
            if (capture_log && saved_cout) {
                std::string t = s + '\n';
                saved_cout->sputn(t.data(), t.size());
            } else {
                std::cout << s << std::endl;
            }
        };

        // 处理单帧: YOLO/ROI → 策略 → PnP; 返回 PipelineResult (失败帧 success=false)
        auto processFrame = [&](int frame, const cv::Mat& L, const cv::Mat& R) -> PipelineResult {
            PipelineResult result;
            result.disparity_only=!mono_mode&&stereo_flow_cfg.disparity_only;

            if (mono_mode) {
                auto* mt = static_cast<MonoTracker*>(tracker.get());
                RoiGroup left_group;
                std::vector<RoiCandidate> roi_candidates;
                if (use_manual_roi) {
                    left_group = RoiGroup{manual_rl, {}, false};
                } else if (yolo_ok) {
                    auto t_yolo = steadyNow();
                    if (roi_selection_cfg.enabled &&
                        roi_selection_cfg.verify_mono) {
                      roi_candidates = yolo.detectMonoCandidates(L);
                      if (!roi_candidates.empty())
                        left_group = roi_candidates.front().group;
                    } else
                      left_group = yolo.detectMono(L);
                    yolo_ms_total += elapsedMs(t_yolo); ++yolo_calls;
                    if (!left_group.valid()) {
                        tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                        termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                        return result;
                    }
                }
                tracker->setFrameNumber(frame);
                auto t_pose = steadyNow();
                result =
                    roi_candidates.empty()
                        ? mt->process(L, visualize && !video_input, &left_group)
                        : mt->processCandidates(L, visualize && !video_input, roi_candidates);
                pose_ms_total += elapsedMs(t_pose); ++pose_calls;
            } else {
                auto* st = static_cast<StereoTracker*>(tracker.get());
                RoiGroup lg, rg;
                bool left_bc_flow = false;
                StereoTracker::LegacyStereoProvider legacy_provider;
                double legacy_yolo_ms=0;
                if (use_manual_roi) {
                    lg = RoiGroup{manual_rl, {}, false};
                    rg = RoiGroup{manual_rr, {}, false};
                    left_bc_flow = st->prefersLeftStereoFlow(lg);
                    if (verbose_console) {
                        std::cout << "  手动 ROI: left=(" << manual_rl.x << "," << manual_rl.y << ","
                                  << manual_rl.width << "," << manual_rl.height << "), right=("
                                  << manual_rr.x << "," << manual_rr.y << "," << manual_rr.width
                                  << "," << manual_rr.height << ")" << std::endl;
                    }
                } else if (yolo_ok) {
                    auto t_yolo = steadyNow();
                    if (stereo_flow_cfg.enabled) {
                        const LeftDetection left_detection = yolo.detectLeft(L);
                        lg = left_detection.group;
                        left_bc_flow = st->prefersLeftStereoFlow(lg);
                        if(left_bc_flow && lg.is_dual && stereo_flow_cfg.dual_roi_legacy_fallback)
                            legacy_provider=[&,left_detection]() {
                                const auto start=steadyNow();
                                auto groups=yolo.completeStereo(left_detection,R);
                                legacy_yolo_ms=elapsedMs(start);yolo_ms_total+=legacy_yolo_ms;
                                return groups;
                            };
                        if (!left_bc_flow && !stereo_flow_cfg.disparity_only)
                            std::tie(lg, rg) = yolo.completeStereo(left_detection, R);
                    } else {
                        std::tie(lg, rg) = yolo.detect(L, R);
                    }
                    yolo_ms_total += elapsedMs(t_yolo); ++yolo_calls;
                    if (!lg.valid() && !rg.valid()) {
                        tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                        termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                        return result;
                    }
                    if (lg.is_dual && verbose_console) {
                        std::cout << "  双 ROI 模式: secondary=(" << lg.secondary.width
                                  << "x" << lg.secondary.height << ")" << std::endl;
                    }
                }
                tracker->setFrameNumber(frame);
                if (stereo_flow_cfg.disparity_only && !left_bc_flow) {
                    result.disparity_only = true;
                    result.failure_reason = "disparity_only_requires_supported_flow_strategy";
                    tracker->onFrameSkipped();
                } else if (left_bc_flow) {
                    auto t_pose = steadyNow();
                    result = st->process(L, R, visualize, &lg, nullptr,legacy_provider);
                    pose_ms_total += elapsedMs(t_pose)-legacy_yolo_ms; ++pose_calls;
                } else if (lg.valid() && rg.valid()) {
                    // 双侧检测 → 双目
                    auto t_pose = steadyNow();
                    result = st->process(L, R, visualize, &lg, &rg);
                    pose_ms_total += elapsedMs(t_pose); ++pose_calls;
                } else if (stereo_mono_fallback && (lg.valid() || rg.valid())) {
                    // 单侧检测 → 单目降级（需 stereo_mono_fallback=true）
                    auto t_pose = steadyNow();
                    result = st->processMono(lg.valid() ? L : R, visualize,
                                              lg.valid() ? &lg : &rg);
                    pose_ms_total += elapsedMs(t_pose); ++pose_calls;
                } else {
                    tracker->onFrameSkipped();   // skip 帧也让位姿 seed 老化
                    termLine("[Frame " + std::to_string(frame) + "] YOLO未检测到目标");
                    return result;
                }
            }

            // ---- 详细输出（verbose / capture_log 时进日志）----
            if (result.success) ++processed_ok;   // 汇总统计
            if(result.disparity_only&&result.disparity_success)++disparity_ok;
            if (verbose_console) {
                std::cout << "  特征点: " << result.n_kp_left
                          << "  匹配: " << result.n_matched
                          << "  投影: " << result.n_projected
                          << "  模板: " << result.n_template_match;
                if (!result.disparity.empty()) {
                    std::vector<double> abs_disp;
                    for (double d : result.disparity) abs_disp.push_back(std::abs(d));
                    std::cout << "  视差: " << computeMedian(std::move(abs_disp)) << "px";
                }
                // 耗时口径: 提取+解算两相之和 (与日志文件 extract_ms/pnp_ms 列一致,
                // 跨退化链累加; 单/Dual 路径同口径可比)
                std::cout << "  GPNP: " << (result.disparity_only ? "跳过（仅视差）" : result.gpnp_success ? "成功" : "失败")
                          << "  耗时: " << (result.extract_ms + result.pnp_ms) << "ms";
                if (result.success) {
                    std::cout << "  PnP t(mm)=[" << result.t.x() << ", "
                              << result.t.y() << ", " << result.t.z() << "]"
                              << "  rot_unwrap=" << result.rot_deg_unwrapped << "deg";
                }
                if (result.stereo_method == "LK") {
                    std::cout << " frontend=LK source=" << poseSourceName(result.pose_source)
                              << " stereo=" << result.stereo_success << " match=" << result.stereo_match_ms << "ms";
                    if (!result.failure_reason.empty()) std::cout << " reason=" << result.failure_reason;
                    if(result.disparity_only)std::cout << " match_ok=" << result.disparity_success;
                    if(!result.route_detail.empty())std::cout << " route=" << result.route_detail;
                }
                std::cout << std::endl;
            }

            // 提取/解算两相耗时 (tracker 内部计时, 跨退化链累加)
            extract_ms_total += result.extract_ms;   ++extract_calls;
            pnp_solve_ms_total += result.pnp_ms;     ++pnp_solve_calls;

            return result;
        };

        std::map<std::string,int> video_timestamp_names;
        auto saveVideoResult=[&](int frame,const input::SensorPacket& packet,const PipelineResult& result) {
            if(!video_input)return;
            const auto& info=packet.image_info;
            std::string filename;
            if(visualize && !packet.left_image.empty()) {
                std::string stem=info.source_timestamp;
                if(stem.empty()) {
                    std::ostringstream name;name<<"t"<<std::setfill('0')<<std::setw(16)<<packet.timestamp_us<<"us";
                    stem=name.str();
                }
                if(video_timestamp_names[stem]++>0)stem+="_f"+std::to_string(info.source_frame_index);
                filename=stem+".png";
                cv::Mat image=packet.left_image.clone();
                std::ostringstream title;title<<"source="<<info.source_frame_index<<" frame="<<frame<<" "<<result.strategy_name;
                if(result.success)drawLiveOverlay(image,K,result.R,result.t,title.str());
                else cv::putText(image,"NO VALID POSE  "+title.str(),{12,28},cv::FONT_HERSHEY_SIMPLEX,.65,{0,0,255},2,cv::LINE_AA);
                drawLiveCorners(image,result);
                const std::string stamp=info.source_timestamp.empty()?"logical_us="+std::to_string(packet.timestamp_us):info.source_timestamp;
                cv::putText(image,stamp,{12,image.rows-16},cv::FONT_HERSHEY_SIMPLEX,.7,{0,0,0},4,cv::LINE_AA);
                cv::putText(image,stamp,{12,image.rows-16},cv::FONT_HERSHEY_SIMPLEX,.7,{0,255,255},1,cv::LINE_AA);
                utils::AsyncImageSaver::write(output_dir+"/"+filename,image);
                if(gravity_pose_cfg.enabled)utils::AsyncImageSaver::write(
                    output_dir+"/gravity_visualization/"+filename,
                    gravityPosePreview(packet.left_image,result.visual_observation,result.gravity_pose));
                if(radar_pose_cfg.enabled)utils::AsyncImageSaver::write(output_dir+"/radar_visualization/"+filename,gravityPosePreview(packet.left_image,result.visual_observation,result.radar_pose,"IMU+RADAR "));
            }
            auto cell=[&](const std::string& s) {input_frames_csv<<'"';for(char ch:s){input_frames_csv<<ch;if(ch=='"')input_frames_csv<<'"';}input_frames_csv<<'"';};
            input_frames_csv<<frame<<','<<info.source_frame_index<<','<<packet.timestamp_us<<',';
            cell(info.source_timestamp);input_frames_csv<<','<<info.clock_segment<<','<<info.timestamp_quality<<','
                <<info.decoded_frames<<','<<info.skipped_frames<<','<<info.decode_ms<<',';
            cell(filename);input_frames_csv<<','<<result.success<<','<<result.pts_left_match.size()<<',';
            cell(result.failure_reason);input_frames_csv<<','<<info.seek_method<<','<<info.startup_ms<<','<<info.seek_bypassed_frames<<'\n';
            if(imu_alignment_csv&&packet.imu_alignment) {
                const auto& m=*packet.imu_alignment;const auto p=packet.gravity_prior.value_or(GravityNormalPrior{});
                imu_alignment_csv<<frame<<','<<info.source_frame_index<<",\""<<info.source_timestamp<<"\",\""<<m.imu_stamp<<"\","
                    <<m.sample_index<<','<<m.raw_row<<','<<m.age_s*1000<<','<<m.valid<<','<<m.status<<','
                    <<m.skipped_prefix<<','<<m.samples_since_frame<<','<<m.raw_rows<<','<<m.unique_samples<<','<<m.duplicate_rows;
                for(int axis=0;axis<3;++axis)imu_alignment_csv<<','<<m.gyro(axis);
                for(int axis=0;axis<3;++axis)imu_alignment_csv<<','<<m.acc(axis);
                imu_alignment_csv<<','<<m.roll_deg<<','<<m.pitch_deg;
                for(int axis=0;axis<3;++axis)imu_alignment_csv<<','<<p.normal_camera(axis);
                imu_alignment_csv<<'\n';
            }
            if(radar_alignment_csv && packet.radar_prior) {
                const auto& r=*packet.radar_prior;
                radar_alignment_csv<<frame<<','<<info.source_frame_index<<",\""<<info.source_timestamp<<"\",\""<<r.stamp<<"\","
                    <<r.sample_id<<','<<r.raw_row<<','<<r.merged_rows<<','<<r.age_s*1000<<','<<r.height_m<<','<<r.valid<<','<<r.status<<'\n';
            }
            // Two full-resolution previews per frame can otherwise accumulate
            // an unbounded queue. Offline input may wait; no frame is dropped.
            if(imu_alignment_csv&&visualize&&frame%16==0)utils::AsyncImageSaver::flush();
        };

        if (use_input_system) {
            // Phase 3.2: 注册优雅退出信号 — Ctrl+C 后循环自然收尾 (汇总+日志落盘)
            std::signal(SIGINT, handleSignal);
            std::signal(SIGTERM, handleSignal);

            // 新版输入系统路径 —— 数据驱动的帧循环
            int frame = 0;
            input::SensorPacket packet;
            double t_prev = -1.0;   ///< 上一相机帧时刻 (s), 合成源生成区间用
            auto t_start = std::chrono::steady_clock::now();
            while (frame < max_frames && !g_stop_requested) {
                // 有限超时取帧: 让循环周期性醒来检查 Ctrl+C;
                // getNextPacket false 时用 isCaptureStopped 区分"源结束"与"超时重试"
                if (!input_provider.getNextPacket(packet, /*timeout_ms=*/200)) {
                    if (input_provider.isCaptureStopped()) break;  // 源结束/断开/失败
                    continue;                                       // 超时 → 重试
                }
                ++frame;
                double t_cam = packet.timestamp_us * 1e-6;
                bool video_saved=false;
                try {
                    if(visual_output.fusion_mode)receiveVisualFeedback(nullptr);
                    // ---- ESKF: 合成源按 [t_prev, t_cam] 生成 IMU/雷达样本 ----
                    if (fusion_active) {
                        if(packet.imu)fusion.feedImu(packet.imu->t,packet.imu->acc,packet.imu->gyro);
                        if(packet.height)fusion.feedRadar(t_cam,*packet.height);
                        if (t_prev >= 0.0) {
                            for (const auto& s : sim_imu.generate(t_prev, t_cam, last_R_cam_w))
                                fusion.feedImu(s.t, s.acc, s.gyro);
                            for (const auto& s : sim_radar.generate(t_prev, t_cam, last_cam_height))
                                fusion.feedRadar(s.t, s.height);
                        }
                        // 排干 IMU/雷达至 t_cam (覆盖 YOLO 丢失路径的惯性传播);
                        // 线程化模式下由融合工作线程自动排干, 无需在此显式调用
                        if (!eskf_cfg.threaded)
                            fusion.propagateTo(t_cam);
                    }
                    t_prev = t_cam;   // 提前推进, 异常路径也安全

                    if (verbose_console)
                        std::cout << "\n===== 第 " << frame << " 帧"
                                  << (mono_mode ? " (单目)" : "") << " =====" << std::endl;
                    if(gravity_pose_cfg.enabled) {
                        static_cast<MonoTracker*>(tracker.get())->setGravityNormalPrior(frame,t_cam,
                            packet.gravity_prior.value_or(GravityNormalPrior{}));
                        if(radar_pose_cfg.enabled)static_cast<MonoTracker*>(tracker.get())->setRadarHeightPrior(frame,packet.radar_prior.value_or(RadarHeightPrior{}));
                    }
                    PipelineResult result = processFrame(frame, packet.left_image, packet.right_image);
                    if(gravity_pose_cfg.enabled && result.gravity_pose.status=="disabled") {
                        result.gravity_pose.status="no_finalized_visual_observation";
                        result.gravity_pose.t_sample=t_cam;result.gravity_pose.method=gravity_pose_cfg.method;
                    }
                    if(radar_pose_cfg.enabled && result.radar_pose.status=="disabled") {result.radar_pose.status="no_finalized_visual_observation";result.radar_pose.t_sample=t_cam;result.radar_pose.method="soft_normal_height";}
                    auto& observation=result.visual_observation;
                    observation.frame_id=frame;observation.t_sample=t_cam;
                    const bool camera_clock=input_sys_cfg.image.type==input::ImageSourceType::Camera;
                    observation.t_arrival=camera_clock?
                        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count():t_cam;
                    observation.timestamp_quality=video_input?packet.image_info.timestamp_quality:
                        camera_clock?"host_grab_approx_steady_clock":"source_logical_clock";

                    // ---- ESKF: PnP 位姿 → 相机观测 (同时更新合成源真值参考) ----
                    if (fusion_active && !result.disparity_only) {
                        if(visual_output.fusion_mode) {
                            fusion.feedVisualObservation(observation);
                            receiveVisualFeedback(&result);
                            if(result.success)++processed_ok;
                        }else fusion.feedCameraPose(t_cam, result.R, result.t, result.success);
                        if (result.success) {
                            last_R_cam_w = eskf_cfg.R_template_world * result.R.transpose();
                            last_cam_height = fusion.position()(2);
                        }
                    }
                    if(flight_filter)flight_filter->step(packet,result.radar_pose);
                    exportPose(frame, result);
                    if(gravity_pose_cfg.enabled && visualize && !video_input && !packet.left_image.empty())
                        utils::AsyncImageSaver::write(output_dir+"/gravity_visualization/gravity_f"+std::to_string(frame)+".png",
                            gravityPosePreview(packet.left_image,result.visual_observation,result.gravity_pose));
                    saveVideoResult(frame,packet,result);video_saved=true;
                    if(visualize && !video_input && visual_output.fusion_mode && !packet.left_image.empty()) {
                        cv::Mat preview=packet.left_image.clone();
                        if(result.success)drawLiveOverlay(preview,K,result.R,result.t,"BACKEND ACCEPTED visual candidate");
                        else if(!observation.candidates.empty()) {
                            const auto& c=observation.candidates.front();
                            drawLiveOverlay(preview,K,c.R,c.t_mm,"UNCONFIRMED candidate - "+result.continuity_status);
                        }
                        drawLiveCorners(preview,result);
                        utils::AsyncImageSaver::write(output_dir+"/fusion_f"+std::to_string(frame)+".png",preview);
                    }

                    // ---- 实时预览: 实测角点 + 有效位姿三维轴 ----
                    if (live_enabled && live_server.running() &&
                        live_server.clientCount() > 0 && !packet.left_image.empty()) {
                        cv::Mat disp = packet.left_image.clone();
                        std::ostringstream os;
                        if (result.success) {
                            os << result.strategy_name << "  t(mm)=["
                               << result.t.x() << ", " << result.t.y() << ", "
                               << result.t.z() << "]  corners=" << result.pts_left_match.size();
                            drawLiveOverlay(disp, K, result.R, result.t, os.str());
                        } else if(visual_output.fusion_mode && !observation.candidates.empty()) {
                            const auto& candidate=observation.candidates.front();
                            drawLiveOverlay(disp,K,candidate.R,candidate.t_mm,"UNCONFIRMED candidate - "+result.continuity_status);
                        } else {
                            os << "PnP FAIL  corners=" << result.pts_left_match.size();
                            if(!result.continuity_status.empty())os << "  " << result.continuity_status;
                            cv::putText(disp, os.str(), cv::Point(12, 28),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                        }
                        // Mark corners last so the axes cannot cover them.
                        drawLiveCorners(disp, result);
                        live_server.submitFrame(disp);
                    }

                    // ---- normal 模式: 每帧一行 → 终端 (ESKF 融合为主输出) ----
                    if (!verbose_console || capture_log) {
                        std::ostringstream os;
                        fusion::FusionState st = fusion.getLatestState();
                        if (result.disparity_only) {
                            os << "[Frame " << frame << "] disparity_only match="
                               << (result.disparity_success ? "OK" : "FAIL") << " n=" << result.disparity.size();
                            if (!result.disparity.empty()) os << " median=" << computeMedian(result.disparity) << "px";
                            os << " pnp=SKIP match_ms=" << result.stereo_match_ms
                               << " roi=" << result.flow_roi.width << 'x' << result.flow_roi.height
                               << " full_retry=" << result.flow_full_fallback;
                            if (!result.failure_reason.empty()) os << " reason=" << result.failure_reason;
                            termLine(os.str());
                        } else if (fusion_active && st.initialized) {
                            const char* qs =
                                (st.quality == fusion::FusionQuality::Normal)   ? "N"
                              : (st.quality == fusion::FusionQuality::Degraded) ? "D"
                              : (st.quality == fusion::FusionQuality::Stale)    ? "S" : "?";
                            os << "[Frame " << frame << "] ESKF"
                               << " p=[" << st.position.x() << ", " << st.position.y() << ", " << st.position.z() << "]m"
                               << " v=[" << st.velocity.x() << ", " << st.velocity.y() << ", " << st.velocity.z() << "]"
                               << " q=[" << st.quaternion(0) << ", " << st.quaternion(1) << ", " << st.quaternion(2) << ", " << st.quaternion(3) << "]"
                               << " Q=" << qs;
                            if (result.success) {
                                os << " | PnP t(mm)=[" << result.t.x() << ", "
                                   << result.t.y() << ", " << result.t.z() << "]";
                            }
                            if (result.stereo_method == "LK")
                                os << " | frontend=LK source=" << poseSourceName(result.pose_source)
                                   << " stereo=" << result.stereo_success;
                            termLine(os.str());
                        } else if (result.success) {
                            Eigen::AngleAxisd aa(result.R);
                            Eigen::Vector3d rvec = aa.angle() * aa.axis();
                            std::string sname = result.strategy_name.empty()
                                ? "Unknown" : result.strategy_name;
                            os << "[Frame " << frame << "] " << sname
                               << "  n=" << result.n_matched
                               << "  r=[" << rvec.x() << ", " << rvec.y() << ", " << rvec.z() << "]"
                               << "  t=[" << result.t.x() << ", " << result.t.y() << ", " << result.t.z() << "]";
                            if (result.stereo_method == "LK")
                                os << " frontend=LK source=" << poseSourceName(result.pose_source)
                                   << " stereo=" << result.stereo_success << " match=" << result.stereo_match_ms << "ms";
                            termLine(os.str());
                        } else {
                            termLine("[Frame " + std::to_string(frame) + "] FAILED"
                                     + (result.failure_reason.empty() ? "" : " reason=" + result.failure_reason));
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Frame " << frame << "] 异常: " << e.what() << std::endl;
                    if(conditioned)throw; // Never continue a partially applied sensor batch.
                    if(video_input && !video_saved) {
                        PipelineResult failed;failed.failure_reason=e.what();saveVideoResult(frame,packet,failed);
                    }
                    // 继续下一帧，不中断整条序列
                }
            }

            // ---- 停止融合工作线程 (若已启动) ----
            if (fusion_active && eskf_cfg.threaded) fusion.stop();
            if(visual_output.fusion_mode)receiveVisualFeedback(nullptr);
            if(visual_output.fusion_mode && fusion_active)
                processed_ok=fusion.stats().cam_updates; // 包含晚于该帧落盘的异步验收

            // ---- 停止实时预览服务器 ----
            live_server.stop();

            // ---- Phase 3.2: 收尾汇总 — 停止采集线程 → 打印统计 ----
            input_provider.shutdown();
            auto termOut = [&](const std::string& s) {
                if (capture_log && saved_cout) saved_cout->sputn(s.data(), s.size());
                std::cout << s;   // capture_log 时同时进入日志文件
            };
            if (g_stop_requested) {
                termOut("用户中断 (Ctrl+C)，停止采集\n");
            } else if (frame == 0) {
                std::cerr << "警告: 输入系统未产生任何帧" << std::endl;
            }
            if (frame > 0) {
                auto t_end = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(t_end - t_start).count();
                input::InputProvider::InputStats ist = input_provider.stats();
                termOut("===== 运行汇总 =====\n");
                termOut("处理帧数: " + std::to_string(frame)
                        + "  位姿成功: " + std::to_string(processed_ok)
                        + " (" + std::to_string(100.0 * processed_ok / frame) + "%)\n");
                if(stereo_flow_cfg.disparity_only)termOut("视差匹配成功: "+std::to_string(disparity_ok)
                    +" ("+std::to_string(100.0*disparity_ok/frame)+"%)\n");
                if (secs > 0.0) {
                    termOut("实际处理 FPS: " + std::to_string(frame / secs) + "\n");
                }
                if (yolo_calls > 0) {
                    termOut("YOLO 平均用时: " + std::to_string(yolo_ms_total / yolo_calls)
                            + " ms/帧  (共 " + std::to_string(yolo_calls) + " 帧)\n");
                }
                if(video_input && frame>0) {
                    termOut("视频输入: 读取 "+std::to_string(packet.image_info.decoded_frames)+" 帧, 配置跳过 "+
                        std::to_string(packet.image_info.skipped_frames)+" 帧, 处理 "+std::to_string(frame)+" 帧; 原始帧末索引="+
                        std::to_string(packet.image_info.source_frame_index)+"; 定位="+packet.image_info.seek_method+
                        "; 启动耗时="+std::to_string(packet.image_info.startup_ms)+"ms\n");
                }
                if (pose_calls > 0) {
                    termOut("单帧跟踪平均用时 (提取+解算+ROI等): " + std::to_string(pose_ms_total / pose_calls)
                            + " ms/帧  (共 " + std::to_string(pose_calls) + " 帧)\n");
                }
                if (extract_calls > 0) {
                    termOut("特征提取平均用时: " + std::to_string(extract_ms_total / extract_calls)
                            + " ms/帧  (共 " + std::to_string(extract_calls) + " 帧)\n");
                }
                if (pnp_solve_calls > 0) {
                    termOut("位姿解算平均用时: " + std::to_string(pnp_solve_ms_total / pnp_solve_calls)
                            + " ms/帧  (共 " + std::to_string(pnp_solve_calls) + " 帧)\n");
                }
                if (yolo_ok) {
                    const auto counts = yolo.inferenceCounts();
                    termOut("YOLO推理次数: left=" + std::to_string(counts.left)
                            + " right=" + std::to_string(counts.right) + "\n");
                }
                if (ist.captured > 0) {
                    termOut("输入统计 (线程化采集): 采集 " + std::to_string(ist.captured)
                            + " 帧, 消费 " + std::to_string(ist.consumed)
                            + " 帧, 丢弃 " + std::to_string(ist.dropped)
                            + " 帧 ("
                            + std::to_string(100.0 * ist.dropped / ist.captured)
                            + "%)\n");
                }
                if (fusion_active) {
                    auto fst = fusion.stats();
                    termOut("ESKF 统计: IMU样本=" + std::to_string(fst.imu_samples)
                            + "  相机更新=" + std::to_string(fst.cam_updates)
                            + "  FDI忽略=" + std::to_string(fst.cam_ignored)
                            + "  姿态跳过=" + std::to_string(fst.cam_rot_skipped)
                            + "  雷达接受=" + std::to_string(fst.radar_accepted)
                            + "  雷达拒绝=" + std::to_string(fst.radar_rejected)
                            + "  延迟兜底=" + std::to_string(fst.cam_late_fallback) + "\n");
                }
            }
        } else {
            // 旧版路径 —— 固定帧数循环（向后兼容）
            for (int frame = 1; frame <= max_frames; ++frame) {
                try {
                    if (verbose_console)
                        std::cout << "\n===== 第 " << frame << " 帧"
                                  << (mono_mode ? " (单目)" : "") << " =====" << std::endl;
                    auto result = processFrame(frame, left_img, right_img);
                    result.visual_observation.frame_id=frame;
                    result.visual_observation.t_sample=result.visual_observation.t_arrival=frame-1;
                    result.visual_observation.timestamp_quality="debug_logical_clock";
                    exportPose(frame, result);
                } catch (const std::exception& e) {
                    std::cerr << "[Frame " << frame << "] 异常: " << e.what() << std::endl;
                }
            }
        }

        // M4: 帧循环结束 → 等待异步图像全部落盘, 避免程序退出时丢失
        gpnp::utils::AsyncImageSaver::flush();

        if (verbose_console)
            tracker->printLogs();
        if (verbose_console && yolo_ok) {
            const auto counts = yolo.inferenceCounts();
            std::cout << "YOLO推理次数: left=" << counts.left << " right=" << counts.right << "\n";
        }

        // ---- 日志文件输出（normal + log_file 时已捕获 cout）----
        if (capture_log) {
            std::cout.rdbuf(saved_cout);   // 恢复终端输出
            verbose_console = false;

            // 配置摘要头
            std::ostringstream hdr;
            hdr << "================================================================\n"
                << " Steretracker 追踪日志\n"
                << "================================================================\n"
                << "运行模式: " << run_mode << (mono_mode ? " (单目)" : " (双目)") << "\n"
                << "配置路径: " << config_path << "\n";
            if (use_input_system) {
                hdr << "输入源目录: " << input_sys_cfg.image.directory_path
                    << "  type=";
                switch (input_sys_cfg.image.type) {
                    case input::ImageSourceType::File:       hdr << "file"; break;
                    case input::ImageSourceType::Directory:  hdr << "directory"; break;
                    case input::ImageSourceType::Sequence:   hdr << "sequence"; break;
                    case input::ImageSourceType::Camera:     hdr << "camera"; break;
                    case input::ImageSourceType::Video:      hdr << "video " << input_sys_cfg.image.video_path; break;
                    default:                                 hdr << "unknown"; break;
                }
                hdr << "\n";
            } else {
                hdr << "左图: " << left_path << "\n"
                    << "右图: " << right_path << "\n";
            }
            hdr << "输出目录: " << output_dir << "\n"
                << "相机: fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy
                << " baseline_mm=" << baseline << "\n"
                << "YOLO: model=" << model_path << " conf=" << conf
                << " class=" << target_cls << " expand=" << expand << "\n"
                << "板模板: " << template_path << "  " << tw << "x" << th << "mm"
                << "  min_pts=" << min_pts << "\n"
                << "BC模板: " << binary_template_dir << "  corners=" << binary_cfg.corners
                << "  kernel=" << binary_cfg.kernel_size
                << "  p2m_c0=" << binary_cfg.pixel_to_meter_scale_class0
                << "  p2m_c1=" << binary_cfg.pixel_to_meter_scale_class1 << "\n"
                << "TT模板: " << tiny_template_dir << "  scale_factor=" << tiny_cfg.scale_factor
                << "  upscale_max_roi_area=" << tiny_cfg.upscale_max_roi_area
                << "  sq_c0=" << tiny_cfg.square_size_m_class0
                << "  sq_c1=" << tiny_cfg.square_size_m_class1 << "\n"
                << "面积阈值: tiny_max=" << tiny_max_area
                << "  class1_tiny=" << tiny_max_area_class1 << "\n"
                << "close_range: enabled=" << close_range_cfg.enabled
                << "  class1_min=" << close_range_cfg.class1_min_area
                << "  expand_ratio=" << close_range_cfg.roi_expand_ratio << "\n"
                << "================================================================\n\n";

            std::string log_path = output_dir + "/tracking_log.txt";
            std::ofstream ofs(log_path);
            if (ofs.is_open()) {
                ofs << hdr.str() << log_body.str();
                ofs.close();
                std::cout << "[Log] Saved: " << log_path << std::endl;
            } else {
                std::cerr << "[Log] Failed to open: " << log_path << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "致命错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
