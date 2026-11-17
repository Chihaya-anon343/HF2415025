#pragma once

/**
 * @file MonoTracker.hpp
 * @brief 单目视觉追踪器 —— 仅左图，EPnP 位姿解算。
 *
 * 继承 TrackerBase，不包含任何双目模块
 * （无 GPnPSolver / InitialPnPSolver 等双目解算与立体滤波组件）。
 */

#include "tracker/TrackerBase.hpp"
#include "pose/MonoPnPSolver.hpp"
#include "detection/RoiGenerator.hpp"
#include "common/RoiIdentityPolicy.hpp"
#include "common/MonoPoseContinuity.hpp"
#include "pose/GravityConstrainedPnPSolver.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gpnp {

class MonoTracker : public TrackerBase {
public:
    MonoTracker(const Eigen::Matrix3d& K,
                const std::string& template_path,
                const TrackerConfig& config,
                const BinaryCornerExtractor::Config& binary_cfg,
                const std::string& binary_template_dir,
                const TinyTargetExtractor::Config& tiny_cfg,
                const std::string& tiny_template_dir);

    /// 单目帧处理（仅左图）：ROI → 策略分发 → 单图特征提取 → EPnP 解算。
    PipelineResult process(const cv::Mat& left_img,
                           bool visualize = false,
                           const RoiGroup* left_group = nullptr);
    PipelineResult
    processCandidates(const cv::Mat &image, bool visualize,
                      const std::vector<RoiCandidate> &candidates);
    // Called on the tracker thread, never from the fusion worker. Only an
    // explicitly accepted original visual candidate may become trusted history.
    bool applyVisualFeedback(const VisualFeedback& feedback);
    // Called on the visual thread for this exact frame. No pure-visual cache changes.
    void setGravityNormalPrior(int64_t frame_id, double exposure_time, const GravityNormalPrior& prior) {
      gravity_frame_=frame_id;gravity_time_=exposure_time;gravity_prior_=prior;
    }
    void setRadarHeightPrior(int64_t frame_id,const RadarHeightPrior& prior) {radar_frame_=frame_id;radar_prior_=prior;}
    void clearCache() override {
      const int64_t reset_frame=current_frame_>0?current_frame_:state_.frame_count;
      TrackerBase::clearCache();
      has_verified_roi_ = false;
      identity_trusted_ = false;
      continuity_.reset();
      tentative_valid_=false;last_feedback_frame_=reset_frame;
      gravity_frame_=-1;gravity_prior_=GravityNormalPrior{};radar_frame_=-1;radar_prior_=RadarHeightPrior{};
    }

  private:
    bool fusionMode() const {return config_.visual_output.fusion_mode;}
    bool continuityActive() const {return !fusionMode() && config_.temporal.enabled && config_.temporal.final_acceptance.enabled;}
    bool tentativeSeed(PoseSeed& seed) const;
    void completeVisualMetadata(PipelineResult& result);
    PoseSeed tentative_seed_;
    bool tentative_valid_{false};
    int64_t tentative_frame_{-1},last_feedback_frame_{-1};
    cv::Size visual_image_size_;
    void finishMonoPose(PipelineResult& result, PoseEstimate& pose, bool class1, int band);
    MonoPoseContinuity continuity_;
    int continuity_band_{2}, continuity_rejections_{0};
    ContinuityDecision continuity_last_rejection_;
    RoiIdentityDecision checkRoiIdentity(const PipelineResult& observation,
        bool class1, const RoiRect& roi) const;
    PipelineResult processDualRoi(const cv::Mat& left_img,
                                   const RoiGroup& left_group,
                                   bool visualize);
    void prepareDualBcTemplate();
    void guardDualCornerRepair(PipelineResult& observations, PoseEstimate& pose,
        const std::vector<Eigen::Vector3d>& model, const RoiRect& roi,
        const PoseSeed* seed, double& pnp_ms, std::string& reason,
        bool points_global=false, uint64_t baseline_rng=0);
    /// Dual-ROI 第 3 级退化: 合并与 BC-only 均失败后, 在 secondary ROI (class1) 上
    /// 依次尝试 BC → TT (class1 3D 尺寸, State 5 同机制; seed 非 null 时注入 MonoPnP
    /// —— class1 图案与板共面无相对旋转, 板系 R seed 直接有效, 仅 t 差同心偏移)。
    /// 成功时 out_result 已截断为 2D/3D 严格 1:1 的配对子集 (全图坐标)，
    /// out_strategy 为 "DualRoi_C1BC"/"DualRoi_C1TT"。
    std::pair<bool, PoseEstimate> runDualRoiClass1Chain(
        const cv::Mat& left_gray, const cv::Mat& left_color,
        const RoiRect& left_sec,
        bool is_first,
        PipelineResult& out_result,
        std::vector<Eigen::Vector3d>& out_pts3d,
        std::string& out_strategy,
        double& extract_ms_acc, double& pnp_ms_acc,
        const PoseSeed* seed = nullptr,
        std::string* out_route_detail = nullptr);

    MonoPnPSolver mono_pnp_;
    std::unique_ptr<GravityConstrainedPnPSolver> gravity_solver_;
    GravityNormalPrior gravity_prior_;
    RadarHeightPrior radar_prior_;
    int64_t radar_frame_{-1};
    int64_t gravity_frame_{-1};
    double gravity_time_{-1};
    // Separate context: enabling class1 BC here must not change ordinary BC,
    // standalone class1 selection, or the stereo frontends.
    std::unique_ptr<BinaryCornerExtractor> dual_class1_bc_;
    bool has_verified_roi_{false};
    RoiRect verified_roi_;
    int verified_roi_frame_{-1};
    bool identity_trusted_{false};
    int identity_frame_{-1};
    cv::Size identity_image_size_;
    Eigen::Matrix3d identity_R_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d identity_t_{Eigen::Vector3d::Zero()};
    std::vector<Eigen::Vector3d> identity_models_[2];
    bool identity_validation_active_{false};
    bool has_prepared_bc_{false}, has_prepared_tt_{false};
    PipelineResult prepared_bc_, prepared_tt_;

    bool dual_bc_template_ready_{false};
    std::vector<int> dual_class0_id_map_;
    std::vector<cv::Point2f> dual_bc_tmpl_corners_;
    std::vector<Eigen::Vector3d> dual_bc_tmpl_pts3d_;
    std::vector<cv::Point2f> dual_c1_tmpl_corners_;    ///< class1 人工标注模板角点 (模板像素坐标)
    std::vector<Eigen::Vector3d> dual_c1_tmpl_pts3d_;  ///< class1 标注点板系 3D (与 corners 同序)
};

} // namespace gpnp
