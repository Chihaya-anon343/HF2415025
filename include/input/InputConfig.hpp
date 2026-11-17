#pragma once

/**
 * @file InputConfig.hpp
 * @brief 输入系统的配置结构体。
 *
 * 从 tracker_config.json 的 "input_system" 节反序列化。
 */

#include <string>
#include <array>

namespace gpnp {
namespace input {

// ============================================================================
// 图像源类型枚举
// ============================================================================

enum class ImageSourceType {
    File,       ///< 静态文件对（当前行为）
    Directory,  ///< 目录中的双目编号图像序列 (left_*/right_*)
    Sequence,   ///< 目录中的单目图像序列 (frame_0000.jpg, frame_0001.jpg...)
    Camera,     ///< 实时摄像头（Phase 3 已实现）
    Video       ///< Offline monocular video; synchronous, ordered, no queue drops.
};

// ============================================================================
// 图像输入配置
// ============================================================================

struct ImageInputConfig {
    ImageSourceType type = ImageSourceType::File;

    /// type == File: 左/右图像路径
    std::string left_path;
    std::string right_path;

    /// type == Directory: 包含 left_*.png / right_*.png 的目录
    std::string directory_path;

    /// type == Directory: 文件名模式（默认 "left" / "right"）
    std::string left_pattern  = "left";
    std::string right_pattern = "right";

    /// type == Sequence: 文件名前缀（如 "frame" 匹配 frame_0000.jpg）
    std::string sequence_pattern = "frame";

    /// type == Camera: 设备路径（如 "/dev/video0;video1"）
    std::string camera_devices;

    /// 帧率限制（Camera 模式），0 表示不限制
    double target_fps = 0.0;

    std::string video_path;
    std::string timestamp_path;
    std::string timestamp_mode{"frame_index"};
    double video_fps{0.0}; ///< 0: decoder-reported rate; logical time only, no sleep.
    int start_frame{0};
    int frame_step{1};
    std::string seek_mode{"sequential"}; ///< sequential (legacy), auto (fallback), indexed (strict).
    std::string indexed_video_path; ///< Verified stream-copy cache; manifest is <path>.index.json.
};

// ============================================================================
// IMU 输入配置
// ============================================================================

struct ImuInputConfig {
    bool full_rate_replay = false;
    bool enabled = false;
    std::string type{"serial"},path,gyro_unit,acc_unit,normal_source,tilt_order,tilt_unit;
    std::string duplicate_policy{"reject"},attitude_valid_from;
    double time_offset_s{0},max_sync_age_s{.02},tilt_sigma_deg{2};
    std::array<double,9> imu_to_camera_rotation{1,0,0,0,1,0,0,0,1};
    bool has_rotation{false};

    /// 串口设备路径，如 "/dev/ttyUSB0"
    std::string port;

    /// 波特率，如 921600
    int baud_rate = 921600;

    /// 协议类型: "custom" = 自定义二进制协议（需在 SerialImuSource 中实现）
    std::string protocol = "custom";
};

// ============================================================================
// 高度计输入配置
// ============================================================================

struct AltimeterInputConfig {
    bool full_rate_replay = false;
    bool enabled = false;
    std::string path, timestamp_column{"aligned_time"}, height_column{"height_m"};
    double max_age_s{.04}, min_height_m{.1}, max_height_m{500.0};
    double max_vertical_speed_m_s{15.0}, jump_margin_m{2.0};

    /// CAN 接口名称，如 "can0"
    std::string can_interface;

    /// 高度计类型: "can" = SocketCAN
    std::string type = "can";
};

// ============================================================================
// 输入系统总配置
// ============================================================================

struct InputSystemConfig {
    ImageInputConfig image;
    ImuInputConfig imu;
    AltimeterInputConfig altimeter;

    /// 若为 true，InputProvider 内部使用 RingBuffer + 采集线程
    /// （Camera 模式无论此值如何均自动启用，见 InputProvider::initialize）
    bool use_threaded_capture = false;

    /// 线程化采集的环形缓冲容量（帧数）。越小延迟越低，越大抗处理耗时尖峰。
    /// 消费策略为 take-latest：处理慢时跳过旧帧，保证拿到最新帧。
    int ring_capacity = 4;
};

} // namespace input
} // namespace gpnp
