#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

namespace gpnp {
struct GravityPoseConfig {
    bool enabled{false};
    std::string method{"soft_normal"}; // fixed_normal is an explicit diagnostic/control.
    double pixel_sigma_px{0.5}, huber_sigma{2.5};
    double normal_sigma_floor_deg{0.5}, plane_sigma_deg{1.0};
    double max_prior_sigma_deg{6.0}, max_prior_age_s{0.02};
    double max_normal_residual_sigma{4.0}, max_reprojection_px{3.0};
    int max_iterations{25};
    double init_window_s{1.0}, max_imu_gap_s{0.02};
    int min_init_samples{100};
    double gravity_m_s2{9.81}, acc_norm_tolerance{0.25}, acc_std_max{0.15};
    double gyro_motion_max_deg_s{3.0};
    double gyro_noise_rad_sqrt_s{0.0001}, gyro_bias_sigma_deg_s{0.02};
    double stationary_correction_tau_s{2.0};
};
void validateGravityPoseConfig(const GravityPoseConfig&);
GravityPoseConfig readGravityPoseConfig(const cv::FileNode&);

struct GravityNormalPrior {
    double t_sample{-1};
    Eigen::Vector3d normal_camera{Eigen::Vector3d::UnitZ()}; // template +Z, i.e. DOWN.
    Eigen::Matrix3d covariance_rad2{Eigen::Matrix3d::Zero()}; // left angular error; tangent rank 2.
    bool valid{false};
    std::string source{"unavailable"}, status{"not_initialized"};
};

// Deliberately separate from VisualPoseCandidate. This is a posterior using IMU,
// NOT a statistically independent six-DoF visual measurement for an ESKF.
struct GravityPoseResult {
    bool success{false}, uses_inertial_prior{true}, covariance_valid{false}, prior_valid{false};
    std::string status{"disabled"}, method, prior_source;
    double t_sample{-1}, prior_age_s{-1}, prior_sigma_deg{-1};
    double mean_error_px{-1}, rms_error_px{-1}, normal_residual_deg{-1};
    double time_ms{0};
    int support{0}, iterations{0};
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()}; // template -> camera
    Eigen::Vector3d prior_normal_camera{Eigen::Vector3d::UnitZ()};
    Eigen::Vector3d t_mm{Eigen::Vector3d::Zero()};
    Eigen::Matrix<double,6,6> posterior_covariance{Eigen::Matrix<double,6,6>::Zero()};
    std::string covariance_kind{"imu_conditioned_posterior_dt_mm_left_dtheta_rad"};
};
} // namespace gpnp
