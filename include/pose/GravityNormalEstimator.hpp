#pragma once
#include "common/GravityPose.hpp"
#include <deque>

namespace gpnp {
// Only caller-declared stationary intervals may initialize or correct gravity.
// A near-g accelerometer norm by itself is NOT proof of zero acceleration.
class GravityNormalEstimator {
public:
    explicit GravityNormalEstimator(const GravityPoseConfig& config,
        const Eigen::Matrix3d& R_imu_cam=Eigen::Matrix3d::Identity());
    void reset();
    bool feedImu(double t, const Eigen::Vector3d& specific_force,
                 const Eigen::Vector3d& gyro_rad_s, bool stationary_hint=false);
    GravityNormalPrior priorAt(double exposure_time) const;
private:
    struct Sample {double t; Eigen::Vector3d acc, gyro;};
    GravityPoseConfig cfg_;
    Eigen::Matrix3d R_imu_cam_;
    std::deque<Sample> window_;
    Eigen::Vector3d normal_imu_{Eigen::Vector3d::UnitZ()}, last_gyro_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias_{Eigen::Vector3d::Zero()};
    double last_t_{-1}, initialized_t_{-1}, last_correction_t_{-1};
    double angular_variance_{0};
    bool initialized_{false};
    std::string status_{"not_initialized"};
};
} // namespace gpnp
