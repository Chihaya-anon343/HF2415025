#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace gpnp {

// Legacy/standalone selection and fusion observation generation are separate.
// Missing configuration preserves the standalone pipeline exactly.
struct VisualOutputConfig {
    bool fusion_mode{false};
    bool export_candidates{false};
    bool estimate_covariance{true};
    double pixel_sigma_floor{0.5};
    double ambiguity_ratio{0.5};
    double ambiguity_gap_px{0.3};
    double separation_deg{10.0};
    double separation_translation_ratio{0.05};
};

using PoseCovariance = Eigen::Matrix<double, 6, 6>;

struct VisualPoseCandidate {
    bool uses_radar_prior{false}; ///< A radar-conditioned posterior is not independent visual evidence.
    bool uses_inertial_prior{false}; ///< Rejected by independent-visual ESKF update.
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()}; // template -> camera
    Eigen::Vector3d t_mm{Eigen::Vector3d::Zero()};
    // [additive dt_camera(mm), left dtheta_camera(rad)], including cross terms.
    // R' = Exp(dtheta) R, t' = t + dt. NOT Rodrigues-vector covariance.
    PoseCovariance covariance{PoseCovariance::Zero()};
    bool covariance_valid{false};
    bool prior_adjusted{false};
    double mean_error_px{0}, rms_error_px{0};
    int support{0};
    std::string source;
};

struct VisualCandidateDiagnostic {
    std::string source, status;
    double raw_error_px{-1}, final_error_px{-1};
    Eigen::Matrix3d raw_R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d raw_t_mm{Eigen::Vector3d::Zero()};
};

struct VisualObservation {
    int64_t frame_id{0};
    double t_sample{0}, t_arrival{0}; // same clock, seconds
    std::string timestamp_quality{"unspecified"};
    std::string strategy, model_id, model_frame;
    bool is_class1{false};
    bool identity_eligible{false};
    int image_width{0}, image_height{0};
    Eigen::Matrix3d camera_matrix{Eigen::Matrix3d::Identity()};
    bool evaluated{false}, ippe_complete{false}, ambiguous{false};
    std::string status{"not_evaluated"};
    double error_ratio{-1}, error_gap_px{-1};
    double rotation_gap_deg{0}, normal_gap_deg{0}, translation_gap_ratio{0};
    int alternate_index{-1};
    std::vector<VisualPoseCandidate> candidates;
    std::vector<VisualCandidateDiagnostic> diagnostics;
    // One observation group per frame. Alternatives share these pixels and
    // must NEVER be treated as independent camera updates.
    std::vector<cv::Point2f> image_points;
    std::vector<Eigen::Vector3d> object_points_mm;
    std::vector<int> point_ids;
};

struct VisualFeedback {
    int64_t frame_id{0};
    int candidate_index{-1};
    bool accepted{false};
    std::string status;
    double nis{-1}, score_gap{-1};
    VisualObservation observation;
};

} // namespace gpnp
