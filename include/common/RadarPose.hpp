#pragma once
#include "common/GravityPose.hpp"
#include <cstdint>

namespace gpnp {
struct RadarPoseConfig {
    bool enabled{false};
    std::string measurement_type{"vertical_height"};
    double height_sigma_m{1.0}, max_age_s{.04}, max_tilt_deg{20.0};
    double age_noise_m_s{10.0}, max_height_residual_sigma{4.0};
};
RadarPoseConfig readRadarPoseConfig(const cv::FileNode&);
void validateRadarPoseConfig(const RadarPoseConfig&);

// A logged height working model, not a verified manufacturer's slant-range model.
// Installation and target/ground offsets are zero for this first version.
struct RadarHeightPrior {
    bool valid{false};
    std::string status{"unavailable"}, source{"flight_csv_vertical_height"}, stamp;
    double t_sample{-1}, height_m{0}, age_s{-1};
    int64_t sample_id{-1}, raw_row{-1};
    size_t merged_rows{0};
};

// Separate posterior. Never append to independent VisualObservation candidates.
struct RadarPoseResult : GravityPoseResult {
    bool uses_radar_prior{false};
    int64_t radar_sample_id{-1};
    double radar_height_m{-1}, radar_age_s{-1}, height_sigma_m{-1};
    double predicted_height_m{-1}, height_residual_m{0}, tilt_deg{-1};
};
}
