#pragma once
#include "common/GravityPose.hpp"
#include "common/RadarPose.hpp"
#include "common/VisualObservation.hpp"

namespace gpnp {
class GravityConstrainedPnPSolver {
public:
    explicit GravityConstrainedPnPSolver(const GravityPoseConfig& config);
    GravityPoseResult solve(const VisualObservation& visual, const GravityNormalPrior& prior,
                            double exposure_time, int min_support=4) const;
    RadarPoseResult solveWithRadar(const VisualObservation&, const GravityNormalPrior&,
        const RadarHeightPrior&, const RadarPoseConfig&, double exposure_time, int min_support=4) const;
private:
    RadarPoseResult solveImpl(const VisualObservation&, const GravityNormalPrior&, double, int,
        const RadarHeightPrior*, const RadarPoseConfig*) const;
    GravityPoseConfig cfg_;
};
} // namespace gpnp
