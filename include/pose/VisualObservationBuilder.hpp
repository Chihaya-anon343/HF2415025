#pragma once
#include "common/VisualObservation.hpp"
namespace gpnp {
void finalizeVisualObservation(VisualObservation& observation,
    const Eigen::Matrix3d& K,const VisualOutputConfig& config);
}
