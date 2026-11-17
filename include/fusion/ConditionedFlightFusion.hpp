#pragma once
#include "common/RadarPose.hpp"
#include "input/SensorTypes.hpp"
#include <memory>

namespace gpnp::fusion {
// Synchronous flight replay. Independent VisualObservation updates remain separate.
class ConditionedFlightFusion {
public:
    ConditionedFlightFusion(const cv::FileNode& eskf_config,const std::string& output);
    ~ConditionedFlightFusion();
    void step(const input::SensorPacket&, const RadarPoseResult&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
