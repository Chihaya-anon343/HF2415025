#pragma once
#include <string>

namespace gpnp {
struct GravityActivation {
    bool enabled{false};
    std::string reason;
};
inline GravityActivation resolveGravityActivation(bool requested,bool imu_input_enabled,
                                                   bool imu_source_ready,bool mono_mode=true) {
    if(!requested)return {false,"disabled_by_config"};
    if(!imu_input_enabled)return {false,"imu_input_disabled"};
    if(!imu_source_ready)return {false,"imu_source_not_ready"};
    if(!mono_mode)return {false,"mono_required"};
    return {true,"enabled"};
}
} // namespace gpnp
