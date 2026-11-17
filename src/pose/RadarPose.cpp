#include "common/RadarPose.hpp"
#include <cmath>
#include <stdexcept>

namespace gpnp {
void validateRadarPoseConfig(const RadarPoseConfig& c) {
    if(c.measurement_type!="vertical_height" ||
       !std::isfinite(c.height_sigma_m)||c.height_sigma_m<=0||
       !std::isfinite(c.max_age_s)||c.max_age_s<=0||c.max_age_s>1||
       !std::isfinite(c.max_tilt_deg)||c.max_tilt_deg<=0||c.max_tilt_deg>=89||
       !std::isfinite(c.age_noise_m_s)||c.age_noise_m_s<0||
       !std::isfinite(c.max_height_residual_sigma)||c.max_height_residual_sigma<=0)
        throw std::invalid_argument("Invalid radar_pose: first version requires vertical_height, no tilt correction and zero offsets");
}
RadarPoseConfig readRadarPoseConfig(const cv::FileNode& n) {
    RadarPoseConfig c;if(n.empty())return c;
    if(!n.isMap())throw std::invalid_argument("radar_pose must be an object");
    if(!n["enabled"].empty()) {
        auto v=n["enabled"];if(!v.isInt()||(int(v)!=0&&int(v)!=1))throw std::invalid_argument("radar_pose.enabled must be boolean");
        c.enabled=int(v)!=0;
    }
    if(!n["measurement_type"].empty()) {
        if(!n["measurement_type"].isString())throw std::invalid_argument("radar measurement_type must be a string");
        c.measurement_type=std::string(n["measurement_type"]);
    }
    for(auto item:{std::pair<const char*,double*>{"height_sigma_m",&c.height_sigma_m},{"max_age_s",&c.max_age_s},
        {"max_tilt_deg",&c.max_tilt_deg},{"age_noise_m_s",&c.age_noise_m_s},{"max_height_residual_sigma",&c.max_height_residual_sigma}}) {
        auto v=n[item.first];if(v.empty())continue;
        if(!v.isInt()&&!v.isReal())throw std::invalid_argument(std::string("radar_pose.")+item.first+" must be numeric");
        *item.second=double(v);
    }
    if(!n["apply_tilt_compensation"].empty() && (!n["apply_tilt_compensation"].isInt() || int(n["apply_tilt_compensation"])!=0))
        throw std::invalid_argument("vertical_height must not apply extra cosine compensation");
    for(auto key:{"camera_to_radar_m","height_bias_m"}) {
        auto v=n[key];if(v.empty())continue;
        if(std::string(key)=="height_bias_m") {
            if((!v.isInt()&&!v.isReal()) || double(v)!=0)throw std::invalid_argument("Radar offsets must be zero in this version");
        } else {
            if(!v.isSeq()||v.size()!=3)throw std::invalid_argument("camera_to_radar_m requires three zeros");
            for(auto x:v)if((!x.isInt()&&!x.isReal())||double(x)!=0)throw std::invalid_argument("Radar offsets must be zero in this version");
        }
    }
    validateRadarPoseConfig(c);return c;
}
}
