#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gpnp {
// Parameters for an already identified normalized BC contour. No physical model.
struct NormalizedCornerRecoveryConfig {
    bool enabled{false}, reclassify_final_h{true}, repair_enabled{true};
    int min_pose_inliers{4};
    double residual_mad_scale{3.0};
    double edge_canny_low{20.0}, edge_canny_high{60.0};
};
inline void validateNormalizedCornerRecovery(const NormalizedCornerRecoveryConfig& c) {
    if(c.min_pose_inliers<4 || c.min_pose_inliers>10 ||
       !std::isfinite(c.residual_mad_scale) || c.residual_mad_scale<0 || c.residual_mad_scale>6 ||
       !std::isfinite(c.edge_canny_low) || !std::isfinite(c.edge_canny_high) ||
       c.edge_canny_low<=0 || c.edge_canny_high<=c.edge_canny_low || c.edge_canny_high>255)
        throw std::invalid_argument("normalized corner recovery parameters invalid");
}
inline NormalizedCornerRecoveryConfig readNormalizedCornerRecovery(
        const cv::FileNode& node, NormalizedCornerRecoveryConfig c={}) {
    if(node.isNone())return c;
    if(!node.isMap())throw std::invalid_argument("corner recovery must be an object");
    for(auto item:{std::pair<const char*,bool*>{"enabled",&c.enabled},
        {"reclassify_final_h",&c.reclassify_final_h},{"repair_enabled",&c.repair_enabled}}) {
        auto n=node[item.first];if(n.isNone())continue;
        if(!n.isInt() || (int(n)!=0 && int(n)!=1))throw std::invalid_argument("corner recovery boolean expected");
        *item.second=int(n)!=0;
    }
    auto n=node["min_pose_inliers"];
    if(!n.isNone()) {
        if(!n.isInt())throw std::invalid_argument("corner recovery integer expected");
        c.min_pose_inliers=int(n);
    }
    for(auto item:{std::pair<const char*,double*>{"residual_mad_scale",&c.residual_mad_scale},
        {"edge_canny_low",&c.edge_canny_low},{"edge_canny_high",&c.edge_canny_high}}) {
        auto v=node[item.first];if(v.isNone())continue;
        if(!v.isInt()&&!v.isReal())throw std::invalid_argument("corner recovery number expected");
        *item.second=double(v);
    }
    validateNormalizedCornerRecovery(c);return c;
}
struct DualCornerRecoveryConfig {
    bool enabled{false};
    NormalizedCornerRecoveryConfig class0, class1;
    DualCornerRecoveryConfig() {
        class0.enabled=class1.enabled=true;
        class1.edge_canny_low=10;class1.edge_canny_high=30;
    }
};
} // namespace gpnp
