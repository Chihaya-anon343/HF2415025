#pragma once
#include "common/VisualObservation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpnp {
inline VisualOutputConfig readVisualOutputConfig(const cv::FileNode& node) {
    VisualOutputConfig c;
    if(node.empty())return c;
    if(!node.isMap())throw std::invalid_argument("vision_output must be an object");
    if(!node["mode"].empty()) {
        const std::string mode=node["mode"];
        if(mode!="standalone" && mode!="fusion")throw std::invalid_argument("vision_output.mode must be standalone or fusion");
        c.fusion_mode=mode=="fusion";
    }
    auto boolean=[&](const char* key,bool& value) {
        const auto n=node[key];if(n.empty())return;
        if(!n.isInt() || ((int)n!=0 && (int)n!=1))throw std::invalid_argument(std::string("vision_output.")+key+" must be boolean");
        value=(int)n!=0;
    };
    boolean("export_candidates",c.export_candidates);boolean("estimate_covariance",c.estimate_covariance);
    auto number=[&](const char* key,double& value,double lo,double hi) {
        const auto n=node[key];if(n.empty())return;
        if(!n.isInt()&&!n.isReal())throw std::invalid_argument(std::string("vision_output.")+key+" must be numeric");
        value=(double)n;
        if(!std::isfinite(value)||value<lo||value>hi)throw std::invalid_argument(std::string("vision_output.")+key+" out of range");
    };
    number("pixel_sigma_floor",c.pixel_sigma_floor,0.05,10);
    number("ambiguity_ratio",c.ambiguity_ratio,0.01,1);
    number("ambiguity_gap_px",c.ambiguity_gap_px,0,8);
    number("separation_deg",c.separation_deg,0.1,180);
    number("separation_translation_ratio",c.separation_translation_ratio,0.001,10);
    if(c.fusion_mode){c.export_candidates=true;if(!c.estimate_covariance)throw std::invalid_argument("fusion mode requires covariance estimation");}
    return c;
}
inline double visualRotationDistance(const Eigen::Matrix3d& a,const Eigen::Matrix3d& b) {
    return Eigen::AngleAxisd(a*b.transpose()).angle()*180.0/CV_PI;
}
inline void assessVisualAmbiguity(VisualObservation& o,const VisualOutputConfig& cfg) {
    o.ambiguous=false;o.alternate_index=-1;
    if(o.candidates.empty()){o.status="no_candidates";return;}
    const auto& a=o.candidates.front();
    for(size_t i=1;i<o.candidates.size();++i) {
        const auto& b=o.candidates[i];
        const double angle=visualRotationDistance(a.R,b.R);
        const double translation=(a.t_mm-b.t_mm).norm()/std::max(a.t_mm.norm(),1.0);
        if(angle<cfg.separation_deg && translation<cfg.separation_translation_ratio)continue;
        o.alternate_index=static_cast<int>(i);o.rotation_gap_deg=angle;
        o.normal_gap_deg=std::acos(std::clamp(a.R.col(2).dot(b.R.col(2)),-1.,1.))*180/CV_PI;
        o.translation_gap_ratio=translation;
        o.error_gap_px=b.mean_error_px-a.mean_error_px;
        // Two near-zero errors do not constitute evidence for either branch.
        o.error_ratio=b.mean_error_px>1e-9?a.mean_error_px/b.mean_error_px:1.0;
        o.ambiguous=o.error_ratio>=cfg.ambiguity_ratio || o.error_gap_px<=cfg.ambiguity_gap_px;
        break;
    }
    o.status=!o.ippe_complete?"incomplete":o.ambiguous?"ambiguous":
        o.candidates.size()==1?"single_admissible":"unambiguous";
}
} // namespace gpnp
