#pragma once
#include "common/Types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpnp {
// Validate a one-to-one canonical-ID map without changing either physical model.
inline std::vector<int> monoDualCanonicalMap(const std::vector<Eigen::Vector3d>& source,
                                            const std::vector<Eigen::Vector3d>& target) {
    if(source.size()!=10 || target.size()!=source.size())return {};
    for(const auto& p:source)if(!p.allFinite())return {};
    for(const auto& p:target)if(!p.allFinite())return {};
    auto normalize=[](const std::vector<Eigen::Vector3d>& points) {
        Eigen::Vector2d low=points.front().head<2>(),high=low;
        for(const auto& p:points) {low=low.cwiseMin(p.head<2>());high=high.cwiseMax(p.head<2>());}
        double span=(high-low).maxCoeff();std::vector<Eigen::Vector2d> out;
        if(!std::isfinite(span)||span<1e-9)return out;
        for(const auto& p:points)out.push_back((p.head<2>()-(low+high)*.5)/span);
        return out;
    };
    const auto a=normalize(source),b=normalize(target);
    if(a.size()!=source.size() || b.size()!=target.size())return {};
    std::vector<int> map;std::vector<bool> used(b.size(),false);
    for(const auto& p:a) {
        double best=.05;int id=-1;
        for(size_t i=0;i<b.size();++i)if((p-b[i]).norm()<best){best=(p-b[i]).norm();id=int(i);}
        if(id<0 || used[id])return {};
        used[id]=true;map.push_back(id);
    }
    // The identity matcher uses a contour cycle; preserve its handedness.
    for(size_t i=0;i<map.size();++i)if(map[(i+1)%map.size()]!=(map[i]+1)%int(map.size()))return {};
    return map;
}
inline void remapMonoDualClass0(PipelineResult& r,const std::vector<int>& map) {
    auto id=[&](int old){return old>=0&&old<int(map.size())?map[old]:-1;};
    for(auto& m:r.good_matches)m.trainIdx=id(m.trainIdx);
    for(auto& i:r.bc_rejected_ids)i=id(i);
    for(auto& i:r.bc_recovery_base_ids)i=id(i);
    for(auto& o:r.dual_corner_observations)o.point.id=id(o.point.id);
}
inline void appendMonoDualDiagnostics(std::vector<DualCornerRecoveryObservation>& output,
        const PipelineResult& local,const RoiRect& roi,const char* stage) {
    const cv::Point2f offset(float(roi.x),float(roi.y));
    for(auto record:local.dual_corner_observations) {
        record.stage=stage;
        if(record.point.status!="not_observed") {
            record.point.original+=offset;record.point.final_point+=offset;
            if(record.point.has_prediction)record.point.predicted+=offset;
        }
        output.push_back(std::move(record));
    }
}
inline bool monoDualMatchesValid(const PipelineResult& r,
                                const std::vector<Eigen::Vector3d>& model) {
    if(!r.success || r.pts_left_match.size()!=r.good_matches.size() ||
       r.good_matches.size()<size_t(std::max(4,r.pose_min_inliers)))return false;
    std::vector<bool> used(model.size(),false);
    for(size_t i=0;i<r.good_matches.size();++i) {
        const auto& m=r.good_matches[i];const auto p=r.pts_left_match[i];
        if(m.queryIdx!=int(i)||m.trainIdx<0||m.trainIdx>=int(model.size())||
           used[m.trainIdx]||!model[m.trainIdx].allFinite()||
           !std::isfinite(p.x)||!std::isfinite(p.y))return false;
        used[m.trainIdx]=true;
    }
    return true;
}
inline MonoDualRoiConfig readMonoDualRoiConfig(const cv::FileNode& node) {
    MonoDualRoiConfig c;
    if (node.empty()) return c;
    if (!node.isMap()) throw std::invalid_argument("dual_roi.mono_robustness must be an object");
    auto boolean = [&](const char* key, bool& value) {
        auto n=node[key]; if(n.empty())return;
        if(!n.isInt() || (int(n)!=0 && int(n)!=1))
            throw std::invalid_argument(std::string("dual_roi.mono_robustness.")+key+" must be boolean");
        value=int(n)!=0;
    };
    auto positive = [&](const char* key, double& value) {
        auto n=node[key];if(n.empty())return;
        if(!n.isInt()&&!n.isReal())throw std::invalid_argument(std::string(key)+" must be numeric");
        value=double(n);
        if(!std::isfinite(value)||value<0)throw std::invalid_argument(std::string(key)+" must be finite and nonnegative");
    };
    boolean("enabled",c.enabled);
    boolean("size_aware_class1",c.size_aware_class1);
    boolean("joint_quality_check",c.joint_quality_check);
    positive("max_primary_error_increase_px",c.max_primary_error_increase_px);
    positive("max_group_error_px",c.max_group_error_px);
    positive("max_rotation_delta_deg",c.max_rotation_delta_deg);
    positive("max_translation_delta_ratio",c.max_translation_delta_ratio);
    auto recovery=node["corner_recovery"];
    if(!recovery.isNone()) {
        if(!recovery.isMap())throw std::invalid_argument("DualROI corner_recovery must be object");
        auto n=recovery["enabled"];
        if(!n.isNone()) {
            if(!n.isInt() || (int(n)!=0 && int(n)!=1))throw std::invalid_argument("DualROI recovery enabled must be boolean");
            c.corner_recovery.enabled=int(n)!=0;
        }
        c.corner_recovery.class0=readNormalizedCornerRecovery(recovery["class0"],c.corner_recovery.class0);
        c.corner_recovery.class1=readNormalizedCornerRecovery(recovery["class1"],c.corner_recovery.class1);
        if(c.corner_recovery.enabled && !c.enabled)
            throw std::invalid_argument("DualROI corner recovery requires mono_robustness.enabled");
    }
    if(c.max_rotation_delta_deg>180 || c.max_translation_delta_ratio>1)
        throw std::invalid_argument("DualROI joint pose limits out of range");
    return c;
}

inline double monoGroupReprojection(const PoseEstimate& pose,
    const std::vector<cv::Point2f>& pixels,const std::vector<Eigen::Vector3d>& model,
    const Eigen::Matrix3d& K) {
    if(!pose.success || !pose.R.allFinite() || !pose.t.allFinite() ||
       pixels.empty() || pixels.size()!=model.size())return std::numeric_limits<double>::infinity();
    double sum=0;
    for(size_t i=0;i<pixels.size();++i) {
        const Eigen::Vector3d p=pose.R*model[i]+pose.t;
        if(!p.allFinite()||p.z()<=0||!std::isfinite(pixels[i].x)||!std::isfinite(pixels[i].y))
            return std::numeric_limits<double>::infinity();
        const Eigen::Vector3d u=K*p;
        sum+=std::hypot(u.x()/u.z()-pixels[i].x,u.y()/u.z()-pixels[i].y);
    }
    return sum/pixels.size();
}

// Compare the SAME original primary observations under both poses. A lower
// pooled error on a different/larger point set is not evidence of improvement.
inline std::string checkMonoDualJoint(const PoseEstimate& primary,const PoseEstimate& joint,
    const std::vector<cv::Point2f>& primary_pixels,const std::vector<Eigen::Vector3d>& primary_model,
    const std::vector<cv::Point2f>& secondary_pixels,const std::vector<Eigen::Vector3d>& secondary_model,
    const Eigen::Matrix3d& K,const MonoDualRoiConfig& cfg) {
    const double before=monoGroupReprojection(primary,primary_pixels,primary_model,K);
    const double after=monoGroupReprojection(joint,primary_pixels,primary_model,K);
    const double secondary=monoGroupReprojection(joint,secondary_pixels,secondary_model,K);
    if(!std::isfinite(before)||!std::isfinite(after)||!std::isfinite(secondary))return "nonfinite_or_depth";
    if(after>before+cfg.max_primary_error_increase_px)return "primary_reprojection_worse";
    if(after>cfg.max_group_error_px || secondary>cfg.max_group_error_px)return "group_reprojection";
    const double cosine=std::clamp(((joint.R*primary.R.transpose()).trace()-1)*.5,-1.,1.);
    if(std::acos(cosine)*180/CV_PI>cfg.max_rotation_delta_deg)return "rotation_delta";
    if((joint.t-primary.t).norm()>cfg.max_translation_delta_ratio*std::max(primary.t.norm(),1.))
        return "translation_delta";
    return {};
}
} // namespace gpnp
