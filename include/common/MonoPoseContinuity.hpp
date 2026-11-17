#pragma once

#include "Types.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpnp {

inline FinalPoseAcceptanceConfig readFinalPoseAcceptanceConfig(const cv::FileNode& node) {
    FinalPoseAcceptanceConfig c;
    if(node.empty())return c;
    if(!node.isMap())throw std::invalid_argument("temporal.final_acceptance must be an object");
    auto en=node["enabled"];
    if(!en.empty()) {
        if(!en.isInt() || ((int)en!=0 && (int)en!=1))throw std::invalid_argument("final_acceptance.enabled must be boolean");
        c.enabled=(int)en!=0;
    }
    for(auto item:{std::pair<const char*,int*>{"history_max_age_frames",&c.history_max_age_frames},
                   {"reacquire_confirm_frames",&c.reacquire_confirm_frames}}) {
        auto n=node[item.first];if(n.empty())continue;
        if(!n.isInt())throw std::invalid_argument("final_acceptance requires integer frame counts");
        *item.second=(int)n;
    }
    auto m=node["max_margin"];
    if(!m.empty()) {
        if(!m.isInt() && !m.isReal())throw std::invalid_argument("final_acceptance.max_margin must be numeric");
        c.max_margin=(double)m;
    }
    if(c.history_max_age_frames<1 || c.history_max_age_frames>1000 ||
       c.reacquire_confirm_frames<2 || c.reacquire_confirm_frames>30 ||
       !std::isfinite(c.max_margin) || c.max_margin<1 || c.max_margin>2)
        throw std::invalid_argument("final_acceptance threshold out of range");
    return c;
}

struct ContinuityDecision {
    bool accepted{false}, reinitialized{false};
    std::string status{"no_candidate"};
    double rotation_deg{0}, normal_deg{0}, translation_ratio{0}, scale_ratio{1};
    int age{-1}, pending{0};
};

inline void validateFinalPoseAcceptance(const TemporalConfig& c) {
    const auto& f=c.final_acceptance;
    if(f.history_max_age_frames<1 || f.history_max_age_frames>1000 ||
       f.reacquire_confirm_frames<2 || f.reacquire_confirm_frames>30 ||
       !std::isfinite(f.max_margin) || f.max_margin<1 || f.max_margin>2 ||
       !std::isfinite(c.max_rot_deg) || !std::isfinite(c.max_trans_ratio) ||
       !std::isfinite(c.max_scale_ratio) || !std::isfinite(c.switch_margin) || c.switch_margin<1 ||
       !std::isfinite(c.tie_epsilon_px) || c.tie_epsilon_px<0)
        throw std::invalid_argument("invalid final pose continuity parameters");
}

inline bool finiteMonoPose(const PoseEstimate& p) {
    return p.success && p.R.allFinite() && p.t.allFinite() && p.t.z()>0 && p.t.norm()>1e-9 &&
        std::abs(p.R.determinant()-1)<1e-3 &&
        (p.R.transpose()*p.R-Eigen::Matrix3d::Identity()).norm()<1e-3;
}

inline ContinuityDecision checkPoseContinuity(const PoseEstimate& p, const PoseEstimate& ref,
                                             const TemporalConfig& c, bool switch_band) {
    ContinuityDecision d;
    if(!finiteMonoPose(p) || !finiteMonoPose(ref)) {d.status="invalid_pose";return d;}
    d.rotation_deg=Eigen::AngleAxisd(p.R*ref.R.transpose()).angle()*180/CV_PI;
    d.normal_deg=std::acos(std::clamp(p.R.col(2).dot(ref.R.col(2)),-1.,1.))*180/CV_PI;
    d.translation_ratio=(p.t-ref.t).norm()/ref.t.norm();
    d.scale_ratio=std::max(p.t.norm()/ref.t.norm(),ref.t.norm()/p.t.norm());
    const double margin=switch_band?std::clamp(c.switch_margin,1.,c.final_acceptance.max_margin):1.;
    if(c.max_rot_deg>0 && d.rotation_deg>c.max_rot_deg*margin) {d.status="rotation_rejected";return d;}
    if(c.max_trans_ratio>0 && d.translation_ratio>c.max_trans_ratio*margin &&
       !(c.max_scale_ratio>1 && d.scale_ratio<=c.max_scale_ratio*margin)) {
        d.status="translation_rejected";return d;
    }
    d.accepted=true;d.status="pass";return d;
}

// A frame transaction: candidate checks are pure, finish() advances recovery
// at most once per input frame. Only accepted poses replace the trusted anchor.
class MonoPoseContinuity {
public:
    void reset() {*this=MonoPoseContinuity{};}
    void beginFrame(int frame, cv::Size size, const TemporalConfig& config) {
        validateFinalPoseAcceptance(config);
        if(last_frame_>=0 && (frame<last_frame_ || size!=image_size_))reset();
        config_=config;frame_=frame;image_size_=size;last_frame_=frame;
        if(pending_count_ && frame_!=pending_frame_ && frame_-pending_frame_!=1)pending_count_=0;
    }
    bool fresh() const {
        return anchored_ && frame_>anchor_frame_ && frame_-anchor_frame_<=config_.final_acceptance.history_max_age_frames;
    }
    bool hasAnchor() const {return anchored_;}
    const PoseEstimate& anchor() const {return anchor_;}
    int age() const {return anchored_?frame_-anchor_frame_:-1;}
    bool pendingSeed(PoseSeed& seed) const {
        if(fresh() || pending_count_==0 || pending_frame_+1!=frame_)return false;
        seed.R=pending_.R;seed.t=pending_.t;return true;
    }
    ContinuityDecision check(const PoseEstimate& pose,int band) const {
        if(fresh()) {auto d=checkPoseContinuity(pose,anchor_,config_,band!=anchor_band_);d.age=age();return d;}
        ContinuityDecision d;d.accepted=finiteMonoPose(pose);d.status=d.accepted?"unreferenced":"invalid_pose";d.age=age();return d;
    }
    ContinuityDecision finish(const PoseEstimate& pose,bool class1,int band) {
        if(finished_frame_==frame_) {ContinuityDecision d;d.status="duplicate_frame_commit";return d;}
        finished_frame_=frame_;
        auto d=check(pose,band);
        if(!d.accepted) {pending_count_=0;return d;}
        if(!anchored_)d.status="initialized";
        else if(!fresh()) {
            const bool follows=pending_count_>0 && pending_class1_==class1 && pending_frame_+1==frame_ &&
                checkPoseContinuity(pose,pending_,config_,band!=pending_band_).accepted;
            pending_count_=follows?pending_count_+1:1;
            pending_=pose;pending_class1_=class1;pending_band_=band;pending_frame_=frame_;
            d.pending=pending_count_;
            if(pending_count_<config_.final_acceptance.reacquire_confirm_frames) {
                d.accepted=false;d.status="reacquire_pending";return d;
            }
            d.reinitialized=true;d.status="reinitialized";
        }
        anchor_=pose;anchor_frame_=frame_;anchor_band_=band;anchored_=true;pending_count_=0;
        return d;
    }
private:
    TemporalConfig config_;
    PoseEstimate anchor_,pending_;
    bool anchored_{false},pending_class1_{false};
    int frame_{0},last_frame_{-1},finished_frame_{-1},anchor_frame_{-1},anchor_band_{2};
    int pending_count_{0},pending_frame_{-1},pending_band_{2};
    cv::Size image_size_;
};

} // namespace gpnp
