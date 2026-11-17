#pragma once
#include "common/VisualObservation.hpp"
#include <cmath>
#include <iomanip>
#include <ostream>

namespace gpnp {
inline void visualJsonString(std::ostream& s,const std::string& value) {
    s<<'"';for(unsigned char c:value) {
        if(c=='"'||c=='\\')s<<'\\'<<c;
        else if(c=='\n')s<<"\\n";else if(c=='\r')s<<"\\r";else if(c=='\t')s<<"\\t";
        else if(c>=32)s<<c;
    }s<<'"';
}
inline void visualJsonNumber(std::ostream& s,double v){if(std::isfinite(v))s<<v;else s<<"null";}
template<class Derived> inline void visualJsonMatrix(std::ostream& s,const Eigen::MatrixBase<Derived>& m) {
    s<<'[';bool first=true;for(int i=0;i<m.rows();++i)for(int j=0;j<m.cols();++j) {
        if(!first)s<<',';
        first=false;visualJsonNumber(s,m(i,j));
    }s<<']';
}
inline void writeVisualObservation(std::ostream& s,const VisualObservation& o) {
    s<<std::setprecision(17)<<"{\"schema\":1,\"frame\":"<<o.frame_id<<",\"t_sample\":";
    visualJsonNumber(s,o.t_sample);s<<",\"t_arrival\":";visualJsonNumber(s,o.t_arrival);
    s<<",\"timestamp_quality\":";visualJsonString(s,o.timestamp_quality);
    s<<",\"strategy\":";visualJsonString(s,o.strategy);s<<",\"model_id\":";visualJsonString(s,o.model_id);
    s<<",\"model_frame\":";visualJsonString(s,o.model_frame);
    s<<",\"K\":";visualJsonMatrix(s,o.camera_matrix);
    s<<",\"is_class1\":"<<o.is_class1<<",\"identity_eligible\":"<<o.identity_eligible
     <<",\"image_width\":"<<o.image_width<<",\"image_height\":"<<o.image_height
     <<",\"evaluated\":"<<o.evaluated<<",\"ippe_complete\":"<<o.ippe_complete<<",\"ambiguous\":"<<o.ambiguous;
    s<<",\"status\":";visualJsonString(s,o.status);
    s<<",\"error_ratio\":";visualJsonNumber(s,o.error_ratio);s<<",\"error_gap_px\":";visualJsonNumber(s,o.error_gap_px);
    s<<",\"rotation_gap_deg\":";visualJsonNumber(s,o.rotation_gap_deg);
    s<<",\"normal_gap_deg\":";visualJsonNumber(s,o.normal_gap_deg);
    s<<",\"translation_gap_ratio\":";visualJsonNumber(s,o.translation_gap_ratio);
    s<<",\"alternate_index\":"<<o.alternate_index<<",\"point_ids\":[";
    for(size_t i=0;i<o.point_ids.size();++i){if(i)s<<',';s<<o.point_ids[i];}
    s<<"],\"image_points_px\":[";
    for(size_t i=0;i<o.image_points.size();++i){if(i)s<<',';s<<'[';visualJsonNumber(s,o.image_points[i].x);s<<',';visualJsonNumber(s,o.image_points[i].y);s<<']';}
    s<<"],\"object_points_mm\":[";
    for(size_t i=0;i<o.object_points_mm.size();++i){if(i)s<<',';visualJsonMatrix(s,o.object_points_mm[i]);}
    s<<"],\"covariance_convention\":\"[dt_camera_mm,left_dtheta_camera_rad],row_major\",\"candidates\":[";
    for(size_t i=0;i<o.candidates.size();++i) {
        if(i)s<<',';
        const auto& c=o.candidates[i];s<<"{\"index\":"<<i<<",\"source\":";visualJsonString(s,c.source);
        s<<",\"R\":";visualJsonMatrix(s,c.R);s<<",\"t_mm\":";visualJsonMatrix(s,c.t_mm);
        s<<",\"mean_error_px\":";visualJsonNumber(s,c.mean_error_px);s<<",\"rms_error_px\":";visualJsonNumber(s,c.rms_error_px);
        s<<",\"support\":"<<c.support<<",\"prior_adjusted\":"<<c.prior_adjusted<<",\"covariance_valid\":"<<c.covariance_valid;
        s<<",\"covariance\":";visualJsonMatrix(s,c.covariance);s<<'}';
    }
    s<<"],\"diagnostics\":[";
    for(size_t i=0;i<o.diagnostics.size();++i) {
        if(i)s<<',';
        const auto& d=o.diagnostics[i];s<<"{\"source\":";visualJsonString(s,d.source);
        s<<",\"status\":";visualJsonString(s,d.status);s<<",\"raw_error_px\":";visualJsonNumber(s,d.raw_error_px);
        s<<",\"final_error_px\":";visualJsonNumber(s,d.final_error_px);s<<",\"raw_R\":";
        if(d.status=="base_quality_failed")s<<"null";else visualJsonMatrix(s,d.raw_R);
        s<<",\"raw_t_mm\":";
        if(d.status=="base_quality_failed")s<<"null";else visualJsonMatrix(s,d.raw_t_mm);
        s<<'}';
    }s<<"]}\n";
}
inline void writeVisualFeedback(std::ostream& s,const VisualFeedback& f) {
    s<<std::setprecision(17)<<"{\"frame\":"<<f.frame_id<<",\"candidate_index\":"<<f.candidate_index
        <<",\"accepted\":"<<f.accepted<<",\"status\":";visualJsonString(s,f.status);
    s<<",\"nis\":";visualJsonNumber(s,f.nis);s<<",\"score_gap\":";visualJsonNumber(s,f.score_gap);s<<"}\n";
}
} // namespace gpnp
