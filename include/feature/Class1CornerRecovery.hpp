#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace gpnp {
struct Class1CornerRecoveryConfig {
    bool enabled{false};
    bool repair_enabled{true};
    int min_reliable_points{8};
    int min_pose_inliers{8}; // separate PnP acceptance; missing field preserves old behavior
    double min_model_coverage{0.60};
    double residual_ratio{0.045};
    double residual_mad_scale{3.0};
    double edge_canny_low{10.0}, edge_canny_high{30.0};
    double edge_band_ratio{0.018}, edge_radius_ratio{0.16};
    double edge_max_rms_ratio{0.015};
};
struct CornerRecoveryObservation {
    int id{-1};
    cv::Point2f original, predicted, final_point;
    double residual{-1};
    double threshold_px{0};
    bool has_prediction{false};
    std::string status{"unexamined"};
};
using Class1CornerObservation = CornerRecoveryObservation;

inline bool validClass1CornerRecoveryConfig(const Class1CornerRecoveryConfig& c) {
    return c.min_reliable_points>=8 && c.min_pose_inliers>=4 &&
        c.min_pose_inliers<=c.min_reliable_points && std::isfinite(c.min_model_coverage) &&
        c.min_model_coverage>0 && c.min_model_coverage<=1 &&
        std::isfinite(c.residual_ratio) && c.residual_ratio>0 && c.residual_ratio<=0.1 &&
        std::isfinite(c.residual_mad_scale) && c.residual_mad_scale>=0 && c.residual_mad_scale<=6 &&
        std::isfinite(c.edge_canny_low) && std::isfinite(c.edge_canny_high) &&
        c.edge_canny_low>0 && c.edge_canny_high>c.edge_canny_low && c.edge_canny_high<=255 &&
        std::isfinite(c.edge_band_ratio) && c.edge_band_ratio>=0.005 && c.edge_band_ratio<=0.05 &&
        std::isfinite(c.edge_radius_ratio) && c.edge_radius_ratio>=0.05 && c.edge_radius_ratio<=0.3 &&
        std::isfinite(c.edge_max_rms_ratio) && c.edge_max_rms_ratio>0 && c.edge_max_rms_ratio<=0.03;
}

inline double class1RecoveryMedian(std::vector<double> values) {
    const size_t n=values.size();
    std::sort(values.begin(),values.end());
    return n%2 ? values[n/2] : .5*(values[n/2-1]+values[n/2]);
}
struct Class1CornerRecoveryResult {
    bool accepted{false};
    std::string status{"not_run"};
    std::vector<cv::Point2f> points;
    std::vector<int> ids;
    std::vector<Class1CornerObservation> observations;
};

// A repair is an intersection of two independently observed edge segments.
// The homography only supplies search directions and is never an observation.
inline bool recoverClass1EdgeCorner(const cv::Mat& edges, const cv::Point2f& p,
                                    const cv::Point2f& prev, const cv::Point2f& next,
                                    double span, cv::Point2f& out,
                                    const Class1CornerRecoveryConfig& cfg = {}) {
    const double radius=std::clamp(span*cfg.edge_radius_ratio, 10.0, 90.0);
    const double band=std::clamp(span*cfg.edge_band_ratio, 2.0, 20.0);
    const double max_rms=std::clamp(span*cfg.edge_max_rms_ratio, 2.5, 8.0);
    cv::Vec4f lines[2];
    const cv::Point2f targets[2]={prev,next};
    for(int k=0;k<2;++k) {
        cv::Point2f direction=targets[k]-p;
        double length=cv::norm(direction);
        if(length<8) return false;
        direction*=static_cast<float>(1.0/length);
        std::vector<cv::Point2f> samples;
        const int x0=std::max(0,(int)std::floor(p.x-radius));
        const int x1=std::min(edges.cols-1,(int)std::ceil(p.x+radius));
        const int y0=std::max(0,(int)std::floor(p.y-radius));
        const int y1=std::min(edges.rows-1,(int)std::ceil(p.y+radius));
        for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
            if(!edges.at<uchar>(y,x)) continue;
            cv::Point2f d=cv::Point2f((float)x,(float)y)-p;
            double along=d.dot(direction), across=std::abs(d.x*direction.y-d.y*direction.x);
            if(along>=3 && along<=std::min(radius,length*0.8) && across<=band)
                samples.emplace_back((float)x,(float)y);
        }
        if(samples.size()<6) return false;
        cv::fitLine(samples,lines[k],cv::DIST_HUBER,0,0.01,0.01);
        cv::Point2f v(lines[k][0],lines[k][1]);
        if(std::abs(v.dot(direction))<0.92) return false;
        double low=1e9, high=-1e9, sum=0;
        for(const auto& q:samples) {
            cv::Point2f d=q-cv::Point2f(lines[k][2],lines[k][3]);
            double t=d.dot(v), e=d.x*v.y-d.y*v.x;
            low=std::min(low,t); high=std::max(high,t); sum+=e*e;
        }
        if(high-low<6 || std::sqrt(sum/samples.size())>max_rms) return false;
    }
    cv::Point2f a(lines[0][2],lines[0][3]), b(lines[1][2],lines[1][3]);
    cv::Point2f u(lines[0][0],lines[0][1]), v(lines[1][0],lines[1][1]);
    double det=u.x*v.y-u.y*v.x;
    if(std::abs(det)<0.25) return false;
    cv::Point2f d=b-a;
    out=a+u*static_cast<float>((d.x*v.y-d.y*v.x)/det);
    return std::isfinite(out.x) && std::isfinite(out.y) &&
           out.x>=0 && out.y>=0 && out.x<edges.cols && out.y<edges.rows &&
           cv::norm(out-p)<=std::min(radius,band*2.0+2.0);
}

inline Class1CornerRecoveryResult recoverClass1Corners(
    const cv::Mat& gray, const std::vector<cv::Point2f>& observed,
    const std::vector<cv::Point2f>& model, const Class1CornerRecoveryConfig& cfg) {
    Class1CornerRecoveryResult r;
    for(size_t i=0;i<observed.size();++i) {
        Class1CornerObservation o; o.id=(int)i; o.original=observed[i]; o.final_point=observed[i];
        r.observations.push_back(o);
    }
    if(gray.empty() || observed.size()!=model.size() || model.size()<8 ||
       !validClass1CornerRecoveryConfig(cfg) || cfg.min_reliable_points>(int)model.size()) {
        r.status="invalid_input"; return r;
    }
    for(size_t i=0;i<model.size();++i) {
        if(!std::isfinite(model[i].x) || !std::isfinite(model[i].y) ||
           !std::isfinite(observed[i].x) || !std::isfinite(observed[i].y)) {
            r.status="invalid_input";return r;
        }
        for(size_t j=0;j<i;++j) if(cv::norm(observed[i]-observed[j])<1e-3 || cv::norm(model[i]-model[j])<1e-6) {
            r.status="duplicate_points";return r;
        }
    }
    cv::Rect bbox=cv::boundingRect(observed);
    double span=std::max(bbox.width,bbox.height);
    double threshold=std::clamp(span*cfg.residual_ratio,3.0,25.0);
    cv::Mat mask, h;
    try {
        // Keep the global OpenCV random state unchanged for the rest of the pipeline.
        uint64 saved=cv::theRNG().state;
        cv::theRNG().state=0x1341ab57;
        try { h=cv::findHomography(model,observed,cv::RANSAC,threshold,mask,2000,0.995); }
        catch(...) { cv::theRNG().state=saved; throw; }
        cv::theRNG().state=saved;
    } catch(const cv::Exception&) { r.status="homography_exception"; return r; }
    if(h.empty() || mask.total()!=model.size()) {r.status="homography_failed";return r;}
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(model,projected,h);
    std::vector<double> residuals;
    for(size_t i=0;i<model.size();++i) {
        if(!std::isfinite(projected[i].x) || !std::isfinite(projected[i].y)) {
            r.status="invalid_projection"; return r;
        }
        auto& o=r.observations[i]; o.has_prediction=true; o.predicted=projected[i];
        o.residual=cv::norm(observed[i]-projected[i]);
        residuals.push_back(o.residual);
    }
    // RANSAC's mask selects a fitting seed, not the final observation set.
    // Reclassify every original observation against the final H. A robust
    // noise floor tolerates modest shape/model differences without forcing
    // any particular template ID to be repaired or retained.
    const double median=class1RecoveryMedian(residuals);
    std::vector<double> deviations;
    for(double value:residuals) deviations.push_back(std::abs(value-median));
    threshold=std::max(threshold,median+cfg.residual_mad_scale*1.4826*class1RecoveryMedian(deviations));
    for(auto& o:r.observations) {
        o.threshold_px=threshold;
        o.status=o.residual<=threshold ? "kept" : "rejected";
    }
    // Require broad geometric support before using a hypothesis to guide repairs.
    std::vector<cv::Point2f> support, hull_all, hull_support;
    for(size_t i=0;i<model.size();++i) if(r.observations[i].status=="kept") support.push_back(model[i]);
    if((int)support.size()<cfg.min_reliable_points) {r.status="insufficient_support";return r;}
    cv::convexHull(model,hull_all); cv::convexHull(support,hull_support);
    const double all_area=std::abs(cv::contourArea(hull_all));
    if(all_area<=1e-6 || std::abs(cv::contourArea(hull_support))/all_area<cfg.min_model_coverage) {
        r.status="insufficient_coverage"; return r;
    }
    if(cfg.repair_enabled && support.size()<model.size()) {
        cv::Mat edges; cv::Canny(gray,edges,cfg.edge_canny_low,cfg.edge_canny_high);
        for(size_t i=0;i<model.size();++i) if(r.observations[i].status=="rejected") {
            cv::Point2f fixed;
            if(recoverClass1EdgeCorner(edges,projected[i],projected[(i+model.size()-1)%model.size()],
                                       projected[(i+1)%model.size()],span,fixed,cfg) &&
               cv::norm(fixed-projected[i])<=threshold) {
                r.observations[i].final_point=fixed; r.observations[i].status="repaired";
            }
        }
    }
    // Accept repaired observations only if the combined set remains consistent.
    std::vector<cv::Point2f> fit_model, fit_image, fit_projection;
    for(const auto& o:r.observations) if(o.status!="rejected") {
        fit_model.push_back(model[o.id]); fit_image.push_back(o.final_point);
    }
    cv::Mat refit=cv::findHomography(fit_model,fit_image,0);
    bool consistent=!refit.empty();
    if(consistent) {
        cv::perspectiveTransform(fit_model,fit_projection,refit);
        for(size_t i=0;i<fit_model.size();++i)
            if(!std::isfinite(fit_projection[i].x) || !std::isfinite(fit_projection[i].y) ||
               cv::norm(fit_projection[i]-fit_image[i])>threshold) consistent=false;
    }
    if(!consistent) for(auto& o:r.observations) if(o.status=="repaired") {
        o.status="rejected"; o.final_point=o.original;
    }
    for(const auto& o:r.observations) if(o.status!="rejected") {
        r.ids.push_back(o.id); r.points.push_back(o.final_point);
    }
    r.accepted=true; r.status="accepted"; return r;
}
} // namespace gpnp
