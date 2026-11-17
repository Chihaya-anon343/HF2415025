#pragma once
#include "common/GravityPose.hpp"
#include "common/RadarPose.hpp"
#include "common/VisualObservation.hpp"
#include "utils/VisualObservationIO.hpp"
#include "utils/PoseAxisDisplay.hpp"
#include <opencv2/calib3d.hpp>

namespace gpnp {
inline void writeGravityPose(std::ostream& s,int64_t frame,const GravityPoseResult& p,const RadarPoseResult* radar=nullptr) {
    s<<std::setprecision(17)<<"{\"schema\":1,\"frame\":"<<frame<<",\"t_sample\":";visualJsonNumber(s,p.t_sample);
    s<<",\"success\":"<<p.success<<",\"uses_inertial_prior\":true,\"status\":";visualJsonString(s,p.status);
    s<<",\"method\":";visualJsonString(s,p.method);s<<",\"prior_source\":";visualJsonString(s,p.prior_source);
    s<<",\"prior_valid\":"<<p.prior_valid<<",\"prior_normal_camera\":";visualJsonMatrix(s,p.prior_normal_camera);
    s<<",\"prior_sigma_deg\":";visualJsonNumber(s,p.prior_sigma_deg);s<<",\"prior_age_s\":";visualJsonNumber(s,p.prior_age_s);
    s<<",\"mean_error_px\":";visualJsonNumber(s,p.mean_error_px);s<<",\"rms_error_px\":";visualJsonNumber(s,p.rms_error_px);
    s<<",\"normal_residual_deg\":";visualJsonNumber(s,p.normal_residual_deg);
    s<<",\"support\":"<<p.support<<",\"iterations\":"<<p.iterations<<",\"solve_ms\":";visualJsonNumber(s,p.time_ms);
    s<<",\"R\":";visualJsonMatrix(s,p.R);s<<",\"t_mm\":";visualJsonMatrix(s,p.t_mm);
    s<<",\"covariance_valid\":"<<p.covariance_valid<<",\"covariance_kind\":";visualJsonString(s,p.covariance_kind);
    s<<",\"posterior_covariance\":";visualJsonMatrix(s,p.posterior_covariance);
    if(radar) {
        s<<",\"uses_radar_prior\":"<<radar->uses_radar_prior<<",\"radar_sample_id\":"<<radar->radar_sample_id;
        s<<",\"radar_height_m\":";visualJsonNumber(s,radar->radar_height_m);
        s<<",\"radar_age_s\":";visualJsonNumber(s,radar->radar_age_s);
        s<<",\"height_sigma_m\":";visualJsonNumber(s,radar->height_sigma_m);
        s<<",\"predicted_height_m\":";visualJsonNumber(s,radar->predicted_height_m);
        s<<",\"height_residual_m\":";visualJsonNumber(s,radar->height_residual_m);
        s<<",\"tilt_deg\":";visualJsonNumber(s,radar->tilt_deg);
    }
    s<<"}\n";
}
// This is a separate, explicitly labelled preview. Never overwrite pure visual axes.
inline cv::Mat gravityPosePreview(const cv::Mat& input,const VisualObservation& visual,const GravityPoseResult& p,const std::string& label="IMU-conditioned ") {
    cv::Mat image;if(input.channels()==1)cv::cvtColor(input,image,cv::COLOR_GRAY2BGR);else image=input.clone();
    for(size_t i=0;i<visual.image_points.size();++i) {
        const auto u=visual.image_points[i];cv::circle(image,u,2,{0,255,255},1,cv::LINE_AA);
        const int id=i<visual.point_ids.size()?visual.point_ids[i]:int(i);
        cv::putText(image,std::to_string(id),u+cv::Point2f(3,-3),cv::FONT_HERSHEY_SIMPLEX,.35,{0,255,255},1,cv::LINE_AA);
    }
    if(p.success) {
        cv::Mat K(3,3,CV_64F),R(3,3,CV_64F),t(3,1,CV_64F),rv;
        for(int i=0;i<3;++i){t.at<double>(i)=p.t_mm(i);for(int j=0;j<3;++j){K.at<double>(i,j)=visual.camera_matrix(i,j);R.at<double>(i,j)=p.R(i,j);}}
        cv::Rodrigues(R,rv);
        const double length=utils::poseAxisDisplayLength(p.t_mm.norm());
        std::vector<cv::Point3d> model{{0,0,0},{length,0,0},{0,length,0},{0,0,utils::upNormalZ(length,p.t_mm.z())}};
        std::vector<cv::Point2d> uv;cv::projectPoints(model,rv,t,K,cv::Mat(),uv);
        for(int axis=1;axis<=2;++axis)if(std::abs(uv[axis].x)<1e7&&std::abs(uv[axis].y)<1e7)
            cv::arrowedLine(image,uv[0],uv[axis],axis==1?cv::Scalar(0,0,255):cv::Scalar(0,255,0),2,cv::LINE_AA);
        utils::drawUpNormal(image,uv[0],uv[3],2);
    }
    cv::putText(image,label+p.method+": "+p.status,{12,28},cv::FONT_HERSHEY_SIMPLEX,.6,{0,0,0},3,cv::LINE_AA);
    cv::putText(image,label+p.method+": "+p.status,{12,28},cv::FONT_HERSHEY_SIMPLEX,.6,{0,255,255},1,cv::LINE_AA);
    return image;
}
} // namespace gpnp
