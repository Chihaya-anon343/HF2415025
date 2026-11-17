#include "pose/GravityNormalEstimator.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpnp {
namespace {
constexpr double rad=3.14159265358979323846/180.0;
Eigen::Matrix3d rotation(const Eigen::Vector3d& v) {
    const double a=v.norm();
    return a<1e-15?Eigen::Matrix3d::Identity():Eigen::AngleAxisd(a,v/a).toRotationMatrix();
}
}
void validateGravityPoseConfig(const GravityPoseConfig& c) {
    if(c.method!="fixed_normal" && c.method!="soft_normal")throw std::invalid_argument("gravity_pose.method must be fixed_normal or soft_normal");
    for(double v:{c.pixel_sigma_px,c.huber_sigma,c.normal_sigma_floor_deg,c.plane_sigma_deg,
        c.max_prior_sigma_deg,c.max_prior_age_s,c.max_normal_residual_sigma,c.max_reprojection_px,
        c.init_window_s,c.max_imu_gap_s,c.gravity_m_s2,c.acc_norm_tolerance,c.acc_std_max,
        c.gyro_motion_max_deg_s,c.gyro_noise_rad_sqrt_s,c.gyro_bias_sigma_deg_s,c.stationary_correction_tau_s})
        if(!std::isfinite(v)||v<=0)throw std::invalid_argument("gravity_pose numeric settings must be finite and positive");
    if(c.max_iterations<1||c.max_iterations>100||c.min_init_samples<2||c.min_init_samples>10000 ||
       c.max_prior_sigma_deg>=45 || c.normal_sigma_floor_deg>=c.max_prior_sigma_deg || c.plane_sigma_deg>=45)
        throw std::invalid_argument("gravity_pose iteration/sample/uncertainty settings out of range");
}
GravityPoseConfig readGravityPoseConfig(const cv::FileNode& n) {
    GravityPoseConfig c;if(n.empty())return c;
    if(!n.isMap())throw std::invalid_argument("gravity_pose must be an object");
    if(!n["enabled"].empty()) {
        if(!n["enabled"].isInt() || (int(n["enabled"])!=0 && int(n["enabled"])!=1))
            throw std::invalid_argument("gravity_pose.enabled must be boolean");
        c.enabled=int(n["enabled"])!=0;
    }
    if(!n["method"].empty()) {
        if(!n["method"].isString())throw std::invalid_argument("gravity_pose.method must be string");
        c.method=std::string(n["method"]);
    }
    auto number=[&](const char* key,double& v) {
        if(n[key].empty())return;
        if(!n[key].isInt()&&!n[key].isReal())throw std::invalid_argument(std::string("gravity_pose.")+key+" must be numeric");
        v=double(n[key]);
    };
#define READ_NUM(key) number(#key,c.key)
    READ_NUM(pixel_sigma_px); READ_NUM(huber_sigma); READ_NUM(normal_sigma_floor_deg);
    READ_NUM(plane_sigma_deg); READ_NUM(max_prior_sigma_deg); READ_NUM(max_prior_age_s);
    READ_NUM(max_normal_residual_sigma); READ_NUM(max_reprojection_px); READ_NUM(init_window_s);
    READ_NUM(max_imu_gap_s); READ_NUM(gravity_m_s2); READ_NUM(acc_norm_tolerance); READ_NUM(acc_std_max);
    READ_NUM(gyro_motion_max_deg_s); READ_NUM(gyro_noise_rad_sqrt_s); READ_NUM(gyro_bias_sigma_deg_s);
    READ_NUM(stationary_correction_tau_s);
#undef READ_NUM
    for(auto item:{std::pair<const char*,int*>{"max_iterations",&c.max_iterations},{"min_init_samples",&c.min_init_samples}}) {
        if(n[item.first].empty())continue;
        if(!n[item.first].isInt())throw std::invalid_argument(std::string("gravity_pose.")+item.first+" must be integer");
        *item.second=int(n[item.first]);
    }
    validateGravityPoseConfig(c);return c;
}
GravityNormalEstimator::GravityNormalEstimator(const GravityPoseConfig& c,const Eigen::Matrix3d& r):cfg_(c),R_imu_cam_(r) {
    validateGravityPoseConfig(c);
    if(!r.allFinite()||(r.transpose()*r-Eigen::Matrix3d::Identity()).norm()>1e-6||std::abs(r.determinant()-1)>1e-6)
        throw std::invalid_argument("gravity normal: invalid IMU-to-camera rotation");
}
void GravityNormalEstimator::reset() {
    window_.clear();initialized_=false;last_t_=initialized_t_=last_correction_t_=-1;
    angular_variance_=0;normal_imu_=Eigen::Vector3d::UnitZ();last_gyro_.setZero();gyro_bias_.setZero();status_="not_initialized";
}
bool GravityNormalEstimator::feedImu(double t,const Eigen::Vector3d& a,const Eigen::Vector3d& w,bool stationary) {
    if(!std::isfinite(t)||t<0||!a.allFinite()||!w.allFinite()||a.norm()<1e-6) {
        reset();status_="invalid_imu";return false;
    }
    if(last_t_>=0 && t<=last_t_) {status_="out_of_order_imu";return false;}
    if(last_t_>=0 && t-last_t_>cfg_.max_imu_gap_s+1e-9) {reset();status_="imu_gap";}
    const double dt=last_t_<0?0:t-last_t_;
    const Eigen::Vector3d midpoint=(last_gyro_+w)*.5-gyro_bias_;
    if(initialized_) {
        normal_imu_=(rotation(-midpoint*dt)*normal_imu_).normalized();
        const double elapsed=t-initialized_t_,old_elapsed=std::max(0.,last_t_-initialized_t_);
        angular_variance_+=cfg_.gyro_noise_rad_sqrt_s*cfg_.gyro_noise_rad_sqrt_s*dt+
            std::pow(cfg_.gyro_bias_sigma_deg_s*rad,2)*(elapsed*elapsed-old_elapsed*old_elapsed);
    }
    const bool quiet=stationary && w.norm()<=cfg_.gyro_motion_max_deg_s*rad;
    if(quiet) {
        window_.push_back({t,a,w});
        while(window_.size()>2 && t-window_[1].t>=cfg_.init_window_s)window_.pop_front();
        if(window_.size()>=size_t(cfg_.min_init_samples) && t-window_.front().t>=cfg_.init_window_s-1e-9) {
            // Transport every accelerometer vector to the WINDOW END using gyro.
            // Averaging untransported samples would confuse small real rotations with bias.
            Eigen::Matrix3d transport=Eigen::Matrix3d::Identity();
            Eigen::Vector3d mean=Eigen::Vector3d::Zero(),sq=Eigen::Vector3d::Zero();
            Eigen::Vector3d gyro_mean=Eigen::Vector3d::Zero();double max_gyro=0;
            for(size_t k=window_.size();k-->0;) {
                gyro_mean+=window_[k].gyro;max_gyro=std::max(max_gyro,window_[k].gyro.norm());
                if(k+1<window_.size())transport=transport*rotation(-(.5*(window_[k].gyro+window_[k+1].gyro)-gyro_bias_)*(window_[k+1].t-window_[k].t));
                Eigen::Vector3d value=transport*window_[k].acc;mean+=value;sq+=value.cwiseProduct(value);
            }
            mean/=double(window_.size());sq/=double(window_.size());
            const double deviation=(sq-mean.cwiseProduct(mean)).cwiseMax(0.).maxCoeff();
            // Test the complete stationary WINDOW. A single ordinary noise
            // sample must not restart one second of acquisition; sustained
            // acceleration or excessive variance still prevents initialization.
            if(deviation<=cfg_.acc_std_max*cfg_.acc_std_max && mean.norm()>1 &&
               std::abs(mean.norm()-cfg_.gravity_m_s2)<=cfg_.acc_norm_tolerance) {
                const Eigen::Vector3d measured=-mean.normalized(); // specific force UP -> template +Z DOWN
                if(!initialized_) {
                    // Only actual angular rest permits zero-rate calibration.
                    // Do not subtract the mean gyro of a visibly jittering hover.
                    if(max_gyro<.1*rad)gyro_bias_=gyro_mean/double(window_.size());
                    normal_imu_=measured;initialized_=true;initialized_t_=t;
                    angular_variance_=std::pow(cfg_.normal_sigma_floor_deg*rad,2);
                } else {
                    const double alpha=1-std::exp(-dt/cfg_.stationary_correction_tau_s);
                    normal_imu_=((1-alpha)*normal_imu_+alpha*measured).normalized();
                    angular_variance_=std::max(std::pow(cfg_.normal_sigma_floor_deg*rad,2),angular_variance_*(1-alpha));
                }
                last_correction_t_=t;status_="ready";
            } else status_=initialized_?"propagated":"stationary_window_rejected";
        }
    } else {window_.clear();status_=initialized_?"propagated":"waiting_stationary";}
    last_t_=t;last_gyro_=w;return initialized_;
}
GravityNormalPrior GravityNormalEstimator::priorAt(double t) const {
    GravityNormalPrior p;p.t_sample=last_t_;p.source="raw_imu_stationary_gyro";p.status=status_;
    if(!initialized_)return p;
    if(!std::isfinite(t)||t<last_t_-1e-9) {p.status="future_imu";return p;}
    if(t-last_t_>cfg_.max_prior_age_s+1e-9) {p.status="stale_imu";return p;}
    p.normal_camera=(R_imu_cam_*rotation(-(last_gyro_-gyro_bias_)*(t-last_t_))*normal_imu_).normalized();
    const double extra=t-last_t_,elapsed=t-initialized_t_,old=last_t_-initialized_t_;
    const double variance=angular_variance_+std::pow(cfg_.gyro_noise_rad_sqrt_s,2)*extra+
        std::pow(cfg_.gyro_bias_sigma_deg_s*rad,2)*(elapsed*elapsed-old*old);
    p.covariance_rad2=variance*(Eigen::Matrix3d::Identity()-p.normal_camera*p.normal_camera.transpose());
    p.valid=std::sqrt(variance)/rad<=cfg_.max_prior_sigma_deg;
    if(!p.valid)p.status="uncertain_imu";
    return p;
}
} // namespace gpnp
