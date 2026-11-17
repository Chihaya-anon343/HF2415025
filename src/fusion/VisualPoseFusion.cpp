#include "fusion/EskfFusionManager.hpp"
#include "common/VisualObservationPolicy.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpnp::fusion {
namespace {
using V6 = Eigen::Matrix<double,6,1>;
using M6 = Eigen::Matrix<double,6,6>;
using H6 = Eigen::Matrix<double,6,15>;

Eigen::Matrix3d rotationOf(const eskf::ESKF_VIO& filter) {
    return Eigen::Quaterniond(filter.x(6),filter.x(7),filter.x(8),filter.x(9)).normalized().toRotationMatrix();
}
Eigen::Vector3d rotationLog(const Eigen::Matrix3d& r) {
    Eigen::AngleAxisd a(r); return a.axis()*a.angle();
}
struct Mode {
    int index=-1;
    Eigen::Matrix3d R=Eigen::Matrix3d::Identity();
    Eigen::Vector3d p=Eigen::Vector3d::Zero();
    M6 covariance=M6::Zero(), innovation_covariance=M6::Zero();
    V6 innovation=V6::Zero();
    H6 H=H6::Zero();
    double nis=0,score=0;
};
bool makeMode(const VisualPoseCandidate& c, const EskfFusionConfig& cfg, Mode& m) {
    if(c.support<4 || !c.covariance_valid || !c.R.allFinite() || !c.t_mm.allFinite() || c.t_mm.z()<=0 ||
       (c.R.transpose()*c.R-Eigen::Matrix3d::Identity()).norm()>1e-5 ||
       std::abs(c.R.determinant()-1)>1e-5 || !c.covariance.allFinite())return false;
    Eigen::LLT<M6> input(c.covariance);
    if(input.info()!=Eigen::Success)return false;
    m.R=cfg.R_template_world*c.R.transpose();
    m.p=-m.R*c.t_mm/1000.0;
    // 输入扰动 [dt_camera(mm), LEFT dtheta_camera]；输出 [dp_world(m), RIGHT dtheta_camera].
    M6 J=M6::Zero();J.topLeftCorner<3,3>()=-m.R/1000.0;
    J.topRightCorner<3,3>()=-m.R*eskf::skew(c.t_mm)/1000.0;
    J.bottomRightCorner<3,3>()=-Eigen::Matrix3d::Identity();
    m.covariance=J*c.covariance*J.transpose();
    m.covariance.topLeftCorner<3,3>().diagonal().array()+=std::pow(cfg.visual_position_floor_m,2);
    m.covariance.bottomRightCorner<3,3>().diagonal().array()+=std::pow(cfg.visual_rotation_floor_deg*CV_PI/180,2);
    m.covariance=(m.covariance+m.covariance.transpose()).eval()/2;
    return m.covariance.allFinite();
}
void inject(eskf::ESKF_VIO& filter, const Eigen::Matrix<double,15,1>& dx,
            const Eigen::Matrix<double,15,15>& covariance) {
    filter.x.segment<3>(0)+=dx.segment<3>(0);
    filter.x.segment<3>(3)+=dx.segment<3>(3);
    Eigen::Matrix<double,4,1> q=filter.x.segment<4>(6);
    q=eskf::quat_mul(q,eskf::quat_exp(dx.segment<3>(6)));q.normalize();
    filter.x.segment<4>(6)=q;
    filter.x.segment<3>(10)+=dx.segment<3>(9);
    filter.x.segment<3>(13)+=dx.segment<3>(12);
    Eigen::Matrix<double,15,15> G=Eigen::Matrix<double,15,15>::Identity();
    G.block<3,3>(6,6)-=0.5*eskf::skew(dx.segment<3>(6));
    filter.P=G*covariance*G.transpose();
    filter.P=(filter.P+filter.P.transpose()).eval()/2;
}
} // namespace

void EskfFusionManager::feedVisualObservation(const VisualObservation& observation) {
    CameraObservation obs;obs.t_exposure=observation.t_sample;obs.t_arrival=observation.t_arrival;
    obs.hypotheses=observation;feedCameraPose(obs);
}

std::vector<VisualFeedback> EskfFusionManager::takeVisualFeedback() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<VisualFeedback> result(visual_feedback_.begin(),visual_feedback_.end());
    visual_feedback_.clear();return result;
}

void EskfFusionManager::processVisualObservation(const VisualObservation& obs) {
    VisualFeedback feedback;feedback.frame_id=obs.frame_id;feedback.observation=obs;
    auto emit=[&](const char* status) {
        feedback.status=status;
        if(!feedback.accepted)++stats_.cam_ignored;
        visual_feedback_.push_back(std::move(feedback));
        // 消费端暂停时内存仍有界；过期反馈不应再写回视觉历史。
        while(visual_feedback_.size()>256)visual_feedback_.pop_front();
    };
    if(std::any_of(obs.candidates.begin(),obs.candidates.end(),[](const auto& c){return c.uses_radar_prior;})) {emit("radar_prior_not_independent");return;}
    if(std::any_of(obs.candidates.begin(),obs.candidates.end(),[](const auto& c){return c.uses_inertial_prior;})) {
        emit("inertial_prior_not_independent");return;
    }
    if(!std::isfinite(obs.t_sample)||!std::isfinite(obs.t_arrival)||obs.t_arrival<obs.t_sample ||
       obs.frame_id<0){emit("invalid_timestamp");return;}
    if(obs.frame_id<=last_visual_frame_ || obs.t_sample<=last_visual_sample_) {
        emit("out_of_order");return;
    }
    last_visual_frame_=obs.frame_id;last_visual_sample_=obs.t_sample;
    if(initialized_ && obs.t_sample>=last_prop_t_)
        propagateInternal(obs.t_sample); // 拒绝视觉观测也须保留惯性预测
    if(obs.candidates.empty()) {
        visual_init_pending_.reset();visual_init_count_=0;
        propagateInternal(std::max(obs.t_sample,last_prop_t_));emit("no_candidate");return;
    }
    std::vector<Mode> modes;
    for(size_t i=0;i<obs.candidates.size();++i) {
        Mode mode;mode.index=static_cast<int>(i);
        if(makeMode(obs.candidates[i],cfg_,mode))modes.push_back(mode);
    }
    if(modes.empty()){emit("invalid_covariance");return;}
    // 首次初始化不能借助一个未经确认的视觉分支构造“IMU 先验”。
    if(!initialized_) {
        if(!obs.ippe_complete || obs.ambiguous || modes.front().index!=0) {
            visual_init_pending_.reset();visual_init_count_=0;emit("ambiguous_initialization");return;
        }
        const auto& best=modes.front();
        if(best.covariance.topLeftCorner<3,3>().diagonal().maxCoeff()>
               std::pow(cfg_.visual_init_max_position_std_m,2) ||
           best.covariance.bottomRightCorner<3,3>().diagonal().maxCoeff()>
               std::pow(cfg_.visual_init_max_rotation_std_deg*CV_PI/180,2)) {
            visual_init_pending_.reset();visual_init_count_=0;emit("weak_initialization");return;
        }
        bool consistent=false;
        if(visual_init_pending_ && obs.frame_id==visual_init_pending_->frame_id+1 &&
           obs.model_frame==visual_init_pending_->model_frame &&
           obs.t_sample-visual_init_pending_->t_sample<=cfg_.max_cam_gap_s) {
            Mode prev;
            consistent=makeMode(visual_init_pending_->candidates.front(),cfg_,prev) &&
                visualRotationDistance(best.R,prev.R)<=cfg_.visual_init_rotation_deg &&
                (best.p-prev.p).norm()<=cfg_.visual_init_translation_m;
        }
        visual_init_count_=consistent?visual_init_count_+1:1;
        visual_init_pending_=obs;
        if(visual_init_count_<cfg_.visual_init_frames){emit("initialization_pending");return;}
        // 尚未初始化时排掉旧样本；积分时间从这帧观测起算，不能使用到达时间。
        propagateInternal(obs.t_sample);
        lazyInit(best.R,best.p+best.R*cfg_.params.p_imu_in_cam);
        M6 lever=M6::Identity();lever.topRightCorner<3,3>()=-best.R*eskf::skew(cfg_.params.p_imu_in_cam);
        const M6 initial_measurement_covariance=lever*best.covariance*lever.transpose();
        const int pose_index[6]={0,1,2,6,7,8};
        for(int i=0;i<6;++i)for(int j=0;j<6;++j)
            eskf_.P(pose_index[i],pose_index[j])+=initial_measurement_covariance(i,j);
        last_prop_t_=last_cam_t_=last_measurement_t_=obs.t_sample;
        state_hist_.clear();imu_hist_.clear();state_hist_.push_back({obs.t_sample,eskf_});
        visual_model_frame_=obs.model_frame;
        feedback.accepted=true;feedback.candidate_index=best.index;
        ++stats_.cam_updates;emit("initialized");return;
    }
    if(obs.model_frame!=visual_model_frame_){emit("model_frame_mismatch");return;}
    // 所有候选在同一个采样时刻、同一份预测状态上比较。延迟越过已提交的
    // 观测更新时拒绝，而不是回放不完整的历史后覆盖雷达/相机信息。
    const double now=last_prop_t_;
    std::optional<eskf::ESKF_VIO> present;
    if(now>obs.t_sample+1e-9) {
        if(now-obs.t_sample>cfg_.backprop_window_s || obs.t_sample<last_measurement_t_-1e-9) {
            emit("late_conflicting_history");return;
        }
        const StateSnap* snap=nullptr;
        for(const auto& s:state_hist_)if(s.t<=obs.t_sample+1e-9)snap=&s;
        if(!snap){emit("late_missing_history");return;}
        present=eskf_;eskf_=snap->eskf;
        double t=snap->t;
        for(const auto& s:imu_hist_) {
            if(s.t<=t+1e-9 || s.t>obs.t_sample+1e-9)continue;
            eskf_.predict_adaptive(s.acc,s.gyro,s.t-t);t=s.t;
        }
        if(t<obs.t_sample)eskf_.predict(std::nullopt,std::nullopt,obs.t_sample-t);
    } else propagateInternal(obs.t_sample);
    const auto predicted=eskf_;
    const Eigen::Matrix3d Rp=rotationOf(predicted);
    const Eigen::Vector3d pp=predicted.x.segment<3>(0)-Rp*cfg_.params.p_imu_in_cam;
    H6 H=H6::Zero();H.block<3,3>(0,0).setIdentity();H.block<3,3>(3,6).setIdentity();
    H.block<3,3>(0,6)=Rp*eskf::skew(cfg_.params.p_imu_in_cam);
    std::vector<Mode> admissible;
    for(auto mode:modes) {
        mode.innovation.head<3>()=mode.p-pp;
        mode.innovation.tail<3>()=rotationLog(Rp.transpose()*mode.R);
        // 测量右扰动在其自身切空间，映射至当前预测的旋转残差切空间。
        const Eigen::Vector3d phi=mode.innovation.tail<3>();
        const double theta=phi.norm();const Eigen::Matrix3d A=eskf::skew(phi);
        const double coeff=theta<1e-5?1.0/12.0:
            (1.0-0.5*theta/std::tan(0.5*theta))/(theta*theta);
        M6 tangent=M6::Identity();tangent.bottomRightCorner<3,3>()=Eigen::Matrix3d::Identity()+0.5*A+coeff*A*A;
        mode.covariance=tangent*mode.covariance*tangent.transpose();
        mode.H=H;
        mode.H.block<3,3>(3,6)=Eigen::Matrix3d::Identity()-0.5*A+coeff*A*A;
        mode.innovation_covariance=mode.H*predicted.P*mode.H.transpose()+mode.covariance;
        Eigen::LLT<M6> llt(mode.innovation_covariance);
        if(llt.info()!=Eigen::Success)continue;
        mode.nis=mode.innovation.dot(llt.solve(mode.innovation));
        mode.score=mode.nis+2*llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
        if(std::isfinite(mode.score)&&mode.nis<=cfg_.visual_nis_gate)admissible.push_back(mode);
    }
    auto reject=[&](const char* status){if(present)eskf_=*present;emit(status);};
    if(admissible.empty()){reject("innovation_rejected");return;}
    std::stable_sort(admissible.begin(),admissible.end(),[](const Mode& a,const Mode& b){return a.score<b.score;});
    const auto& best=admissible.front();feedback.nis=best.nis;
    feedback.score_gap=std::numeric_limits<double>::infinity();
    for(size_t i=1;i<admissible.size();++i) {
        const auto& other=admissible[i];
        const auto& a=obs.candidates[best.index];const auto& b=obs.candidates[other.index];
        if(visualRotationDistance(a.R,b.R)<10 &&
           (a.t_mm-b.t_mm).norm()/std::max(1.,a.t_mm.norm())<0.05)continue;
        feedback.score_gap=other.score-best.score;break;
    }
    if(feedback.score_gap<cfg_.visual_score_gap){reject("ambiguous_innovation");return;}
    Eigen::LLT<M6> llt(best.innovation_covariance);
    const Eigen::Matrix<double,15,6> gain=llt.solve((predicted.P*best.H.transpose()).transpose()).transpose();
    const Eigen::Matrix<double,15,1> dx=gain*best.innovation;
    const auto I=Eigen::Matrix<double,15,15>::Identity();
    const Eigen::Matrix<double,15,15> residual=I-gain*best.H;
    const Eigen::Matrix<double,15,15> P=residual*predicted.P*residual.transpose()+gain*best.covariance*gain.transpose();
    eskf_=predicted;inject(eskf_,dx,P);
    if(!eskf_.x.allFinite()||!eskf_.P.allFinite()||Eigen::LLT<Eigen::MatrixXd>(eskf_.P).info()!=Eigen::Success) {
        eskf_=present?*present:predicted;emit("numeric_update_rejected");return;
    }
    last_cam_t_=last_measurement_t_=obs.t_sample;
    // 修正后重放纯预测；此前已保证窗口中没有后续观测更新。
    const auto old_imu=imu_hist_;imu_hist_.clear();state_hist_.clear();
    state_hist_.push_back({obs.t_sample,eskf_});last_snap_t_=obs.t_sample;
    double t=obs.t_sample;
    if(present) {
        for(const auto& s:old_imu)if(s.t>t+1e-9 && s.t<=now+1e-9) {
            eskf_.predict_adaptive(s.acc,s.gyro,s.t-t);t=s.t;recordHistory(s);
        }
        if(t<now)eskf_.predict(std::nullopt,std::nullopt,now-t);
    }
    last_prop_t_=std::max(obs.t_sample,now);
    feedback.accepted=true;feedback.candidate_index=best.index;
    ++stats_.cam_updates;emit("accepted");
}
} // namespace gpnp::fusion
