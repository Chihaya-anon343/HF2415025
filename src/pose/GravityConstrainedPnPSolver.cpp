#include "pose/GravityConstrainedPnPSolver.hpp"
#include "pose/MonoPoseLimits.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace gpnp {
namespace {
constexpr double pi=3.14159265358979323846,rad=pi/180.;
using Vector6=Eigen::Matrix<double,6,1>;
using Matrix6=Eigen::Matrix<double,6,6>;
using Basis=Eigen::Matrix<double,3,2>;
Eigen::Matrix3d expR(const Eigen::Vector3d& v) {
    double a=v.norm();return a<1e-14?Eigen::Matrix3d::Identity():Eigen::AngleAxisd(a,v/a).toRotationMatrix();
}
Basis tangent(const Eigen::Vector3d& n) {
    Basis b;b.col(0)=n.unitOrthogonal();b.col(1)=n.cross(b.col(0));return b;
}
struct Problem {
    const VisualObservation& observation;
    const GravityPoseConfig& config;
    Basis B;
    Eigen::Vector3d n;
    Eigen::Matrix3d R0;
    Eigen::Matrix2d inverse_sqrt;
    const RadarHeightPrior* radar{nullptr};
    double radar_sigma{1};
    Eigen::Matrix3d poseR(const Vector6& x) const {
        return expR(B*x.tail<2>())*R0*Eigen::AngleAxisd(x(3),Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    bool residual(const Vector6& x,Eigen::VectorXd& r,bool include_prior=true) const {
        const auto R=poseR(x);const Eigen::Vector3d t=x.head<3>();
        const auto& o=observation;const size_t count=o.image_points.size();
        r.resize(2*count+(include_prior?2:0)+(radar?1:0));
        for(size_t i=0;i<count;++i) {
            const Eigen::Vector3d p=R*(o.object_points_mm[i]/1000.)+t,q=o.camera_matrix*p;
            if(!p.allFinite()||p.z()<=1e-6||std::abs(q.z())<1e-10)return false;
            r(2*i)=(q.x()/q.z()-o.image_points[i].x)/config.pixel_sigma_px;
            r(2*i+1)=(q.y()/q.z()-o.image_points[i].y)/config.pixel_sigma_px;
        }
        if(include_prior) {
            const Eigen::Vector3d current=R.col(2),cross=n.cross(current);
            const double sine=cross.norm(),dot=std::clamp(n.dot(current),-1.,1.);
            if(dot<-.99)return false; // signed normal: antipodal is NOT a zero residual.
            const Eigen::Vector3d delta=sine<1e-12?Eigen::Vector3d::Zero().eval():(cross*(std::atan2(sine,dot)/sine)).eval();
            r.segment<2>(2*count)=inverse_sqrt*B.transpose()*delta;
        }
        if(radar)r(r.size()-1)=(R.col(2).dot(t)-radar->height_m)/radar_sigma;
        return r.allFinite();
    }
    double cost(const Eigen::VectorXd& r) const {
        double c=0;const int count=observation.image_points.size();
        for(int i=0;i<count;++i) {
            const double a=r.segment<2>(2*i).norm(),k=config.huber_sigma;
            c+=a<=k?a*a:2*k*a-k*k;
        }
        if(r.size()>2*count)c+=radar?r.tail(r.size()-2*count).squaredNorm():r.tail<2>().squaredNorm();
        return c;
    }
    bool linearize(const Vector6& x,int dim,Eigen::MatrixXd& J,Eigen::VectorXd& r,bool include_prior=true) const {
        if(!residual(x,r,include_prior))return false;
        J.resize(r.size(),dim);
        for(int j=0;j<dim;++j) {
            const double h=1e-6;Vector6 lo=x,hi=x;lo(j)-=h;hi(j)+=h;
            Eigen::VectorXd a,b;
            if(!residual(lo,a,include_prior)||!residual(hi,b,include_prior))return false;
            J.col(j)=(b-a)/(2*h);
        }
        for(int i=0;i<int(observation.image_points.size());++i) {
            const double length=r.segment<2>(2*i).norm();
            const double weight=length<=config.huber_sigma?1:std::sqrt(config.huber_sigma/length);
            J.middleRows(2*i,2)*=weight;r.segment<2>(2*i)*=weight;
        }
        return J.allFinite();
    }
    bool initialTranslation(Vector6& x) const {
        const auto R=poseR(x);const auto inv=observation.camera_matrix.inverse();
        Eigen::MatrixXd A(observation.image_points.size()*2,3);Eigen::VectorXd rhs(A.rows());
        for(int i=0;i<int(observation.image_points.size());++i) {
            Eigen::Vector3d ray=inv*Eigen::Vector3d(observation.image_points[i].x,observation.image_points[i].y,1);
            if(std::abs(ray.z())<1e-12)return false;
            ray/=ray.z();
            const Eigen::Vector3d p=R*(observation.object_points_mm[i]/1000.);
            A.row(2*i)<<1,0,-ray.x();A.row(2*i+1)<<0,1,-ray.y();
            rhs(2*i)=ray.x()*p.z()-p.x();rhs(2*i+1)=ray.y()*p.z()-p.y();
        }
        auto qr=A.colPivHouseholderQr();if(qr.rank()!=3)return false;
        x.head<3>()=qr.solve(rhs);
        if(radar) {const double h=R.col(2).dot(x.head<3>());if(h<=0)return false;x.head<3>()*=radar->height_m/h;}
        return x.allFinite()&&x(2)>0;
    }
    bool optimize(Vector6& x,int dim,int& iterations) const {
        Eigen::VectorXd residual0;if(!residual(x,residual0))return false;
        double current=cost(residual0),lambda=1e-3;bool converged=false;
        for(int i=0;i<config.max_iterations;++i) {
            ++iterations;Eigen::MatrixXd J;Eigen::VectorXd r;
            if(!linearize(x,dim,J,r))return false;
            Eigen::MatrixXd H=J.transpose()*J;Eigen::VectorXd gradient=J.transpose()*r;
            if(gradient.lpNorm<Eigen::Infinity>()<1e-7){converged=true;break;}
            Eigen::MatrixXd damped=H;
            damped.diagonal().array()+=lambda*H.diagonal().array().max(1.);
            const Eigen::VectorXd dx=damped.ldlt().solve(-gradient);
            if(!dx.allFinite())return false;
            if(dx.norm()<1e-9){converged=true;break;}
            Vector6 trial=x;trial.head(dim)+=dx;Eigen::VectorXd next;
            if(residual(trial,next) && cost(next)<current) {
                const double improvement=current-cost(next);x=trial;current=cost(next);lambda=std::max(1e-12,lambda*.3);
                if(improvement<1e-8*(1+current)){converged=true;break;}
            } else {
                lambda*=10.;if(lambda>1e12)break;
            }
        }
        return converged;
    }
};
bool inversePositive(const Eigen::MatrixXd& h,Eigen::MatrixXd& inverse) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e((h+h.transpose())*.5);
    if(e.info()!=Eigen::Success || e.eigenvalues().minCoeff()<=std::max(1e-12,e.eigenvalues().maxCoeff()*1e-12))return false;
    inverse=e.eigenvectors()*e.eigenvalues().cwiseInverse().asDiagonal()*e.eigenvectors().transpose();return inverse.allFinite();
}
}
GravityConstrainedPnPSolver::GravityConstrainedPnPSolver(const GravityPoseConfig& c):cfg_(c) {validateGravityPoseConfig(c);}

GravityPoseResult GravityConstrainedPnPSolver::solve(const VisualObservation& o,const GravityNormalPrior& prior,
                                                   double time,int min_support) const {
    return solveImpl(o,prior,time,min_support,nullptr,nullptr);
}
RadarPoseResult GravityConstrainedPnPSolver::solveWithRadar(const VisualObservation& o,const GravityNormalPrior& prior,
    const RadarHeightPrior& radar,const RadarPoseConfig& config,double time,int min_support) const {
    validateRadarPoseConfig(config);
    return solveImpl(o,prior,time,min_support,&radar,&config);
}
RadarPoseResult GravityConstrainedPnPSolver::solveImpl(const VisualObservation& o,const GravityNormalPrior& prior,
    double time,int min_support,const RadarHeightPrior* radar,const RadarPoseConfig* radar_config) const {
    const auto started=std::chrono::steady_clock::now();
    RadarPoseResult out;out.method=cfg_.method;out.t_sample=time;out.prior_source=prior.source;
    out.prior_normal_camera=prior.normal_camera;
    auto finish=[&](const std::string& status) {
        out.status=status;out.time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return out;
    };
    if(radar) {
        out.method="soft_normal_height";out.radar_sample_id=radar->sample_id;out.radar_height_m=radar->height_m;
        if(!radar_config->enabled)return finish("disabled");
        if(cfg_.method!="soft_normal")return finish("radar_requires_soft_normal");
        if(!radar->valid)return finish("radar_"+radar->status);
        if(!std::isfinite(time)||!std::isfinite(radar->t_sample)||radar->t_sample<0||radar->t_sample>time+1e-9)return finish("radar_invalid_time");
        out.radar_age_s=time-radar->t_sample;
        if(out.radar_age_s>radar_config->max_age_s+1e-9)return finish("radar_stale");
        if(!std::isfinite(radar->height_m)||radar->height_m<=0)return finish("radar_invalid_height");
        out.height_sigma_m=std::hypot(radar_config->height_sigma_m,radar_config->age_noise_m_s*out.radar_age_s);
    }
    if(!cfg_.enabled)return finish("disabled");
    if(!prior.valid)return finish("prior_"+prior.status);
    if(!std::isfinite(time)||!std::isfinite(prior.t_sample)||time<prior.t_sample-1e-9)return finish("prior_future");
    out.prior_age_s=time-prior.t_sample;
    if(out.prior_age_s>cfg_.max_prior_age_s+1e-9)return finish("prior_stale");
    if(!prior.normal_camera.allFinite()||std::abs(prior.normal_camera.norm()-1)>1e-5 ||
       !prior.covariance_rad2.allFinite() || (prior.covariance_rad2-prior.covariance_rad2.transpose()).norm()>1e-8)
        return finish("prior_invalid");
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> full_cov(prior.covariance_rad2);
    if(full_cov.info()!=Eigen::Success||full_cov.eigenvalues().minCoeff()<-1e-12)return finish("prior_invalid_covariance");
    const Basis B=tangent(prior.normal_camera);
    Eigen::Matrix2d prior_cov=B.transpose()*prior.covariance_rad2*B;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> pe(prior_cov);
    if(pe.info()!=Eigen::Success||pe.eigenvalues().minCoeff()<=0)return finish("prior_invalid_covariance");
    out.prior_sigma_deg=std::sqrt(pe.eigenvalues().maxCoeff())/rad;
    if(out.prior_sigma_deg>cfg_.max_prior_sigma_deg)return finish("prior_uncertain");
    prior_cov=pe.eigenvectors()*pe.eigenvalues().cwiseMax(std::pow(cfg_.normal_sigma_floor_deg*rad,2)).asDiagonal()*pe.eigenvectors().transpose();
    prior_cov.diagonal().array()+=std::pow(cfg_.plane_sigma_deg*rad,2);
    if(radar) {
        out.tilt_deg=std::acos(std::clamp(prior.normal_camera.z(),-1.,1.))/rad;
        if(out.tilt_deg>radar_config->max_tilt_deg+1e-9)return finish("radar_tilt_unvalidated");
    }
    out.prior_valid=true;
    Eigen::LLT<Eigen::Matrix2d> llt(prior_cov);
    const Eigen::Matrix2d invsqrt=llt.matrixL().solve(Eigen::Matrix2d::Identity());
    const int count=o.image_points.size();min_support=std::max(4,min_support);
    if(count<min_support||count!=int(o.object_points_mm.size()))return finish("insufficient_points");
    if(!o.camera_matrix.allFinite()||o.camera_matrix(0,0)<=0||o.camera_matrix(1,1)<=0||std::abs(o.camera_matrix.determinant())<1e-9)
        return finish("invalid_camera");
    Eigen::Vector3d center=Eigen::Vector3d::Zero();
    for(int i=0;i<count;++i) {
        const auto& p=o.object_points_mm[i];const auto& u=o.image_points[i];
        if(!p.allFinite()||std::abs(p.z())>1e-6||!std::isfinite(u.x)||!std::isfinite(u.y))return finish("invalid_points");
        center+=p;
    }
    center/=count;Eigen::Matrix2d scatter=Eigen::Matrix2d::Zero();
    for(const auto& p:o.object_points_mm){Eigen::Vector2d d=(p-center).head<2>();scatter+=d*d.transpose();}
    if(scatter.determinant()<1e-8)return finish("degenerate_model");
    Problem problem{o,cfg_,B,prior.normal_camera,
        Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(),prior.normal_camera).toRotationMatrix(),invsqrt};
    problem.radar=radar;if(radar)problem.radar_sigma=out.height_sigma_m;
    std::vector<double> angles;
    for(const auto& c:o.candidates)if(c.R.allFinite()) {
        const Eigen::Matrix3d local=problem.R0.transpose()*c.R;
        angles.push_back(std::atan2(local(1,0)-local(0,1),local(0,0)+local(1,1)));
    }
    for(double value:{0.,pi/2,pi,-pi/2})angles.push_back(value);
    std::vector<double> unique;
    for(double a:angles) {
        bool duplicate=false;for(double b:unique)if(std::abs(std::remainder(a-b,2*pi))<.01)duplicate=true;
        if(!duplicate)unique.push_back(a);
    }
    const bool fixed=cfg_.method=="fixed_normal";
    double best=std::numeric_limits<double>::infinity();Vector6 solution=Vector6::Zero();bool found=false;
    for(double yaw:unique) {
        Vector6 x=Vector6::Zero();x(3)=yaw;int iterations=0;
        if(!problem.initialTranslation(x)||!problem.optimize(x,4,iterations))continue;
        if(!fixed && !problem.optimize(x,6,iterations))continue;
        Eigen::VectorXd residual;if(!problem.residual(x,residual))continue;
        if(problem.cost(residual)<best){best=problem.cost(residual);solution=x;out.iterations=iterations;found=true;}
    }
    if(!found)return finish("not_converged");
    out.R=problem.poseR(solution);out.t_mm=solution.head<3>()*1000;
    Eigen::VectorXd r;if(!problem.residual(solution,r))return finish("invalid_solution");
    double total=0,sse=0;
    for(int i=0;i<count;++i) {
        const double e=r.segment<2>(2*i).norm()*cfg_.pixel_sigma_px;
        total+=e;sse+=e*e;if(e<cfg_.max_reprojection_px)++out.support;
    }
    out.mean_error_px=total/count;out.rms_error_px=std::sqrt(sse/count);
    out.normal_residual_deg=std::acos(std::clamp(out.R.col(2).dot(prior.normal_camera),-1.,1.))/rad;
    // Same translation-norm envelope as the existing MonoPnPSolver (mm).
    if(!(out.t_mm.norm()>kMonoMinTranslationNormMm && out.t_mm.norm()<kMonoMaxTranslationNormMm))return finish("translation_out_of_range");
    // Preserve the front-side convention used by PlanarPosePrior (+Z into board).
    if(out.R.col(2).dot(out.t_mm)<=0)return finish("back_side");
    if(out.support<min_support||out.mean_error_px>=cfg_.max_reprojection_px)return finish("reprojection_rejected");
    if(r.segment<2>(2*count).norm()>cfg_.max_normal_residual_sigma)return finish("gravity_image_conflict");
    if(radar) {
        out.predicted_height_m=out.R.col(2).dot(out.t_mm)/1000.;
        out.height_residual_m=out.predicted_height_m-radar->height_m;
        if(std::abs(out.height_residual_m)>radar_config->max_height_residual_sigma*out.height_sigma_m)return finish("radar_image_conflict");
        out.uses_radar_prior=true;out.covariance_kind="imu_radar_conditioned_posterior_dt_mm_left_dtheta_rad";
    }
    Eigen::MatrixXd J;Eigen::VectorXd weighted;
    if(!problem.linearize(solution,6,J,weighted))return finish("covariance_failed");
    Eigen::MatrixXd parameter_cov;
    if(!fixed) {
        if(!inversePositive(J.transpose()*J,parameter_cov))return finish("degenerate_information");
    } else {
        // A fixed normal is still uncertain: propagate its covariance through
        // re-estimation of translation/yaw, including the cross terms.
        Eigen::MatrixXd image=J.topRows(2*count),free_inverse;
        if(!inversePositive(image.leftCols(4).transpose()*image.leftCols(4),free_inverse))return finish("degenerate_information");
        const Eigen::MatrixXd A=-free_inverse*image.leftCols(4).transpose()*image.rightCols(2);
        parameter_cov=Matrix6::Zero();parameter_cov.topLeftCorner(4,4)=free_inverse+A*prior_cov*A.transpose();
        parameter_cov.topRightCorner(4,2)=A*prior_cov;parameter_cov.bottomLeftCorner(2,4)=prior_cov*A.transpose();
        parameter_cov.bottomRightCorner(2,2)=prior_cov;
    }
    Matrix6 G=Matrix6::Zero();G.topLeftCorner<3,3>()=1000*Eigen::Matrix3d::Identity();
    for(int j=3;j<6;++j) {
        Vector6 hi=solution,lo=solution;hi(j)+=1e-6;lo(j)-=1e-6;
        const Eigen::Matrix3d d=(problem.poseR(hi)-problem.poseR(lo))/(2e-6)*out.R.transpose();
        G.block<3,1>(3,j)=Eigen::Vector3d(d(2,1)-d(1,2),d(0,2)-d(2,0),d(1,0)-d(0,1))*.5;
    }
    out.posterior_covariance=G*parameter_cov*G.transpose();
    out.posterior_covariance=(out.posterior_covariance+out.posterior_covariance.transpose()).eval()*.5;
    out.covariance_valid=out.posterior_covariance.allFinite()&&Eigen::LLT<Matrix6>(out.posterior_covariance).info()==Eigen::Success;
    if(!out.covariance_valid)return finish("invalid_posterior_covariance");
    out.success=true;return finish("accepted");
}
} // namespace gpnp
