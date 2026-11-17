#include "pose/VisualObservationBuilder.hpp"
#include "common/VisualObservationPolicy.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>

namespace gpnp {
namespace {
Eigen::Matrix3d skew(const Eigen::Vector3d& p) {
    Eigen::Matrix3d s;s<<0,-p.z(),p.y(),p.z(),0,-p.x(),-p.y(),p.x(),0;return s;
}
void covariance(VisualPoseCandidate& c,const VisualObservation& o,
                const Eigen::Matrix3d& K,double sigma_floor) {
    c.covariance_valid=false;c.covariance.setZero();
    const size_t n=o.image_points.size();
    if(n<4 || o.object_points_mm.size()!=n)return;
    PoseCovariance information=PoseCovariance::Zero();double sse=0;
    const double length=std::max(c.t_mm.norm(),100.0);
    PoseCovariance D=PoseCovariance::Identity();D.topLeftCorner<3,3>()*=length;
    for(size_t i=0;i<n;++i) {
        const Eigen::Vector3d rotated=c.R*o.object_points_mm[i], p=rotated+c.t_mm,q=K*p;
        if(!p.allFinite()||p.z()<=1e-6||q.z()<=1e-6)return;
        Eigen::Matrix<double,2,3> projection;
        projection.row(0)=(K.row(0)*q.z()-q.x()*K.row(2))/(q.z()*q.z());
        projection.row(1)=(K.row(1)*q.z()-q.y()*K.row(2))/(q.z()*q.z());
        Eigen::Matrix<double,3,6> motion;
        motion.leftCols<3>()=Eigen::Matrix3d::Identity();motion.rightCols<3>()=-skew(rotated);
        const Eigen::Matrix<double,2,6> J=projection*motion*D;
        information.noalias()+=J.transpose()*J;
        const Eigen::Vector2d r(q.x()/q.z()-o.image_points[i].x,q.y()/q.z()-o.image_points[i].y);
        sse+=r.squaredNorm();
    }
    c.rms_error_px=std::sqrt(sse/n);
    Eigen::SelfAdjointEigenSolver<PoseCovariance> eig(information);
    if(eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff()<=
       std::max(1e-12,eig.eigenvalues().maxCoeff()*1e-10))return;
    const double variance=std::max(sigma_floor*sigma_floor,sse/std::max(1.,2.*n-6.));
    c.covariance=variance*D*eig.eigenvectors()*eig.eigenvalues().cwiseInverse().asDiagonal()*eig.eigenvectors().transpose()*D;
    c.covariance=(c.covariance+c.covariance.transpose()).eval()*.5;
    c.covariance_valid=c.covariance.allFinite() && c.covariance.diagonal().minCoeff()>0;
}
}
void finalizeVisualObservation(VisualObservation& o,const Eigen::Matrix3d& K,const VisualOutputConfig& cfg) {
    std::stable_sort(o.candidates.begin(),o.candidates.end(),[](const auto& a,const auto& b){return a.mean_error_px<b.mean_error_px;});
    std::vector<VisualPoseCandidate> unique;
    for(auto& c:o.candidates) {
        bool duplicate=false;
        for(const auto& u:unique)
            if(visualRotationDistance(c.R,u.R)<1.0 &&
               (c.t_mm-u.t_mm).norm()/std::max(u.t_mm.norm(),1.0)<0.005){duplicate=true;break;}
        if(!duplicate) {
            if(cfg.estimate_covariance)covariance(c,o,K,cfg.pixel_sigma_floor);
            unique.push_back(std::move(c));
        }
    }
    o.candidates=std::move(unique);o.evaluated=true;
    assessVisualAmbiguity(o,cfg);
}
} // namespace gpnp
