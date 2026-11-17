#include "fusion/ConditionedFlightFusion.hpp"
#include "input/FlightImuSource.hpp"
#include "eskf/eskf_vio.hpp"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <deque>
#include <chrono>

namespace gpnp::fusion {
namespace {
using namespace Eigen;
using V=Vector3d;
using M=MatrixXd;
using E=input::SensorPacket::Event;
Matrix3d rotation(const cv::FileNode& n) {
    if(n.size()!=3)throw std::invalid_argument("missing flight rotation");
    Matrix3d r;
    for(int i=0;i<3;++i){if(n[i].size()!=3)throw std::invalid_argument("rotation shape");
        for(int j=0;j<3;++j)r(i,j)=double(n[i][j]);}
    if(!r.allFinite()||(r.transpose()*r-Matrix3d::Identity()).norm()>1e-6||std::abs(1-r.determinant())>1e-6)
        throw std::invalid_argument("invalid flight rotation");
    return r;
}
M inverseSpd(const M& m) {
    LLT<M> llt(m);
    if(llt.info()!=Success)throw std::runtime_error("nonpositive flight covariance");
    return llt.solve(M::Identity(m.rows(),m.cols()));
}
double logdet(const M& m) {
    LLT<M> llt(m);
    if(llt.info()!=Success)throw std::runtime_error("nonpositive CI covariance");
    return 2*llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
}
int64_t calendar(std::string t) {
    for(char& c:t)if(c=='-'||c==' '||c==':'||c=='.')c='_';
    return input::parseFlightCalendarUs(t);
}
struct Fix {int64_t us,id;V p;};
std::vector<std::string> split(const std::string& line){
    std::istringstream s(line);std::vector<std::string> a;std::string v;
    while(std::getline(s,v,','))a.push_back(v);
    return a;
}
std::vector<Fix> fixes(const std::string& path) {
    std::ifstream f(path);std::string line;if(!std::getline(f,line))throw std::runtime_error("missing RTK "+path);
    auto cols=split(line);
    for(auto& c:cols)if(!c.empty()&&c.back()=='\r')c.pop_back();
    auto index=[&](std::string name){auto i=std::find(cols.begin(),cols.end(),name);
        if(i==cols.end())throw std::runtime_error("missing RTK column "+name);
        return i-cols.begin();};
    auto ti=index("aligned_time"),v=index("position_valid"),e=index("east_m"),n=index("north_m"),u=index("up_m");
    std::vector<Fix> result;int64_t id=-1;
    while(std::getline(f,line)){++id;auto r=split(line);if(r.at(v)!="1")continue;
        Fix fix{calendar(r.at(ti)),id,{std::stod(r.at(e)),std::stod(r.at(n)),std::stod(r.at(u))}};
        if(!fix.p.allFinite()||(!result.empty()&&fix.us<=result.back().us))throw std::runtime_error("invalid RTK");
        result.push_back(fix);}
    return result;
}
}
struct ConditionedFlightFusion::Impl {
    Matrix3d W;
    eskf::ESKFParams params;
    std::unique_ptr<eskf::ESKF_VIO> f;
    std::vector<Fix> rtk;
    std::string initialization_source="rtk";
    std::optional<double> ground_radar_height;
    std::ofstream states,events,imu_csv,radar_csv;
    std::ofstream timing;
    double position_floor=.1, rotation_floor_deg=2., radar_sigma=1., max_imu_gap_s=.1;
    V offset=V::Zero();double radar_offset=0;
    int64_t origin=-1,last=-1,previous_frame=-1,last_pose=-1,imu_id=-1,radar_id=-1;
    std::optional<E> held;
    struct Seed {int64_t us;V p;Matrix3d r;};
    std::deque<Seed> seeds;
    Impl(const cv::FileNode& cfg,const std::string& out):
        W(rotation(cfg["R_template_world"])) {
        if(!cfg["initialization_source"].empty())
            initialization_source=std::string(cfg["initialization_source"]);
        if(initialization_source=="rtk")rtk=fixes(std::string(cfg["rtk_initialization_csv"]));
        else if(initialization_source!="conditioned_pose")
            throw std::invalid_argument("flight initialization_source must be rtk or conditioned_pose");
        if(!cfg["initialization_ground_radar_height_m"].empty()){
            const auto value=cfg["initialization_ground_radar_height_m"];
            if((!value.isReal()&&!value.isInt())||!std::isfinite(double(value))||double(value)<0)
                throw std::invalid_argument("invalid initialization_ground_radar_height_m");
            if(initialization_source!="conditioned_pose")
                throw std::invalid_argument("radar ground reference requires conditioned_pose initialization");
            ground_radar_height=double(value);
        }
        params.R_imu_cam=rotation(cfg["R_imu_cam"]);
        auto number=[](cv::FileNode node,const char* key,double& dst) {
            auto v=node[key];if(v.empty())return;
            dst=double(v);
            if(!std::isfinite(dst)||dst<=0)throw std::invalid_argument(std::string("invalid flight parameter ")+key);
        };
        const auto ci=cfg["conditioned_pose"], noise=cfg["noise"];
        number(ci,"position_floor_m",position_floor);number(ci,"rotation_floor_deg",rotation_floor_deg);
        number(noise,"radar_alt_noise",radar_sigma);number(cfg,"max_imu_gap_s",max_imu_gap_s);
        number(noise,"sigma_acc",params.sigma_acc);number(noise,"sigma_gyro",params.sigma_gyro);
        number(noise,"sigma_acc_bias",params.sigma_acc_bias);number(noise,"sigma_gyro_bias",params.sigma_gyro_bias);
        number(noise,"sigma_pos_rw",params.sigma_pos_rw);
        if(!ci["update_method"].empty()&&std::string(ci["update_method"])!="covariance_intersection")
            throw std::invalid_argument("flight posterior requires covariance_intersection");
        std::filesystem::create_directories(out);
        states.open(out+"/eskf_states.csv");events.open(out+"/eskf_events.csv");
        imu_csv.open(out+"/imu.csv");radar_csv.open(out+"/radar.csv");
        timing.open(out+"/eskf_timing.csv");
        if(!states||!events||!imu_csv||!radar_csv||!timing)throw std::runtime_error("cannot write flight fusion logs");
        timing<<"source_frame_index,sensor_dispatch_us,pose_update_us\n";
        imu_csv<<std::setprecision(17)<<"sample_id,t_us,ax_m_s2,ay_m_s2,az_m_s2,gx_rad_s,gy_rad_s,gz_rad_s\n";
        radar_csv<<std::setprecision(17)<<"sample_id,t_us,height_m\n";
        states<<std::setprecision(16)<<"source_frame_index,t_us,status,quality,x,y,z,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,pxx,pyy,pzz,ci_weight,cx,cy,cz,tilt_id,pose_radar_id\n";
        events<<std::setprecision(16)<<"t_us,type,sample_id,decision,detail\n";
    }
    void advance(int64_t us){
        if(!f)return;
        if(us<last)throw std::runtime_error("fusion clock moved backwards");
        if(us==last)return;
        // Causal zero-order hold; coast across missing IMU, never integrate stale acceleration.
        const double dt=(us-last)*1e-6;
        if(held && (us-held->us)*1e-6<=max_imu_gap_s)f->predict_adaptive(held->acc,held->gyro,dt);
        else f->predict(std::nullopt,std::nullopt,dt);
        last=us;
    }
    void sensor(const E& e) {
        auto& prev=e.radar?radar_id:imu_id;
        if(e.id<=prev)throw std::runtime_error("duplicate or reversed sensor event");
        prev=e.id;advance(e.us);std::string decision="preinit";double detail=0;
        if(e.radar)radar_csv<<e.id<<','<<e.us-origin<<','<<e.height<<'\n';
        else {
            imu_csv<<e.id<<','<<e.us-origin;
            for(int i=0;i<3;++i)imu_csv<<','<<e.acc(i);
            for(int i=0;i<3;++i)imu_csv<<','<<e.gyro(i);
            imu_csv<<'\n';
        }
        if(e.radar && f) {
            const double z=e.height+radar_offset;
            detail=std::pow(z-f->x(2),2)/(f->P(2,2)+radar_sigma*radar_sigma);
            if(detail<=25){f->update_altitude(z,radar_sigma);decision="accepted";}else decision="outlier";
        }else if(!e.radar){held=e;if(f)decision="propagate";}
        events<<e.us-origin<<','<<(e.radar?"radar":"imu")<<','<<e.id<<','<<decision<<','<<detail<<'\n';
    }
    std::string correct(const RadarPoseResult& c,const input::SensorPacket& pkt,int64_t us,V& measured,double& weight) {
        if(!c.success){seeds.clear();return c.status;}
        if(!pkt.radar_prior||!pkt.radar_prior->valid||pkt.radar_prior->sample_id!=c.radar_sample_id||
           !pkt.imu_alignment||!pkt.imu_alignment->valid||!c.uses_inertial_prior||!c.uses_radar_prior||
           std::abs(pkt.radar_prior->height_m-c.radar_height_m)>1e-5 ||
           std::abs(pkt.radar_prior->age_s-c.radar_age_s)>1e-5 ||
           std::abs(pkt.imu_alignment->age_s-c.prior_age_s)>1e-5)
            throw std::runtime_error("conditioned pose dependency mismatch");
        if(!c.covariance_valid||!c.posterior_covariance.allFinite()||
            c.covariance_kind!="imu_radar_conditioned_posterior_dt_mm_left_dtheta_rad"||
            !c.R.allFinite()||!c.t_mm.allFinite()||std::abs(c.R.determinant()-1)>1e-5||
            (c.R.transpose()*c.R-Matrix3d::Identity()).norm()>1e-5||c.support<4) {seeds.clear();return "invalid_pose";}
        Matrix3d rot=W*c.R.transpose();V p=-rot*c.t_mm/1000;
        M j=M::Zero(6,6);j.block<3,3>(0,0)=-rot/1000;
        j.block<3,3>(0,3)=-rot*eskf::skew(c.t_mm)/1000;j.block<3,3>(3,3)=-Matrix3d::Identity();
        M cov=j*c.posterior_covariance*j.transpose();
        cov.topLeftCorner(3,3).diagonal().array()+=position_floor*position_floor;
        cov.bottomRightCorner(3,3).diagonal().array()+=std::pow(rotation_floor_deg*CV_PI/180,2);
        if(!f){
            if(cov.topLeftCorner(3,3).diagonal().maxCoeff()>4 ||
               cov.bottomRightCorner(3,3).diagonal().maxCoeff()>std::pow(15*CV_PI/180,2)){
                seeds.clear();return "weak_initialization";}
            if(!seeds.empty()&&(us-seeds.back().us>150000 || (p-seeds.back().p).norm()>1. ||
                AngleAxisd(rot.transpose()*seeds.back().r).angle()>.18))seeds.clear();
            seeds.push_back({us,p,rot});if(seeds.size()<5)return "waiting_window";
            V initial_position=p;int64_t fix_id=-1;double fix_age_s=0;
            if(initialization_source=="rtk"){
                auto it=std::upper_bound(rtk.begin(),rtk.end(),us,[](int64_t t,const Fix& fix){return t<fix.us;});
                if(it==rtk.begin()||us-std::prev(it)->us>100000){seeds.pop_front();return "waiting_rtk";}
                const auto fix=*std::prev(it);initial_position=fix.p;
                fix_id=fix.id;fix_age_s=(us-fix.us)*1e-6;
                offset=fix.p-p;
            }else if(ground_radar_height){
                // XY is template-relative. Z is raw radar height above the
                // configured pre-takeoff radar reference, not a GNSS height.
                initial_position.z()=c.radar_height_m-*ground_radar_height;
                offset=initial_position-p;
            }
            // Both the conditioned pose and raw radar updates use the same
            // translated vertical datum, including a configured ground reference.
            radar_offset=initial_position.z()-c.radar_height_m;
            VectorXd x=VectorXd::Zero(16);x.head<3>()=initial_position;
            V mean=V::Zero();double tm=0;for(auto& s:seeds){mean+=s.p;tm+=(s.us-us)*1e-6;}
            mean/=seeds.size();tm/=seeds.size();double denom=0;
            for(auto& s:seeds){double d=(s.us-us)*1e-6-tm;x.segment<3>(3)+=d*(s.p-mean);denom+=d*d;}
            x.segment<3>(3)/=denom;Quaterniond q(rot);x.segment<4>(6)<<q.w(),q.x(),q.y(),q.z();
            x.tail<3>()=pkt.initial_gyro_bias;
            M P=M::Identity(15,15);P.block<3,3>(3,3)*=4;P.block<3,3>(6,6)=cov.bottomRightCorner(3,3);
            P.block<3,3>(9,9)*=.04;P.bottomRightCorner(3,3)*=.0001;
            f=std::make_unique<eskf::ESKF_VIO>(x,P,params);
            if(initialization_source=="rtk")
                events<<us-origin<<",rtk_init,"<<fix_id<<",accepted,"<<fix_age_s<<'\n';
            else
                events<<us-origin<<",pose_init,"<<pkt.image_info.source_frame_index<<",accepted,"<<radar_offset<<'\n';
            if(ground_radar_height)
                events<<us-origin<<",radar_ground_reference,-1,accepted,"<<*ground_radar_height<<'\n';
            rtk.clear();seeds.clear();last=last_pose=us;measured=initial_position;return "initialized";
        }
        measured=p+offset;
        Quaterniond q(f->x(6),f->x(7),f->x(8),f->x(9));
        AngleAxisd aa(q.toRotationMatrix().transpose()*rot);
        if(aa.angle()>CV_PI/3)return "orientation_conflict";
        VectorXd residual(6);residual<<measured-f->x.head<3>(),aa.axis()*aa.angle();
        M H=M::Zero(6,15);H.topLeftCorner(3,3).setIdentity();H.block<3,3>(3,6).setIdentity();
        Matrix3d a=eskf::skew(residual.tail<3>());double th=aa.angle();
        double b=th<1e-5?1./12.:(1.-.5*th/std::tan(.5*th))/(th*th);
        M tangent=M::Identity(6,6);tangent.bottomRightCorner(3,3)+=.5*a+b*a*a;
        cov=tangent*cov*tangent.transpose();
        H.block<3,3>(3,6)+=-.5*a+b*a*a;
        M hp=H*f->P*H.transpose();
        if(residual.dot(inverseSpd(2*(cov+hp))*residual)>36)return "innovation_conflict";
        // Determinant lemma: compare the 15-state log determinants using 6x6 factors.
        double best=0;weight=1;double logR=logdet(cov);
        for(int k=1;k<20;++k){double w=k*.05;
            double delta=-15*std::log(w)+logR-logdet(cov+(1-w)/w*hp);
            if(delta<best){best=delta;weight=w;}}
        if(weight==1)return "prior_preferred";
        M prior=f->P/weight,noise=cov/(1-weight);
        M K=prior*H.transpose()*inverseSpd(H*prior*H.transpose()+noise);
        VectorXd dx=K*residual;M A=M::Identity(15,15)-K*H;
        M updated=A*prior*A.transpose()+K*noise*K.transpose();
        f->x.head<6>()+=dx.head<6>();f->x.tail<6>()+=dx.tail<6>();
        f->x.segment<4>(6)=eskf::quat_mul(f->x.segment<4>(6),eskf::quat_exp(dx.segment<3>(6))).normalized();
        M reset=M::Identity(15,15);reset.block<3,3>(6,6)-=.5*eskf::skew(dx.segment<3>(6));
        M next=reset*updated*reset.transpose();f->P=(next+next.transpose())*.5;
        if(!f->x.allFinite()||!f->P.allFinite()||LLT<M>(f->P).info()!=Success)throw std::runtime_error("invalid CI posterior");
        last_pose=us;return "ci_update";
    }
    void step(const input::SensorPacket& pkt,const RadarPoseResult& c){
        using clock=std::chrono::steady_clock;
        const auto begin=clock::now();
        const int64_t us=calendar(pkt.image_info.source_timestamp);
        if(us<=previous_frame)throw std::runtime_error("out of order flight frame");
        previous_frame=us;if(origin<0)origin=us;
        std::vector<E> ordered=pkt.imu_events;
        ordered.insert(ordered.end(),pkt.radar_events.begin(),pkt.radar_events.end());
        std::stable_sort(ordered.begin(),ordered.end(),[](const E& a,const E& b){return a.us<b.us;});
        for(auto& e:ordered){if(e.us>us)throw std::runtime_error("future input");sensor(e);}
        advance(us);double w=-1;V m=V::Constant(std::numeric_limits<double>::quiet_NaN());
        const auto predicted=clock::now();
        auto status=correct(c,pkt,us,m,w);const auto frame=pkt.image_info.source_frame_index;
        timing<<frame<<','<<std::chrono::duration<double,std::micro>(predicted-begin).count()<<','
              <<std::chrono::duration<double,std::micro>(clock::now()-predicted).count()<<'\n';
        events<<us-origin<<",pnp,"<<frame<<','<<status<<','<<w<<'\n';
        states<<frame<<','<<us-origin<<','<<status<<','<<(!f?"uninitialized":us-last_pose>1000000?"stale":us-last_pose>500000?"degraded":"normal");
        for(int i=0;i<16;++i){states<<',';if(f)states<<f->x(i);}
        for(int i=0;i<3;++i){states<<',';if(f)states<<f->P(i,i);}
        states<<','<<w;
        for(int i=0;i<3;++i){states<<',';if(m.allFinite())states<<m(i);}
        states<<','<<(pkt.imu_alignment?pkt.imu_alignment->sample_index:-1)<<','<<c.radar_sample_id<<'\n';
    }
};
ConditionedFlightFusion::ConditionedFlightFusion(const cv::FileNode& c,const std::string& out):impl_(std::make_unique<Impl>(c,out)){}
ConditionedFlightFusion::~ConditionedFlightFusion()=default;
void ConditionedFlightFusion::step(const input::SensorPacket& p,const RadarPoseResult& c){impl_->step(p,c);}
}
