#include "input/FlightImuSource.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpnp::input {
namespace {
constexpr double rad=3.14159265358979323846/180.;
std::string trim(std::string s) {
    const auto b=s.find_first_not_of(" \t\r\n");
    if(b==std::string::npos)return {};
    return s.substr(b,s.find_last_not_of(" \t\r\n")-b+1);
}
double number(const std::string& text) {
    size_t used=0;const auto value=std::stod(trim(text),&used);
    if(used!=trim(text).size()||!std::isfinite(value))throw std::invalid_argument("Invalid flight IMU number");
    return value;
}
}
int64_t parseFlightCalendarUs(const std::string& s) {
    if(s.size()!=23)throw std::invalid_argument("Flight timestamp requires YYYY_MM_DD_HH_MM_SS_mmm");
    for(size_t i=0;i<s.size();++i) {
        const bool sep=i==4||i==7||i==10||i==13||i==16||i==19;
        if(sep?s[i]!='_':(s[i]<'0'||s[i]>'9'))throw std::invalid_argument("Invalid flight timestamp");
    }
    const int y=std::stoi(s.substr(0,4)),m=std::stoi(s.substr(5,2)),d=std::stoi(s.substr(8,2));
    const int h=std::stoi(s.substr(11,2)),n=std::stoi(s.substr(14,2)),sec=std::stoi(s.substr(17,2)),ms=std::stoi(s.substr(20,3));
    const bool leap=y%4==0&&(y%100!=0||y%400==0);const int month[]={31,28,31,30,31,30,31,31,30,31,30,31};
    if(y<1970||m<1||m>12||d<1||d>month[m-1]+(m==2&&leap)||h>23||n>59||sec>59)
        throw std::invalid_argument("Invalid flight calendar date");
    const int64_t z=y-1;int64_t days=365*z+z/4-z/100+z/400+d-1;
    for(int j=1;j<m;++j)days+=month[j-1]+(j==2&&leap);
    return (((days*24+h)*60+n)*60+sec)*1000000+ms*1000;
}
ImuInputConfig readImuInputConfig(const cv::FileNode& node) {
    ImuInputConfig c;if(node.empty())return c;
    if(!node.isMap())throw std::invalid_argument("input_system.imu must be an object");
    if(!node["enabled"].empty()) {
        if(!node["enabled"].isInt()||(int(node["enabled"])!=0&&int(node["enabled"])!=1))throw std::invalid_argument("imu.enabled must be boolean");
        c.enabled=int(node["enabled"])!=0;
    }
    auto str=[&](const char* key,std::string& v) {if(!node[key].empty()){if(!node[key].isString())throw std::invalid_argument(std::string("imu.")+key+" must be a string");v=std::string(node[key]);}};
    str("type",c.type);str("port",c.port);str("protocol",c.protocol);
    if(!node["baud_rate"].empty())c.baud_rate=int(node["baud_rate"]);
    if(!c.enabled||c.type!="flight_csv")return c;
    str("path",c.path);str("gyro_unit",c.gyro_unit);str("acc_unit",c.acc_unit);
    str("normal_source",c.normal_source);str("tilt_order",c.tilt_order);str("tilt_unit",c.tilt_unit);
    str("duplicate_policy",c.duplicate_policy);str("attitude_valid_from",c.attitude_valid_from);
    for(auto item:{std::pair<const char*,double*>{"time_offset_s",&c.time_offset_s},{"max_sync_age_s",&c.max_sync_age_s},{"tilt_sigma_deg",&c.tilt_sigma_deg}}) {
        auto value=node[item.first];if(value.empty())continue;
        if(!value.isReal()&&!value.isInt())throw std::invalid_argument(std::string("imu.")+item.first+" must be numeric");
        *item.second=double(value);
    }
    auto r=node["imu_to_camera_rotation"];
    if(r.size()!=9)throw std::invalid_argument("flight_csv requires explicit imu_to_camera_rotation (9 values)");
    int i=0;for(auto v:r)c.imu_to_camera_rotation[i++]=double(v);
    c.has_rotation=true;
    return c;
}
FlightImuSource::FlightImuSource(const ImuInputConfig& c):config_(c) {
    if(c.path.empty()||!c.has_rotation||c.gyro_unit!="deg_s"||c.acc_unit!="g"||
       c.normal_source!="logged_tilt"||c.tilt_order!="roll_pitch"||c.tilt_unit!="deg"||
       (c.duplicate_policy!="mean"&&c.duplicate_policy!="reject")||
       !std::isfinite(c.time_offset_s)||std::abs(c.time_offset_s)>3600||
       !std::isfinite(c.max_sync_age_s)||c.max_sync_age_s<=0||c.max_sync_age_s>1||
       !std::isfinite(c.tilt_sigma_deg)||c.tilt_sigma_deg<=0||c.tilt_sigma_deg>=45)
        throw std::invalid_argument("Invalid/unsupported flight_csv IMU configuration");
    for(int i=0;i<9;++i)rotation_(i/3,i%3)=c.imu_to_camera_rotation[i];
    if(!rotation_.allFinite()||(rotation_.transpose()*rotation_-Eigen::Matrix3d::Identity()).norm()>1e-6||std::abs(rotation_.determinant()-1)>1e-6)
        throw std::invalid_argument("imu_to_camera_rotation must be a proper rotation");
    if(c.attitude_valid_from.empty())throw std::invalid_argument("logged_tilt requires explicit attitude_valid_from");
    valid_from_=parseFlightCalendarUs(c.attitude_valid_from);
    std::ifstream file(c.path);if(!file)throw std::invalid_argument("Cannot open IMU file: "+c.path);
    std::string line;size_t row=0;
    while(std::getline(file,line)) {
        ++row;line=trim(line);if(row==1&&line.compare(0,3,"\xEF\xBB\xBF")==0)line=trim(line.substr(3));
        if(line.empty())continue;
        if(samples_.empty()&&(line.rfind("timestamp,",0)==0||line.rfind("时间戳,",0)==0))continue;
        std::vector<std::string> cols;std::istringstream in(line);std::string cell;
        while(std::getline(in,cell,','))cols.push_back(trim(cell));
        if(cols.size()!=9||line.back()==',')throw std::invalid_argument("IMU row "+std::to_string(row)+" requires 9 columns");
        Sample s;s.time_us=parseFlightCalendarUs(cols[0]);s.stamp=cols[0];s.row=row;
        for(int j=0;j<3;++j){s.gyro(j)=number(cols[j+1])*rad;s.acc(j)=number(cols[j+4])*9.81;}
        s.roll=number(cols[7])*rad;s.pitch=number(cols[8])*rad;
        s.angles_valid=std::abs(s.roll)<89*rad&&std::abs(s.pitch)<89*rad;
        if(!s.gyro.allFinite()||!s.acc.allFinite())throw std::invalid_argument("IMU unit conversion overflow");
        if(!samples_.empty()&&s.time_us<samples_.back().time_us)throw std::invalid_argument("IMU timestamps moved backwards at row "+std::to_string(row));
        if(!samples_.empty()&&s.time_us==samples_.back().time_us) {
            if(c.duplicate_policy=="reject")throw std::invalid_argument("Duplicate IMU timestamps");
            auto& last=samples_.back();const double count=double(last.merged);
            last.gyro=(last.gyro*count+s.gyro)/(count+1);last.acc=(last.acc*count+s.acc)/(count+1);
            // Roll/pitch are restricted away from the Euler singularity below.
            last.roll=(last.roll*count+s.roll)/(count+1);last.pitch=(last.pitch*count+s.pitch)/(count+1);
            last.angles_valid=last.angles_valid&&s.angles_valid;
            ++last.merged;++duplicates_;continue;
        }
        samples_.push_back(s);
    }
    if(samples_.empty())throw std::invalid_argument("No IMU samples");
    if(c.full_rate_replay) {
        size_t n=0; Eigen::Vector3d acc=Eigen::Vector3d::Zero();
        double a2=0,w2=0;
        for(const auto& s:samples_) {
            if(s.time_us>=samples_.front().time_us+5000000)break;
            ++n;initial_bias_+=s.gyro;w2+=s.gyro.squaredNorm();
            acc+=s.acc;a2+=s.acc.squaredNorm();
        }
        initial_bias_/=n;acc/=n;
        if(n<100 || std::abs(acc.norm()-9.81)>.3 || a2/n-acc.squaredNorm()>.09 ||
           std::sqrt(w2/n)>3*rad)
            throw std::invalid_argument("ESKF stationary bias prefix failed qualification");
    }
}
void FlightImuSource::reset(){cursor_=0;started_=false;last_query_=-1;}
void FlightImuSource::synchronize(SensorPacket& packet) {
    packet.imu.reset();packet.gravity_prior.reset();packet.imu_alignment=ImuAlignment{};
    auto& info=*packet.imu_alignment;GravityNormalPrior prior;prior.source="logged_tilt_roll_pitch";
    info.raw_rows=samples_.size()+duplicates_;info.unique_samples=samples_.size();info.duplicate_rows=duplicates_;
    const auto frame=parseFlightCalendarUs(packet.image_info.source_timestamp);
    const auto query=frame+std::llround(config_.time_offset_s*1e6);
    if(last_query_>=0&&query<=last_query_)throw std::runtime_error("Video sensor clock is not strictly increasing; reset input before replay");
    last_query_=query;
    auto end=std::upper_bound(samples_.begin(),samples_.end(),query,[](int64_t t,const Sample& s){return t<s.time_us;});
    const size_t next=size_t(end-samples_.begin());
    if(!started_){info.skipped_prefix=next?next-1:0;cursor_=info.skipped_prefix;started_=true;}
    info.samples_since_frame=next>=cursor_?next-cursor_:0;
    packet.imu_events.clear();
    if(config_.full_rate_replay) {
        if(frame<samples_.front().time_us+5000000 || config_.time_offset_s!=0)
            throw std::invalid_argument("ESKF needs an earlier stationary prefix and aligned input");
        packet.initial_gyro_bias=initial_bias_;
        for(size_t i=cursor_;i<next;++i) {
            const auto& s=samples_[i];
            packet.imu_events.push_back({s.time_us,int64_t(i),false,s.acc,s.gyro,0});
        }
    }
    cursor_=next;
    prior.status="before_imu_range";
    if(next) {
        const auto& s=samples_[next-1];info.sample_index=next-1;info.raw_row=s.row;info.imu_stamp=s.stamp;
        info.age_s=(query-s.time_us)*1e-6;info.acc=s.acc;info.gyro=s.gyro;info.roll_deg=s.roll/rad;info.pitch_deg=s.pitch/rad;
        const double logical=packet.timestamp_us*1e-6;
        prior.t_sample=logical-info.age_s;
        prior.status=info.age_s>config_.max_sync_age_s+1e-9?"stale_imu":s.time_us<valid_from_?"attitude_not_ready":"ready";
        if(!s.angles_valid)prior.status="tilt_out_of_range";
        if(prior.status=="ready") {
            const Eigen::Vector3d down(std::sin(s.pitch),-std::sin(s.roll)*std::cos(s.pitch),-std::cos(s.roll)*std::cos(s.pitch));
            prior.normal_camera=(rotation_*down).normalized();
            prior.covariance_rad2=std::pow(config_.tilt_sigma_deg*rad,2)*(Eigen::Matrix3d::Identity()-prior.normal_camera*prior.normal_camera.transpose());
            prior.valid=true;info.valid=true;
            ImuData imu;imu.t=prior.t_sample;imu.acc=s.acc;imu.gyro=s.gyro;
            imu.angle={s.roll/rad,s.pitch/rad,std::numeric_limits<double>::quiet_NaN()};packet.imu=imu;
        }
    }
    info.status=prior.status;packet.gravity_prior=prior;
}
}
