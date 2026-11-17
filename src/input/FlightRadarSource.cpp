#include "input/FlightRadarSource.hpp"
#include "input/FlightImuSource.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace gpnp::input {
namespace {
std::vector<std::string> columns(std::string s) {
    if(!s.empty()&&s.back()=='\r')s.pop_back();
    std::vector<std::string> result;std::string part;std::istringstream stream(s);
    while(std::getline(stream,part,','))result.push_back(part);
    if(!s.empty()&&s.back()==',')result.push_back("");
    return result;
}
std::string calendar(std::string s) {
    if(s.size()==23 && s[4]=='-'&&s[10]==' '&&s[19]=='.') {
        for(int i:{4,7,10,13,16,19})s[i]='_';
    }
    return s;
}
}
AltimeterInputConfig readAltimeterInputConfig(const cv::FileNode& n) {
    AltimeterInputConfig c;if(n.empty())return c;
    if(!n.isMap())throw std::invalid_argument("altimeter must be an object");
    auto e=n["enabled"];if(!e.empty()) {
        if(!e.isInt()||(int(e)!=0&&int(e)!=1))throw std::invalid_argument("altimeter.enabled must be boolean");
        c.enabled=int(e)!=0;
    }
    for(auto item:{std::pair<const char*,std::string*>{"type",&c.type},{"path",&c.path},{"timestamp_column",&c.timestamp_column},
        {"height_column",&c.height_column},{"can_interface",&c.can_interface}}) {
        auto v=n[item.first];if(v.empty())continue;
        if(!v.isString())throw std::invalid_argument("Invalid altimeter string field");
        *item.second=std::string(v);
    }
    for(auto item:{std::pair<const char*,double*>{"max_age_s",&c.max_age_s},{"min_height_m",&c.min_height_m},{"max_height_m",&c.max_height_m},{"max_vertical_speed_m_s",&c.max_vertical_speed_m_s},{"jump_margin_m",&c.jump_margin_m}}) {
        auto v=n[item.first];if(v.empty())continue;
        if(!v.isReal()&&!v.isInt())throw std::invalid_argument("Invalid altimeter numeric field");
        *item.second=double(v);
    }
    return c;
}
FlightRadarSource::FlightRadarSource(const AltimeterInputConfig& c):config_(c) {
    if(c.type!="flight_csv"||c.path.empty()||c.timestamp_column.empty()||c.height_column.empty()||
       !std::isfinite(c.max_age_s)||c.max_age_s<=0||c.max_age_s>1||
       !std::isfinite(c.min_height_m)||!std::isfinite(c.max_height_m)||c.min_height_m<0||c.min_height_m>=c.max_height_m||
       !std::isfinite(c.max_vertical_speed_m_s)||c.max_vertical_speed_m_s<=0||!std::isfinite(c.jump_margin_m)||c.jump_margin_m<0)
        throw std::invalid_argument("Invalid radar file configuration");
    std::ifstream file(c.path);if(!file)throw std::invalid_argument("Cannot open radar CSV: "+c.path);
    std::string line;if(!std::getline(file,line))throw std::invalid_argument("Empty radar CSV");
    if(line.compare(0,3,"\xEF\xBB\xBF")==0)line.erase(0,3);
    const auto header=columns(line);std::map<std::string,size_t> fields;
    for(size_t i=0;i<header.size();++i)if(!fields.emplace(header[i],i).second)throw std::invalid_argument("Duplicate radar CSV column");
    if(!fields.count(c.timestamp_column)||!fields.count(c.height_column))throw std::invalid_argument("Radar CSV missing required columns");
    int64_t row=1;
    while(std::getline(file,line)) {
        ++row;if(line.empty()||line=="\r")continue;
        const auto cols=columns(line);if(cols.size()!=header.size())throw std::invalid_argument("Malformed radar CSV row");
        auto text=cols[fields[c.height_column]];size_t used=0;double h=std::stod(text,&used);
        if(used!=text.size()||!std::isfinite(h))throw std::invalid_argument("Invalid radar height number");
        const auto stamp=calendar(cols[fields[c.timestamp_column]]);const auto t=parseFlightCalendarUs(stamp);
        if(!samples_.empty()&&t<samples_.back().time)throw std::invalid_argument("Radar timestamp moved backwards");
        if(!samples_.empty()&&t==samples_.back().time) {
            auto& s=samples_.back();s.height=(s.height*s.merged+h)/(s.merged+1);++s.merged;
        }else samples_.push_back({t,h,stamp,row,1});
    }
    if(samples_.empty())throw std::invalid_argument("Radar CSV has no data");
    const Sample* trusted=nullptr;
    for(auto& s:samples_) {
        if(!std::isfinite(s.height))throw std::invalid_argument("Merged radar height overflow");
        if(s.height<c.min_height_m||s.height>c.max_height_m)continue;
        if(trusted && std::abs(s.height-trusted->height)>c.jump_margin_m+c.max_vertical_speed_m_s*(s.time-trusted->time)*1e-6) {
            s.jump_valid=false;continue; // A spike must not become the next trusted reference.
        }
        trusted=&s;
    }
}
void FlightRadarSource::synchronize(SensorPacket& packet) {
    RadarHeightPrior p;const auto query=parseFlightCalendarUs(packet.image_info.source_timestamp);
    if(last_query_>=0&&query<=last_query_)throw std::runtime_error("Radar query clock not increasing; reset first");
    const auto previous=last_query_;last_query_=query;
    auto end=std::upper_bound(samples_.begin(),samples_.end(),query,[](int64_t t,const Sample& s){return t<s.time;});
    packet.radar_events.clear();
    if(config_.full_rate_replay && end!=samples_.begin()) {
        auto begin=previous<0?std::prev(end):std::upper_bound(samples_.begin(),end,previous,[](int64_t t,const Sample& s){return t<s.time;});
        for(auto it=begin;it!=end;++it)if(it->jump_valid && it->height>=config_.min_height_m && it->height<=config_.max_height_m)
            packet.radar_events.push_back({it->time,int64_t(it-samples_.begin()),true,{0,0,0},{0,0,0},it->height});
    }
    p.status="before_radar_range";
    if(end!=samples_.begin()) {
        const auto& s=*std::prev(end);p.age_s=(query-s.time)*1e-6;p.t_sample=packet.timestamp_us*1e-6-p.age_s;
        p.sample_id=std::distance(samples_.begin(),std::prev(end));p.raw_row=s.row;p.merged_rows=s.merged;
        p.height_m=s.height;p.stamp=s.stamp;
        p.status=p.age_s>config_.max_age_s+1e-9?"stale_radar":s.height<config_.min_height_m||s.height>config_.max_height_m?"height_out_of_range":!s.jump_valid?"height_jump":"ready";
        p.valid=p.status=="ready";
    }
    packet.radar_prior=p;
    // Do not silently feed this radar-conditioned stream to the legacy ESKF height channel.
}
}
