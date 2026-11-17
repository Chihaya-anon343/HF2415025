#include "input/VideoSource.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace gpnp::input {
namespace {
int64_t civilMillis(int year,int month,int day,int hour,int minute,int second,int millis) {
    const bool leap=year%4==0 && (year%100!=0 || year%400==0);
    const int lengths[]={31,28,31,30,31,30,31,31,30,31,30,31};
    if(year<1 || month<1 || month>12 || day<1 ||
       day>lengths[month-1]+(month==2 && leap) || hour>23 || minute>59 || second>59)
        throw std::invalid_argument("Invalid video timestamp calendar date");
    const int64_t y=year-1;
    int64_t days=365*y+y/4-y/100+y/400+day-1;
    for(int m=1;m<month;++m)days+=lengths[m-1]+(m==2 && leap);
    // Calendar coordinate for detecting discontinuities, not a UTC conversion.
    return ((days*24+hour)*60+minute)*60000+second*1000+millis;
}
}

ImageInputConfig readVideoInputConfig(const cv::FileNode& node) {
    ImageInputConfig c;c.type=ImageSourceType::Video;
    c.video_path=std::string(node["video_path"]);
    if(!node["timestamp_path"].empty())c.timestamp_path=std::string(node["timestamp_path"]);
    if(!node["timestamp_mode"].empty())c.timestamp_mode=std::string(node["timestamp_mode"]);
    if(!node["seek_mode"].empty())c.seek_mode=std::string(node["seek_mode"]);
    if(!node["indexed_video_path"].empty())c.indexed_video_path=std::string(node["indexed_video_path"]);
    if(!node["video_fps"].empty()) {
        if(!node["video_fps"].isReal() && !node["video_fps"].isInt())throw std::invalid_argument("video_fps must be numeric");
        c.video_fps=double(node["video_fps"]);
    }
    auto integer=[&](const char* key,int& target) {
        if(!node[key].empty()) {
            if(!node[key].isInt())throw std::invalid_argument(std::string(key)+" must be an integer");
            target=int(node[key]);
        }
    };
    integer("start_frame",c.start_frame);integer("frame_step",c.frame_step);
    if(c.video_path.empty() || c.start_frame<0 || c.frame_step<1 || !std::isfinite(c.video_fps) ||
       c.video_fps<0 || c.timestamp_mode!="frame_index" ||
       (c.seek_mode!="sequential" && c.seek_mode!="auto" && c.seek_mode!="indexed") ||
       (c.seek_mode=="indexed" && c.indexed_video_path.empty()))throw std::invalid_argument("Invalid offline video configuration");
    return c;
}

void VideoSource::loadTimestamps() {
    stamps_.clear();
    if(config_.timestamp_path.empty())return;
    std::ifstream input(config_.timestamp_path);
    if(!input)throw std::invalid_argument("Cannot open video timestamp file: "+config_.timestamp_path);
    const std::regex pattern(R"(^([0-9]+)_([0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{3})$)");
    std::string line;int segment=0;int64_t last=0;
    // Recorder files can switch from CRLF to bare CR after the frame counter
    // gains a digit. Whitespace tokenization supports CR, LF and CRLF uniformly.
    while(input>>line) {
        if(stamps_.empty() && line.compare(0,3,"\xEF\xBB\xBF")==0)line.erase(0,3);
        if(!line.empty() && line.back()=='\r')line.pop_back();
        std::smatch match;
        if(!std::regex_match(line,match,pattern) || std::stoll(match[1])!=int64_t(stamps_.size())+1)
            throw std::invalid_argument("Video timestamps must be consecutive 1-based rows");
        const std::string text=match[2];
        const int64_t t=civilMillis(std::stoi(text.substr(0,4)),std::stoi(text.substr(5,2)),
            std::stoi(text.substr(8,2)),std::stoi(text.substr(11,2)),std::stoi(text.substr(14,2)),
            std::stoi(text.substr(17,2)),std::stoi(text.substr(20,3)));
        if(!stamps_.empty() && (t<=last || t-last>1000)) {
            ++segment;
            std::cout<<"[VideoSource] clock discontinuity at source frame "<<stamps_.size()
                     <<", segment="<<segment<<"; preserving raw timestamp\n";
        }
        stamps_.push_back({text,segment});last=t;
    }
    if(stamps_.empty())throw std::invalid_argument("Video timestamp file is empty");
}

bool VideoSource::open(const std::string& path) {
    ImageInputConfig config;config.type=ImageSourceType::Video;config.video_path=path;
    return open(config);
}

bool VideoSource::open(const ImageInputConfig& config) {
    close();config_=config;
    const auto begin=std::chrono::steady_clock::now();
    try {
        if(config.video_path.empty() || config.start_frame<0 || config.frame_step<1 ||
           !std::isfinite(config.video_fps) || config.video_fps<0 || config.timestamp_mode!="frame_index" ||
           (config.seek_mode!="sequential" && config.seek_mode!="auto" && config.seek_mode!="indexed") ||
           (config.seek_mode=="indexed" && config.indexed_video_path.empty()))
            throw std::invalid_argument("Invalid video path/start/step/fps/timestamp_mode");
        loadTimestamps();
        if(!stamps_.empty() && config.start_frame>=int(stamps_.size()))
            throw std::invalid_argument("Video start_frame is beyond timestamp records");
        bool indexed=false;
        if(config.seek_mode!="sequential" && !config.indexed_video_path.empty() &&
           (config.seek_mode=="indexed" || config.start_frame>0)) {
            try {
                seek_index_=VideoSeekIndex::load(config.video_path,config.indexed_video_path,config.video_fps);
                if(!stamps_.empty() && stamps_.size()!=seek_index_.frame_hashes.size())
                    throw std::runtime_error("Video seek manifest/timestamp count mismatch");
                if(size_t(config.start_frame)>=seek_index_.frame_hashes.size())
                    throw std::runtime_error("Video start_frame beyond indexed video");
                if(!cap_.open(config.indexed_video_path,cv::CAP_FFMPEG) ||
                   (config.start_frame>0 && !cap_.set(cv::CAP_PROP_POS_FRAMES,config.start_frame)) ||
                   !cap_.read(first_frame_))throw std::runtime_error("Indexed decoder could not seek/read");
                const double position=cap_.get(cv::CAP_PROP_POS_FRAMES);
                if(first_frame_.cols!=seek_index_.width || first_frame_.rows!=seek_index_.height ||
                   first_frame_.type()!=CV_8UC3 ||
                   videoImageHash(first_frame_)!=seek_index_.frame_hashes[config.start_frame] ||
                   !std::isfinite(position) || std::abs(position-(double(config.start_frame)+1))>.01)
                    throw std::runtime_error("Indexed seek frame verification failed");
                fps_=seek_index_.fps;next_index_=config.start_frame;skipped_=config.start_frame;
                info_.seek_method="indexed_verified";info_.seek_bypassed_frames=config.start_frame;indexed=true;
            } catch(const std::exception& e) {
                if(config.seek_mode=="indexed")throw;
                std::cerr<<"[VideoSource] indexed seek unavailable: "<<e.what()<<"; sequential fallback\n";
                cap_.release();first_frame_.release();seek_index_={};
            }
        }
        if(!indexed)openSequential();
        if(!std::isfinite(fps_) || fps_<=0)
            throw std::invalid_argument("Video frame rate unknown; set video_fps explicitly");
        selected_index_=config.start_frame;
        info_.startup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::cout<<"[VideoSource] opened "<<config.video_path<<" "<<cap_.get(cv::CAP_PROP_FRAME_WIDTH)
                 <<"x"<<cap_.get(cv::CAP_PROP_FRAME_HEIGHT)<<" fps="<<fps_
                 <<" timestamp_rows="<<stamps_.size()<<" logical_clock=frame_index"
                 <<" start_frame="<<config.start_frame<<" step="<<config.frame_step<<" seek="<<info_.seek_method<<'\n';
        return true;
    } catch(const std::exception& e) {
        std::cerr<<"[VideoSource] "<<e.what()<<'\n';close();return false;
    }
}

void VideoSource::openSequential() {
    if(!cap_.open(config_.video_path,cv::CAP_FFMPEG))
        throw std::runtime_error("Cannot open video with FFmpeg: "+config_.video_path);
    fps_=config_.video_fps>0?config_.video_fps:cap_.get(cv::CAP_PROP_FPS);
    info_.seek_method="sequential";
}

bool VideoSource::decode(cv::Mat& frame) {
    bool ok;
    if(!first_frame_.empty()){frame=first_frame_;first_frame_.release();ok=true;}
    else ok=cap_.read(frame);
    if(!ok) {
        ended_=true;actual_total_=next_index_;
        if(!seek_index_.frame_hashes.empty() && size_t(next_index_)!=seek_index_.frame_hashes.size())
            throw std::runtime_error("Indexed video frame count changed");
        if(!stamps_.empty() && next_index_!=int(stamps_.size()))
            throw std::runtime_error("Video ended before timestamp records: decoded="+
                std::to_string(next_index_)+" timestamps="+std::to_string(stamps_.size()));
        return false;
    }
    if(!stamps_.empty() && next_index_>=int(stamps_.size()))
        throw std::runtime_error("Video has more frames than timestamp records");
    if(!seek_index_.frame_hashes.empty() && (size_t(next_index_)>=seek_index_.frame_hashes.size()))
        throw std::runtime_error("Indexed video has extra frames");
    ++next_index_;++decoded_;return true;
}

bool VideoSource::nextFrame(cv::Mat& left,cv::Mat& right,int64_t& timestamp_us) {
    if(!isOpen() || ended_)return false;
    const auto begin=std::chrono::steady_clock::now();
    cv::Mat frame;
    while(next_index_<=selected_index_) {
        if(!decode(frame))return false;
        if(next_index_-1!=selected_index_)++skipped_;
    }
    current_index_=selected_index_;
    if(selected_index_>std::numeric_limits<int>::max()-config_.frame_step)
        throw std::runtime_error("Video frame index overflow");
    selected_index_+=config_.frame_step;
    left=frame;right=frame.clone();
    const double t=current_index_*1e6/fps_;
    if(t>double(std::numeric_limits<int64_t>::max())-1e6)
        throw std::runtime_error("Video logical timestamp overflow");
    timestamp_us=std::llround(t);
    info_.source_frame_index=current_index_;
    info_.source_timestamp=stamps_.empty()?"":stamps_[current_index_].text;
    info_.clock_segment=stamps_.empty()?0:stamps_[current_index_].segment;
    info_.timestamp_quality="video_frame_index_logical";
    info_.decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    if(current_index_==config_.start_frame)info_.startup_ms+=info_.decode_ms;
    info_.decoded_frames=decoded_;info_.skipped_frames=skipped_;
    return true;
}

int VideoSource::totalFrames() const {
    if(actual_total_<0)return -1;
    if(actual_total_<=config_.start_frame)return 0;
    return 1+(actual_total_-1-config_.start_frame)/config_.frame_step;
}
void VideoSource::close() {
    cap_.release();first_frame_.release();seek_index_={};stamps_.clear();info_={};fps_=0;next_index_=selected_index_=0;
    current_index_=-1;actual_total_=-1;ended_=false;skipped_=decoded_=0;
}
bool VideoSource::reset() {const auto config=config_;return open(config);}
}
