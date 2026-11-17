#pragma once
#include "input/SensorTypes.hpp"
#include "input/InputConfig.hpp"
#include <vector>

namespace gpnp::input {
AltimeterInputConfig readAltimeterInputConfig(const cv::FileNode&);
class FlightRadarSource {
public:
    explicit FlightRadarSource(const AltimeterInputConfig&);
    void synchronize(SensorPacket&);
    void reset(){last_query_=-1;}
    size_t samples()const{return samples_.size();}
private:
    struct Sample {int64_t time;double height;std::string stamp;int64_t row;size_t merged{1};bool jump_valid{true};};
    AltimeterInputConfig config_;
    std::vector<Sample> samples_;
    int64_t last_query_{-1};
};
}
