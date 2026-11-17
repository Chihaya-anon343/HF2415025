#pragma once
#include "input/InputConfig.hpp"
#include "input/SensorTypes.hpp"
#include <vector>

namespace gpnp::input {
// Same calendar coordinate for both recorder files; does not assume UTC/local TZ.
int64_t parseFlightCalendarUs(const std::string& text);
ImuInputConfig readImuInputConfig(const cv::FileNode& node);

class FlightImuSource {
public:
    explicit FlightImuSource(const ImuInputConfig& config);
    void synchronize(SensorPacket& packet);
    void reset();
    size_t samples() const {return samples_.size();}
    size_t duplicateRows() const {return duplicates_;}
private:
    struct Sample {
        int64_t time_us;
        Eigen::Vector3d gyro,acc;
        double roll,pitch;
        std::string stamp;
        size_t row,merged{1};
        bool angles_valid{true};
    };
    ImuInputConfig config_;
    Eigen::Vector3d initial_bias_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation_;
    std::vector<Sample> samples_;
    size_t duplicates_{0},cursor_{0};
    bool started_{false};
    int64_t last_query_{-1},valid_from_{0};
};
}
