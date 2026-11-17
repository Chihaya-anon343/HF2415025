#pragma once
#include "input/IStereoImageSource.hpp"
#include "input/InputConfig.hpp"
#include "input/VideoSeekIndex.hpp"
#include <opencv2/videoio.hpp>
#include <vector>

namespace gpnp::input {

ImageInputConfig readVideoInputConfig(const cv::FileNode& node);

class VideoSource : public IStereoImageSource {
public:
    bool open(const std::string& path) override;
    bool open(const ImageInputConfig& config);
    bool nextFrame(cv::Mat& left,cv::Mat& right,int64_t& timestamp_us) override;
    void close() override;
    bool isOpen() const override { return cap_.isOpened(); }
    // Raw streams do not provide a trustworthy frame count before EOF.
    int totalFrames() const override;
    int currentFrame() const override { return current_index_; }
    bool reset() override;
    ImageFrameInfo frameInfo() const override { return info_; }

private:
    struct Stamp {std::string text;int segment{0};};
    void loadTimestamps();
    bool decode(cv::Mat& frame);
    void openSequential();
    VideoSeekIndex seek_index_;
    cv::Mat first_frame_;
    cv::VideoCapture cap_;
    ImageInputConfig config_;
    std::vector<Stamp> stamps_;
    ImageFrameInfo info_;
    double fps_{0};
    int next_index_{0}, selected_index_{0}, current_index_{-1};
    int actual_total_{-1};
    bool ended_{false};
    int64_t skipped_{0};
    int64_t decoded_{0};
};
}
