#pragma once
#include <cstdint>
#include <string>

namespace gpnp::input {
// Optional source metadata; legacy image/camera sources retain empty defaults.
struct ImageFrameInfo {
    int64_t source_frame_index{-1};
    std::string source_timestamp;
    std::string timestamp_quality{"unspecified"};
    int clock_segment{0};
    double decode_ms{0};
    int64_t decoded_frames{0}, skipped_frames{0};
    std::string seek_method{"sequential"};
    double startup_ms{0};
    int64_t seek_bypassed_frames{0};
};
}
