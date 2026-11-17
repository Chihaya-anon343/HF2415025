#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gpnp::input {
// Content fingerprints detect stale caches and wrong seek positions; not authentication.
inline uint64_t videoHashBytes(const unsigned char* p,size_t n,uint64_t h=14695981039346656037ULL) {
    // Consume all pixels in little-endian 64-bit words, avoiding a dependency chain per byte.
    while(n>=8) {
        const uint64_t word=uint64_t(p[0])|(uint64_t(p[1])<<8)|(uint64_t(p[2])<<16)|(uint64_t(p[3])<<24)|
            (uint64_t(p[4])<<32)|(uint64_t(p[5])<<40)|(uint64_t(p[6])<<48)|(uint64_t(p[7])<<56);
        h=(h^word)*1099511628211ULL;p+=8;n-=8;
    }
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;
}
inline std::string videoHashText(uint64_t h) {
    std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return out.str();
}
inline std::string videoFileHash(const std::string& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Cannot fingerprint video: "+path);
    std::array<unsigned char,65536> buffer{};uint64_t h=14695981039346656037ULL;
    while(file){file.read(reinterpret_cast<char*>(buffer.data()),buffer.size());h=videoHashBytes(buffer.data(),size_t(file.gcount()),h);}
    if(!file.eof())throw std::runtime_error("Video fingerprint read failed: "+path);
    return videoHashText(h);
}
inline std::string videoImageHash(const cv::Mat& image) {
    uint64_t h=14695981039346656037ULL;
    for(int y=0;y<image.rows;++y)h=videoHashBytes(image.ptr<unsigned char>(y),image.cols*image.elemSize(),h);
    return videoHashText(h);
}
struct VideoSeekIndex {
    double fps{0};
    int width{0},height{0};
    std::vector<std::string> frame_hashes;
    static VideoSeekIndex load(const std::string& source,const std::string& cache,double fps) {
        cv::FileStorage fs(cache+".index.json",cv::FileStorage::READ);
        if(!fs.isOpened() || int(fs["version"])!=1)throw std::runtime_error("Missing/unsupported video seek manifest");
        if(std::string(fs["source_hash"])!=videoFileHash(source) ||
           std::string(fs["cache_hash"])!=videoFileHash(cache))throw std::runtime_error("Video seek cache is stale or changed");
        VideoSeekIndex index;index.fps=double(fs["fps"]);index.width=int(fs["width"]);index.height=int(fs["height"]);
        if(!std::isfinite(index.fps) || index.fps<=0 || index.width<=0 || index.height<=0 ||
           (fps>0 && std::abs(index.fps-fps)>1e-6))throw std::runtime_error("Video seek cache geometry/fps mismatch");
        auto hashes=fs["frame_hashes"];
        if(!hashes.isSeq() || hashes.empty())throw std::runtime_error("Video seek cache has no frame fingerprints");
        for(auto h:hashes) {
            auto text=std::string(h);
            if(text.size()!=16 || text.find_first_not_of("0123456789abcdef")!=std::string::npos)
                throw std::runtime_error("Invalid video frame fingerprint");
            index.frame_hashes.push_back(text);
        }
        return index;
    }
};
}
