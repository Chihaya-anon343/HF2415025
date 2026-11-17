#pragma once

#include "common/Types.hpp"
#include "common/TemplateGeometry.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>

namespace gpnp {

// Annotation rows are stable IDs. All geometry is expressed in the class0
// white-outline bbox frame (+X right, +Y down, +Z into the board), in mm.
struct TargetClassModel {
    std::vector<cv::Point2f> image_points;
    std::vector<Eigen::Vector3d> bc_points;
    std::vector<Eigen::Vector3d> tt_points; // fixed board-axis TL/TR/BR/BL reference box
    cv::Point2d center_px;
    Eigen::Vector2d white_size_mm{Eigen::Vector2d::Zero()};
    std::vector<TemplateData> templates; // generated once, never read from old angle files
};

struct TargetModel {
    std::array<TargetClassModel, 2> classes;
    cv::Size image_size;
    Eigen::Vector2d board_size_mm;
    Eigen::Vector2d mm_per_pixel;
    cv::Point2d origin_px;
    std::string image_path;
    std::array<std::string, 2> annotation_paths;
    std::string version{"board_annotations_v1"};
};

inline bool readUnifiedTargetModelEnabled(const cv::FileNode& root) {
    const auto node = root["target_model"];
    if (node.empty()) return false;
    if (!node.isMap()) throw std::invalid_argument("target_model must be an object");
    const auto enabled = node["enabled"];
    if (enabled.empty()) return false;
    if (!enabled.isInt() || (int(enabled) != 0 && int(enabled) != 1))
        throw std::invalid_argument("target_model.enabled must be boolean");
    if (!int(enabled)) return false;
    // Multiple active physical definitions are a configuration error, not a
    // reason to silently fall back to an older model.
    for (const char* key : {"pixel_to_meter_scale_class0", "pixel_to_meter_scale_class1"})
        if (!root["strategies"]["binary_corner"][key].empty())
            throw std::invalid_argument(std::string("target_model: remove independent ") + key);
    for (const char* key : {"square_size_m_class0", "square_size_m_class1"})
        if (!root["strategies"]["tiny_target"][key].empty())
            throw std::invalid_argument(std::string("target_model: remove independent ") + key);
    if (!root["template_geometry"]["class1_center_in_class0_mm"].empty())
        throw std::invalid_argument("target_model: class1 center is derived; remove manual offset");
    const auto common = root["template_geometry"]["enabled"];
    if (!common.empty() && (!common.isInt() || int(common) != 1))
        throw std::invalid_argument("target_model requires the common class0 frame");
    return true;
}

inline TemplateGeometryConfig readTargetFrame(const cv::FileNode& root) {
    if(readUnifiedTargetModelEnabled(root)) {
        TemplateGeometryConfig c;c.enabled=true;return c;
    }
    return readTemplateGeometryConfig(root["template_geometry"]);
}

inline void configureTargetModel(TrackerConfig& c,const cv::FileNode& root) {
    c.unified_target_model=readUnifiedTargetModelEnabled(root);
    c.template_geometry=readTargetFrame(root);
    if(c.unified_target_model) {
        auto b=root["strategies"]["board_template"],d=root["strategies"]["dual_roi"];
        c.template_real_width_mm=double(b["template_real_width_mm"]);
        c.template_real_height_mm=double(b["template_real_height_mm"]);
        c.dual_roi_class0_points_path=std::string(d["class0_points_path"]);
        c.dual_roi_class1_points_path=std::string(d["class1_points_path"]);
    }
}

inline std::vector<cv::Point2f> readTargetAnnotation(const std::string& path,
        const std::string& image_path, cv::Size size) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("target_model: cannot read annotation " + path);
    const std::regex point(R"(^\s*Corner_\d+:\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*$)");
    const std::regex dimensions(R"(^#\s*Image Size:\s*(\d+)x(\d+)\s*$)");
    const std::regex image(R"(^#\s*Image:\s*(.*?)\s*$)");
    std::vector<cv::Point2f> points;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::smatch m;
        if (std::regex_match(line, m, dimensions)) {
            if (std::stoi(m[1]) != size.width || std::stoi(m[2]) != size.height)
                throw std::invalid_argument("target_model: annotation image size mismatch: " + path);
        } else if (std::regex_match(line, m, image)) {
            if (std::filesystem::path(m[1].str()).filename() != std::filesystem::path(image_path).filename())
                throw std::invalid_argument("target_model: annotation image name mismatch: " + path);
        } else if (std::regex_match(line, m, point)) {
            const cv::Point2f p(std::stof(m[1]), std::stof(m[2]));
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < -.5 || p.y < -.5 ||
                p.x > size.width-.5 || p.y > size.height-.5)
                throw std::invalid_argument("target_model: annotation outside board: " + path);
            points.push_back(p);
        } else if (line.find_first_not_of(" \t") != std::string::npos && line[0] != '#') {
            throw std::invalid_argument("target_model: malformed annotation row: " + path);
        }
    }
    if (points.size() != 10 || std::abs(cv::contourArea(points)) < 1e-3)
        throw std::invalid_argument("target_model requires ten nondegenerate ordered corners: " + path);
    auto cross = [](cv::Point2f a, cv::Point2f b) { return double(a.x)*b.y-double(a.y)*b.x; };
    for (size_t i=0; i<points.size(); ++i) {
        const auto a=points[i], b=points[(i+1)%points.size()];
        if (cv::norm(a-b)<1e-4) throw std::invalid_argument("target_model: duplicate adjacent corners");
        for (size_t j=i+1; j<points.size(); ++j) {
            if (j==(i+1)%points.size() || (j+1)%points.size()==i) continue;
            const auto c=points[j], d=points[(j+1)%points.size()];
            if (cross(b-a,c-a)*cross(b-a,d-a)<0 && cross(d-c,a-c)*cross(d-c,b-c)<0)
                throw std::invalid_argument("target_model: crossing annotation edges");
        }
    }
    return points;
}

inline std::vector<TemplateData> makeTargetAngleTemplates(const std::vector<cv::Point2f>& points) {
    std::vector<TemplateData> templates;
    // Positive angles use OpenCV's CCW image rotation, matching legacy -angle
    // observation uprighting. Row IDs survive every rotation and class switch.
    for (int angle=0; angle<360; angle+=15) {
        const double r=angle*CV_PI/180., c=std::cos(r), s=std::sin(r);
        std::vector<cv::Point2f> rotated;
        for (auto p:points) rotated.emplace_back(float(c*p.x+s*p.y),float(-s*p.x+c*p.y));
        float x0=rotated[0].x,x1=x0,y0=rotated[0].y,y1=y0;
        for (auto p:rotated) {x0=std::min(x0,p.x);x1=std::max(x1,p.x);y0=std::min(y0,p.y);y1=std::max(y1,p.y);}
        const double span=std::max(x1-x0,y1-y0);
        if (!(span>1e-6)) throw std::invalid_argument("target_model: zero outline span");
        TemplateData t;t.angle=angle;t.template_width=t.template_height=50;
        std::vector<cv::Point> polygon;
        for (auto p:rotated) {
            cv::Point2f q(float(24.5+(p.x-(x0+x1)*.5)*49./span),
                          float(24.5+(p.y-(y0+y1)*.5)*49./span));
            t.corners.push_back(q);polygon.emplace_back(cvRound(q.x*256),cvRound(q.y*256));
        }
        t.image=cv::Mat::zeros(50,50,CV_8U);
        cv::fillPoly(t.image,std::vector<std::vector<cv::Point>>{polygon},cv::Scalar(255),cv::LINE_8,8);
        t.image_bool=t.image;templates.push_back(std::move(t));
    }
    return templates;
}

inline std::shared_ptr<const TargetModel> buildTargetModel(const cv::Mat& gray,
        const std::string& image_path, const std::array<std::string,2>& annotations,
        double board_width_mm, double board_height_mm) {
    if (gray.empty() || gray.type()!=CV_8UC1 || gray.cols<2 || gray.rows<2 ||
        !std::isfinite(board_width_mm) || !std::isfinite(board_height_mm) ||
        board_width_mm<=0 || board_height_mm<=0)
        throw std::invalid_argument("target_model: invalid full-board image or physical size");
    auto model=std::make_shared<TargetModel>();
    model->image_size=gray.size();model->image_path=image_path;model->annotation_paths=annotations;
    model->board_size_mm={board_width_mm,board_height_mm};
    // Full image edges correspond to physical edges; annotation coordinates are
    // pixel centers. The half-pixel cancels on subtracting the common origin.
    model->mm_per_pixel={board_width_mm/gray.cols,board_height_mm/gray.rows};
    std::array<cv::Point2f,2> lo,hi;
    for (int cls=0; cls<2; ++cls) {
        auto& m=model->classes[cls];m.image_points=readTargetAnnotation(annotations[cls],image_path,gray.size());
        lo[cls]=hi[cls]=m.image_points.front();
        for (auto p:m.image_points) {
            lo[cls].x=std::min(lo[cls].x,p.x);lo[cls].y=std::min(lo[cls].y,p.y);
            hi[cls].x=std::max(hi[cls].x,p.x);hi[cls].y=std::max(hi[cls].y,p.y);
        }
        m.center_px={(double(lo[cls].x)+hi[cls].x)*.5,(double(lo[cls].y)+hi[cls].y)*.5};
        m.white_size_mm={(hi[cls].x-lo[cls].x)*model->mm_per_pixel.x(),
                         (hi[cls].y-lo[cls].y)*model->mm_per_pixel.y()};
        m.templates=makeTargetAngleTemplates(m.image_points);
    }
    model->origin_px=model->classes[0].center_px;
    auto xyz=[&](cv::Point2f p) {return Eigen::Vector3d((p.x-model->origin_px.x)*model->mm_per_pixel.x(),
        (p.y-model->origin_px.y)*model->mm_per_pixel.y(),0.);};
    for (int cls=0; cls<2; ++cls) {
        auto& m=model->classes[cls];
        for (auto p:m.image_points)m.bc_points.push_back(xyz(p));
        for (auto p:{lo[cls],cv::Point2f(hi[cls].x,lo[cls].y),hi[cls],cv::Point2f(lo[cls].x,hi[cls].y)})
            m.tt_points.push_back(xyz(p));
    }
    for (auto p:model->classes[1].image_points)
        if (cv::pointPolygonTest(model->classes[0].image_points,p,false)<0)
            throw std::invalid_argument("target_model: class1 white outline must be inside class0 white outline");
    return model;
}

inline void writeTargetModel(const std::string& path,const TargetModel& model) {
    cv::FileStorage out(path,cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    if(!out.isOpened())throw std::runtime_error("Cannot write target model: "+path);
    out<<"version"<<model.version<<"frame"<<"class0_white_bbox"<<"unit"<<"mm"
       <<"image"<<model.image_path<<"image_size_px"<<"["<<model.image_size.width<<model.image_size.height<<"]"
       <<"board_size_mm"<<"["<<model.board_size_mm.x()<<model.board_size_mm.y()<<"]"
       <<"origin_px"<<"["<<model.origin_px.x<<model.origin_px.y<<"]"<<"classes"<<"[";
    for(int cls=0;cls<2;++cls) {
        const auto& m=model.classes[cls];
        out<<"{"<<"class_id"<<cls<<"annotation"<<model.annotation_paths[cls]
           <<"white_size_mm"<<"["<<m.white_size_mm.x()<<m.white_size_mm.y()<<"]"
           <<"image_points_px"<<"[";
        for(auto p:m.image_points)out<<"["<<p.x<<p.y<<"]";
        out<<"]"<<"bc_points_mm"<<"[";
        for(const auto& p:m.bc_points)out<<"["<<p.x()<<p.y()<<p.z()<<"]";
        out<<"]"<<"tt_points_mm"<<"[";
        for(const auto& p:m.tt_points)out<<"["<<p.x()<<p.y()<<p.z()<<"]";
        out<<"]"<<"}";
    }
    out<<"]";
}

// For standalone extraction tools; tracker constructors reuse their already
// loaded image instead and share the resulting object across all extractors.
inline std::shared_ptr<const TargetModel> loadConfiguredTargetModel(const cv::FileNode& root) {
    if(!readUnifiedTargetModelEnabled(root))return {};
    auto b=root["strategies"]["board_template"],d=root["strategies"]["dual_roi"];
    const std::string image=std::string(b["template_path"]);
    return buildTargetModel(cv::imread(image,cv::IMREAD_GRAYSCALE),image,
        {std::string(d["class0_points_path"]),std::string(d["class1_points_path"])},
        double(b["template_real_width_mm"]),double(b["template_real_height_mm"]));
}

} // namespace gpnp
