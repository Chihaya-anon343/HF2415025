#pragma once

#include "Types.hpp"
#include <algorithm>
#include <cmath>

namespace gpnp {

// Bounds are floating point: integer boundingRect rounding is material for TT.
inline cv::Rect2d identityBounds(const std::vector<cv::Point2f>& points,
                                 cv::Point2f offset = {}) {
  if (points.empty()) return {};
  double x0 = points[0].x, x1 = x0, y0 = points[0].y, y1 = y0;
  for (auto p : points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return {};
    x0 = std::min(x0, (double)p.x); x1 = std::max(x1, (double)p.x);
    y0 = std::min(y0, (double)p.y); y1 = std::max(y1, (double)p.y);
  }
  return {x0 + offset.x, y0 + offset.y, x1 - x0, y1 - y0};
}

inline cv::Rect2d identityProjection(const std::vector<Eigen::Vector3d>& model,
    const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  std::vector<cv::Point2f> projected;
  for (const auto& point : model) {
    const Eigen::Vector3d p = R * point + t;
    if (!p.allFinite() || p.z() <= 1e-6) return {};
    const Eigen::Vector3d q = K * p;
    projected.emplace_back(q.x() / q.z(), q.y() / q.z());
  }
  return identityBounds(projected);
}

struct RoiIdentityDecision {
  bool accepted{true};
  const char* reason{"identity_no_evidence"};
  double expected_ratio{0}, alternate_ratio{0};
};

// A scale change alone is insufficient: require the opposite physical model
// to explain BOTH image dimensions and the observation's location as well.
inline RoiIdentityDecision evaluateRoiIdentity(const cv::Rect2d& observed,
    const cv::Rect2d& expected, const cv::Rect2d& alternate,
    const RoiIdentityConfig& cfg) {
  RoiIdentityDecision result;
  if (!cfg.enabled) { result.reason = "identity_disabled"; return result; }
  for (const auto& b : {observed, expected, alternate})
    if (!std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.width) ||
        !std::isfinite(b.height) || b.width < cfg.min_projected_span_px ||
        b.height < cfg.min_projected_span_px) return result;
  auto ratio = [&](const cv::Rect2d& b) {
    return std::max({observed.width / b.width, b.width / observed.width,
                     observed.height / b.height, b.height / observed.height});
  };
  result.expected_ratio = ratio(expected);
  result.alternate_ratio = ratio(alternate);
  const cv::Point2d delta(observed.x + observed.width / 2 - alternate.x - alternate.width / 2,
                           observed.y + observed.height / 2 - alternate.y - alternate.height / 2);
  const double center_distance = cv::norm(delta) / std::hypot(alternate.width, alternate.height);
  result.accepted = !(result.expected_ratio > cfg.max_scale_ratio &&
      result.alternate_ratio <= cfg.alternate_scale_ratio &&
      center_distance <= cfg.max_center_distance_ratio);
  result.reason = result.accepted ? "identity_scale_consistent" : "identity_scale_inconsistent";
  return result;
}

} // namespace gpnp
