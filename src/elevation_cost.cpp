#include "hound_mapping/elevation_cost.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

namespace hound_mapping {
namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

ElevationCostExtractor::ElevationCostExtractor(const ElevationCostConfig& config)
: config_(config),
  cuda_stream_(std::make_shared<nvblox::CudaStreamOwning>())
{
}

void ElevationCostExtractor::ensureHost(int width, int height)
{
  if (width == host_w_ && height == host_h_ && !elevation_.empty()) {
    return;
  }
  elevation_.create(height, width, CV_32FC1);
  observed_.create(height, width, CV_8UC1);
  cost_.create(height, width, CV_32FC1);
  normals_.create(height, width, CV_32FC3);
  luminance_.create(height, width, CV_32FC1);
  elev_color_.create(height, width, CV_8UC3);
  host_w_ = width;
  host_h_ = height;
}

void ElevationCostExtractor::lowResTeleaInpaintElev()
{
  if (cv::countNonZero(observed_) == host_w_ * host_h_) {
    return;
  }
  const float src_res = config_.resolution_m > 0.f
      ? config_.resolution_m
      : 0.05f;
  const float lr_res = std::max(config_.inpaint_resolution_m, src_res);
  const float scale = src_res / lr_res;
  const int dw = std::max(1, static_cast<int>(std::lround(host_w_ * scale)));
  const int dh = std::max(1, static_cast<int>(std::lround(host_h_ * scale)));

  cv::Mat elev_lr;
  cv::Mat obs_lr;
  cv::resize(elevation_, elev_lr, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
  cv::resize(observed_, obs_lr, cv::Size(dw, dh), 0, 0, cv::INTER_NEAREST);

  cv::Mat mask_lr;
  cv::compare(obs_lr, 0, mask_lr, cv::CMP_EQ);
  if (cv::countNonZero(mask_lr) == 0) {
    return;
  }

  // Telea is 8U: map known Z into [0,255], unknown → 0.
  double zmin = 0.0;
  double zmax = 1.0;
  cv::minMaxLoc(elev_lr, &zmin, &zmax, nullptr, nullptr, obs_lr);
  if (!(zmax > zmin)) {
    zmin = 0.0;
    zmax = 1.0;
  }
  const float z_scale = 255.f / static_cast<float>(zmax - zmin);
  cv::Mat elev8;
  elev_lr.convertTo(
      elev8, CV_8UC1, z_scale, -static_cast<float>(zmin) * z_scale);
  elev8.setTo(0, mask_lr);

  cv::Mat filled8;
  cv::inpaint(
      elev8, mask_lr, filled8, std::max(1, config_.inpaint_radius),
      cv::INPAINT_TELEA);
  cv::Mat filled_lr;
  filled8.convertTo(
      filled_lr, CV_32FC1, 1.0 / z_scale, zmin);

  cv::Mat filled_hr;
  cv::resize(
      filled_lr, filled_hr, elevation_.size(), 0, 0, cv::INTER_LINEAR);
  // Only overwrite unknown cells at native resolution.
  filled_hr.copyTo(elevation_, observed_ == 0);
}

LocalMaps ElevationCostExtractor::extract(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    const nvblox::Vector3f& center_L, float clearing_z_L,
    ExtractTiming* timing)
{
  const auto t_all = Clock::now();
  const float res = config_.resolution_m > 0.f
      ? config_.resolution_m
      : tsdf.voxel_size();
  const int half =
      static_cast<int>(std::ceil(config_.window_radius_m / res));
  const int width = 2 * half;
  const int height = 2 * half;
  ensureHost(width, height);

  LocalMaps out;
  out.width = width;
  out.height = height;
  out.resolution = res;
  out.origin_x = center_L.x() - half * res;
  out.origin_y = center_L.y() - half * res;
  out.elevation = elevation_;
  out.observed = observed_;
  out.cost = cost_;
  out.normals = normals_;
  out.luminance = luminance_;
  out.elev_color = elev_color_;

  TsdfColorToBevParams p;
  p.origin_x = out.origin_x;
  p.origin_y = out.origin_y;
  p.resolution = out.resolution;
  p.width = out.width;
  p.height = out.height;
  p.z_min = clearing_z_L + config_.elev_z_min_rel;
  p.z_max = clearing_z_L + config_.elev_z_max_rel;
  p.unknown_elev_z = clearing_z_L;
  p.robot_height_m = config_.robot_height_m;
  p.tsdf_weight_min = config_.tsdf_weight_min;
  p.unobserved_luminance = config_.unobserved_luminance;
  p.color_weight_min = config_.color_weight_min;
  p.lethal_slope_deg = config_.lethal_slope_deg;

  auto t0 = Clock::now();
  tsdf_color_to_bev(
      tsdf, color, p, &bev_ws_, reinterpret_cast<float*>(elevation_.data),
      reinterpret_cast<float*>(luminance_.data),
      reinterpret_cast<float*>(cost_.data), observed_.data, *cuda_stream_);
  const double bev_ms = msSince(t0);

  double inpaint_ms = 0.0;
  if (config_.do_inpaint) {
    t0 = Clock::now();
    lowResTeleaInpaintElev();
    inpaint_ms = msSince(t0);
  }

  // Normals + slope after inpaint so finite differences see filled elev.
  t0 = Clock::now();
  apply_slope_normals_cost(
      &bev_ws_, width, height, res, config_.lethal_slope_deg,
      reinterpret_cast<float*>(elevation_.data),
      reinterpret_cast<float*>(cost_.data), observed_.data,
      reinterpret_cast<float*>(normals_.data), *cuda_stream_);
  const double slope_ms = msSince(t0);

  t0 = Clock::now();
  colorizeFromLuminance();
  const double colorize_ms = msSince(t0);

  if (timing) {
    timing->bev_ms = bev_ms;
    timing->inpaint_ms = inpaint_ms;
    timing->slope_ms = slope_ms;
    timing->colorize_ms = colorize_ms;
    timing->total_ms = msSince(t_all);
  }
  return out;
}

LocalMaps ElevationCostExtractor::extract(
    const MapperCore& core, const nvblox::Vector3f& center_L,
    float clearing_z_L, ExtractTiming* timing)
{
  return extract(
      core.mapper().tsdf_layer(), core.mapper().color_layer(), center_L,
      clearing_z_L, timing);
}

void ElevationCostExtractor::colorizeFromLuminance()
{
  for (int row = 0; row < host_h_; ++row) {
    for (int col = 0; col < host_w_; ++col) {
      const float L =
          std::clamp(luminance_.at<float>(row, col), 0.f, 255.f);
      const uint8_t v = static_cast<uint8_t>(L);
      elev_color_.at<cv::Vec3b>(row, col) =
          cv::Vec3b(0, v, static_cast<uint8_t>(255 - v));
    }
  }
}

}  // namespace hound_mapping
