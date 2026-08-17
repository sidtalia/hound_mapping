#include "hound_mapping/elevation_cost.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

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

TsdfAabb aabbFromTsdfLayer(const nvblox::TsdfLayer& tsdf, float pad_m)
{
  TsdfAabb out;
  const auto indices = tsdf.getAllBlockIndices();
  if (indices.empty()) {
    return out;
  }
  const float bs = tsdf.block_size();
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float min_z = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  float max_z = -std::numeric_limits<float>::infinity();
  for (const nvblox::Index3D& bi : indices) {
    const float x0 = static_cast<float>(bi.x()) * bs;
    const float y0 = static_cast<float>(bi.y()) * bs;
    const float z0 = static_cast<float>(bi.z()) * bs;
    min_x = std::min(min_x, x0);
    min_y = std::min(min_y, y0);
    min_z = std::min(min_z, z0);
    max_x = std::max(max_x, x0 + bs);
    max_y = std::max(max_y, y0 + bs);
    max_z = std::max(max_z, z0 + bs);
  }
  const float pad = std::max(0.f, pad_m);
  out.min_x = min_x - pad;
  out.min_y = min_y - pad;
  out.min_z = min_z - pad;
  out.max_x = max_x + pad;
  out.max_y = max_y + pad;
  out.max_z = max_z + pad;
  out.valid = true;
  return out;
}

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
  luminance_.create(height, width, CV_32FC1);
  host_w_ = width;
  host_h_ = height;
}

void ElevationCostExtractor::lowResTeleaInpaintField(
    cv::Mat* field, const cv::Mat& hole_mask)
{
  if (!field || field->empty() || hole_mask.empty() ||
      field->size() != hole_mask.size() || field->type() != CV_32FC1) {
    return;
  }
  if (cv::countNonZero(hole_mask) == 0) {
    return;
  }
  const int w = field->cols;
  const int h = field->rows;
  const float src_res = config_.resolution_m > 0.f
      ? config_.resolution_m
      : 0.05f;
  const float lr_res = std::max(config_.inpaint_resolution_m, src_res);
  const float scale = src_res / lr_res;
  const int dw = std::max(1, static_cast<int>(std::lround(w * scale)));
  const int dh = std::max(1, static_cast<int>(std::lround(h * scale)));

  cv::Mat field_lr;
  cv::Mat mask_lr;
  cv::resize(*field, field_lr, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
  cv::resize(hole_mask, mask_lr, cv::Size(dw, dh), 0, 0, cv::INTER_NEAREST);
  if (cv::countNonZero(mask_lr) == 0) {
    return;
  }

  cv::Mat known_lr;
  cv::bitwise_not(mask_lr, known_lr);
  double vmin = 0.0;
  double vmax = 1.0;
  cv::minMaxLoc(field_lr, &vmin, &vmax, nullptr, nullptr, known_lr);
  if (!(vmax > vmin)) {
    vmin = 0.0;
    vmax = 1.0;
  }
  const float v_scale = 255.f / static_cast<float>(vmax - vmin);
  cv::Mat field8;
  field_lr.convertTo(
      field8, CV_8UC1, v_scale, -static_cast<float>(vmin) * v_scale);
  field8.setTo(0, mask_lr);

  cv::Mat filled8;
  cv::inpaint(
      field8, mask_lr, filled8, std::max(1, config_.inpaint_radius),
      cv::INPAINT_TELEA);
  cv::Mat filled_lr;
  filled8.convertTo(filled_lr, CV_32FC1, 1.0 / v_scale, vmin);

  cv::Mat filled_hr;
  cv::resize(filled_lr, filled_hr, field->size(), 0, 0, cv::INTER_LINEAR);
  filled_hr.copyTo(*field, hole_mask);
}

void ElevationCostExtractor::lowResTeleaInpaintElev(
    cv::Mat* elevation, const cv::Mat& observed)
{
  if (!elevation || elevation->empty() || observed.empty()) {
    return;
  }
  cv::Mat hole;
  cv::compare(observed, 0, hole, cv::CMP_EQ);
  lowResTeleaInpaintField(elevation, hole);
}

LocalMaps ElevationCostExtractor::extractBevGrid(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    float origin_x, float origin_y, int width, int height, float z_min,
    float z_max, float unknown_elev_z, ExtractTiming* timing, bool use_prior)
{
  const auto t_all = Clock::now();
  if (width <= 0 || height <= 0 || !(z_max > z_min)) {
    return {};
  }
  ensureHost(width, height);

  const float res = config_.resolution_m > 0.f
      ? config_.resolution_m
      : tsdf.voxel_size();

  TsdfColorToBevParams p;
  p.origin_x = origin_x;
  p.origin_y = origin_y;
  p.resolution = res;
  p.width = width;
  p.height = height;
  p.z_min = z_min;
  p.z_max = z_max;
  p.unknown_elev_z = unknown_elev_z;
  p.robot_height_m = config_.robot_height_m;
  p.tsdf_weight_min = config_.tsdf_weight_min;
  p.bev_occ_tsdf_max_m = config_.bev_occ_tsdf_max_m;
  p.unobserved_luminance = config_.unobserved_luminance;
  p.color_weight_min = config_.color_weight_min;
  p.color_z_search_half_band_m = config_.color_z_search_half_band_m;
  p.lethal_slope_deg = config_.lethal_slope_deg;

  const PriorBevGpu* prior_arg =
      (use_prior && prior_ && prior_->enabled) ? prior_ : nullptr;

  const auto t0 = Clock::now();
  tsdf_color_to_bev(
      tsdf, color, p, &bev_ws_, reinterpret_cast<float*>(elevation_.data),
      reinterpret_cast<float*>(luminance_.data),
      reinterpret_cast<float*>(cost_.data), observed_.data, *cuda_stream_,
      prior_arg);
  const double bev_ms = msSince(t0);

  LocalMaps out;
  out.width = width;
  out.height = height;
  out.resolution = res;
  out.origin_x = origin_x;
  out.origin_y = origin_y;
  out.elevation = elevation_.clone();
  out.observed = observed_.clone();
  out.cost = cost_.clone();
  out.luminance = luminance_.clone();

  if (timing) {
    timing->bev_ms = bev_ms;
    timing->inpaint_ms = 0.0;
    timing->slope_ms = 0.0;
    timing->colorize_ms = 0.0;
    timing->total_ms = msSince(t_all);
  }
  return out;
}

LocalMaps ElevationCostExtractor::extractBev(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    const nvblox::Vector3f& center_L, float clearing_z_L,
    ExtractTiming* timing)
{
  const float res = config_.resolution_m > 0.f
      ? config_.resolution_m
      : tsdf.voxel_size();
  const int half =
      static_cast<int>(std::ceil(config_.window_radius_m / res));
  const int width = 2 * half;
  const int height = 2 * half;
  const float origin_x = center_L.x() - half * res;
  const float origin_y = center_L.y() - half * res;
  const float z_min = clearing_z_L + config_.elev_z_min_rel;
  const float z_max = clearing_z_L + config_.elev_z_max_rel;
  return extractBevGrid(
      tsdf, color, origin_x, origin_y, width, height, z_min, z_max,
      clearing_z_L, timing, /*use_prior=*/true);
}

void ElevationCostExtractor::finishLocalMap(
    LocalMaps* maps, ExtractTiming* timing)
{
  if (!maps || maps->width <= 0 || maps->height <= 0 ||
      maps->elevation.empty()) {
    return;
  }
  const auto t_all = Clock::now();

  double inpaint_ms = 0.0;
  if (config_.do_inpaint) {
    const auto t0 = Clock::now();
    cv::Mat hole;
    cv::compare(maps->observed, 0, hole, cv::CMP_EQ);
    lowResTeleaInpaintField(&maps->elevation, hole);
    // Same holes as elev: pull cost/lum from neighbors (no leftover grey).
    if (!maps->cost.empty() && maps->cost.size() == maps->elevation.size()) {
      lowResTeleaInpaintField(&maps->cost, hole);
    }
    if (!maps->luminance.empty() &&
        maps->luminance.size() == maps->elevation.size()) {
      lowResTeleaInpaintField(&maps->luminance, hole);
    }
    inpaint_ms = msSince(t0);
  }

  if (maps->normals.empty() || maps->normals.size() != maps->elevation.size()) {
    maps->normals.create(maps->height, maps->width, CV_32FC3);
  }
  if (!maps->cost.isContinuous()) {
    maps->cost = maps->cost.clone();
  }
  if (!maps->elevation.isContinuous()) {
    maps->elevation = maps->elevation.clone();
  }

  auto t0 = Clock::now();
  apply_slope_normals_cost(
      &bev_ws_, maps->width, maps->height, maps->resolution,
      config_.lethal_slope_deg, reinterpret_cast<float*>(maps->elevation.data),
      reinterpret_cast<float*>(maps->cost.data), maps->observed.data,
      reinterpret_cast<float*>(maps->normals.data), *cuda_stream_);
  const double slope_ms = msSince(t0);

  t0 = Clock::now();
  colorizeFromLuminance(maps->luminance, &maps->elev_color);
  const double colorize_ms = msSince(t0);

  if (timing) {
    timing->inpaint_ms = inpaint_ms;
    timing->slope_ms = slope_ms;
    timing->colorize_ms = colorize_ms;
    timing->total_ms = msSince(t_all);
  }
}

LocalMaps ElevationCostExtractor::extract(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    const nvblox::Vector3f& center_L, float clearing_z_L,
    ExtractTiming* timing)
{
  const auto t_all = Clock::now();
  ExtractTiming bev_t;
  LocalMaps maps = extractBev(tsdf, color, center_L, clearing_z_L, &bev_t);
  ExtractTiming fin_t;
  finishLocalMap(&maps, &fin_t);
  if (timing) {
    timing->bev_ms = bev_t.bev_ms;
    timing->inpaint_ms = fin_t.inpaint_ms;
    timing->slope_ms = fin_t.slope_ms;
    timing->colorize_ms = fin_t.colorize_ms;
    timing->total_ms = msSince(t_all);
  }
  return maps;
}

LocalMaps ElevationCostExtractor::extract(
    const MapperCore& core, const nvblox::Vector3f& center_L,
    float clearing_z_L, ExtractTiming* timing)
{
  return extract(
      core.mapper().tsdf_layer(), core.mapper().color_layer(), center_L,
      clearing_z_L, timing);
}

void ElevationCostExtractor::colorizeFromLuminance(
    const cv::Mat& luminance, cv::Mat* elev_color)
{
  elev_color->create(luminance.rows, luminance.cols, CV_8UC3);
  for (int row = 0; row < luminance.rows; ++row) {
    for (int col = 0; col < luminance.cols; ++col) {
      const float L =
          std::clamp(luminance.at<float>(row, col), 0.f, 255.f);
      const uint8_t v = static_cast<uint8_t>(L);
      elev_color->at<cv::Vec3b>(row, col) =
          cv::Vec3b(0, v, static_cast<uint8_t>(255 - v));
    }
  }
}

}  // namespace hound_mapping
