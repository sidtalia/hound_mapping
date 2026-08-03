#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/nvblox.h>

#include "hound_mapping/mapper_core.hpp"
#include "hound_mapping/tsdf_color_to_bev.hpp"

namespace hound_mapping {

struct ElevationCostConfig {
  float resolution_m = 0.05f;
  float window_radius_m = 32.0f;
  float tsdf_weight_min = 1e-3f;
  float robot_height_m = 0.4f;
  float unobserved_luminance = 0.f;
  float color_weight_min = 1e-3f;
  float elev_z_min_rel = -5.0f;
  float elev_z_max_rel = 2.0f;
  float lethal_slope_deg = 60.f;
  /// Low-res Telea fill of unknown elev, then upsample + mask.
  bool do_inpaint = true;
  /// Resolution used for Telea (m). Coarser ⇒ faster. Typical 0.25.
  float inpaint_resolution_m = 0.25f;
  int inpaint_radius = 3;
};

struct LocalMaps {
  cv::Mat elevation;   // world Z; unknowns filled if do_inpaint
  cv::Mat observed;    // 255 = TSDF-observed surface (pre-inpaint)
  cv::Mat cost;        // [0,1], semantic OR slope (unknowns: slope only)
  cv::Mat normals;     // HxWx3 unit normals (after inpaint)
  cv::Mat luminance;
  cv::Mat elev_color;  // BGR viz from luminance (debug topic)
  float resolution = 0.05f;
  float origin_x = 0.f;
  float origin_y = 0.f;
  int width = 0;
  int height = 0;
};

struct ExtractTiming {
  double bev_ms = 0.0;  // TSDF→BEV scatter
  double inpaint_ms = 0.0;
  double slope_ms = 0.0;
  double colorize_ms = 0.0;
  double total_ms = 0.0;
};

class ElevationCostExtractor {
 public:
  explicit ElevationCostExtractor(const ElevationCostConfig& config);

  LocalMaps extract(
      const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
      const nvblox::Vector3f& center_L, float clearing_z_L,
      ExtractTiming* timing = nullptr);

  LocalMaps extract(
      const MapperCore& core, const nvblox::Vector3f& center_L,
      float clearing_z_L, ExtractTiming* timing = nullptr);

 private:
  void ensureHost(int width, int height);
  void colorizeFromLuminance();
  void lowResTeleaInpaintElev();

  ElevationCostConfig config_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  TsdfColorToBevWorkspace bev_ws_;

  cv::Mat elevation_;
  cv::Mat observed_;
  cv::Mat cost_;
  cv::Mat normals_;
  cv::Mat luminance_;
  cv::Mat elev_color_;
  int host_w_ = 0;
  int host_h_ = 0;
};

}  // namespace hound_mapping
