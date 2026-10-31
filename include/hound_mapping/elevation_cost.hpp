#pragma once

#include <memory>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/nvblox.h>

#include "hound_mapping/lethal_overlay_gpu.hpp"
#include "hound_mapping/mapper_core.hpp"
#include "hound_mapping/prior_bev_gpu.hpp"
#include "hound_mapping/tsdf_color_to_bev.hpp"

namespace hound_mapping {

struct ElevationCostConfig {
  float resolution_m = 0.05f;
  float window_radius_m = 32.0f;
  float tsdf_weight_min = 1e-3f;
  /// See TsdfColorToBevParams::bev_occ_tsdf_max_m.
  float bev_occ_tsdf_max_m = 0.f;
  float robot_height_m = 0.4f;
  float unobserved_luminance = 128.f;
  float color_weight_min = 1e-3f;
  /// See TsdfColorToBevParams::color_z_search_half_band_m (0 = auto).
  float color_z_search_half_band_m = 0.f;
  /// See TsdfColorToBevParams::color_z_column_above_m (0 = full BEV Z span).
  float color_z_column_above_m = 0.f;
  float elev_z_min_rel = -5.0f;
  float elev_z_max_rel = 2.0f;
  float lethal_slope_deg = 60.f;
  /// Low-res Telea fill of unknown elev, then upsample + mask.
  bool do_inpaint = true;
  /// Resolution used for Telea (m). Coarser ⇒ faster. Typical 0.25.
  float inpaint_resolution_m = 0.25f;
  int inpaint_radius = 3;
  /// Median on elev before slope/normals. OpenCV 32F allows 3 or 5; 0 = off.
  int elev_median_ksize = 5;
};

struct LocalMaps {
  cv::Mat elevation;   // world Z; unknowns filled if do_inpaint
  cv::Mat observed;    // 255 = TSDF-observed surface (pre-inpaint)
  cv::Mat cost;        // [0,255] free-ness (255 free / 0 lethal; semantic OR slope)
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

/// Axis-aligned bounds of allocated TSDF blocks in layer frame (meters).
struct TsdfAabb {
  float min_x{0}, min_y{0}, min_z{0};
  float max_x{0}, max_y{0}, max_z{0};
  bool valid{false};
};

TsdfAabb aabbFromTsdfLayer(const nvblox::TsdfLayer& tsdf, float pad_m = 0.f);

/// Load a LayerCake and extract a full-coverage BEV (same grid as prior fill).
bool extractLayerCakeBev(
    const std::string& cake_path, float voxel_size_m,
    const ElevationCostConfig& ec, LocalMaps* out, std::string* err = nullptr);

class ElevationCostExtractor {
 public:
  explicit ElevationCostExtractor(const ElevationCostConfig& config);

  /// Optional GPU prior: no live surface → elev+cost; live surface, no color → cost.
  void setPrior(const PriorBevGpu* prior) { prior_ = prior; }
  const PriorBevGpu* prior() const { return prior_; }

  /// Optional no-go overlay. Applied in finishLocalMap after slope (AND).
  void setLethalOverlay(const LethalOverlayGpu* overlay) { overlay_ = overlay; }
  const LethalOverlayGpu* lethalOverlay() const { return overlay_; }

  /// Loop1: TSDF/color → elev/cost/obs/lum (owned clones). No inpaint/normals.
  LocalMaps extractBev(
      const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
      const nvblox::Vector3f& center_L, float clearing_z_L,
      ExtractTiming* timing = nullptr);

  /// Explicit grid + absolute Z band (full-coverage prior extract).
  LocalMaps extractBevGrid(
      const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
      float origin_x, float origin_y, int width, int height, float z_min,
      float z_max, float unknown_elev_z, ExtractTiming* timing = nullptr,
      bool use_prior = true);

  /// Loop2: inpaint elev → median → slope/normals → elev_color.
  void finishLocalMap(LocalMaps* maps, ExtractTiming* timing = nullptr);

  /// Full pipeline (BEV + finish). Used by offline smoke tools.
  LocalMaps extract(
      const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
      const nvblox::Vector3f& center_L, float clearing_z_L,
      ExtractTiming* timing = nullptr);

  LocalMaps extract(
      const MapperCore& core, const nvblox::Vector3f& center_L,
      float clearing_z_L, ExtractTiming* timing = nullptr);

 private:
  void ensureHost(int width, int height);
  void colorizeFromLuminance(const cv::Mat& luminance, cv::Mat* elev_color);
  /// Telea-fill holes (mask 255) at low res; write back only where mask.
  void lowResTeleaInpaintField(cv::Mat* field, const cv::Mat& hole_mask);
  void lowResTeleaInpaintElev(cv::Mat* elevation, const cv::Mat& observed);

  ElevationCostConfig config_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  TsdfColorToBevWorkspace bev_ws_;
  const PriorBevGpu* prior_{nullptr};
  const LethalOverlayGpu* overlay_{nullptr};

  cv::Mat elevation_;
  cv::Mat observed_;
  cv::Mat cost_;
  cv::Mat luminance_;
  int host_w_ = 0;
  int host_h_ = 0;
};

}  // namespace hound_mapping
