#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <nvblox/core/cuda_stream.h>

namespace hound_mapping {

/// User-authored no-go mask on GPU. Mask stays in yaml `frame_id` (usually map).
/// `tx,ty,yaw` is live T_global_overlay: p_global = R(yaw)*p_overlay + (tx,ty).
/// Updated from /tf_static; sampling kernel applies it each LocalMap extract.
struct LethalOverlayGpu {
  float* mask_d{nullptr};
  float origin_x{0.f};
  float origin_y{0.f};
  float resolution{0.05f};
  int width{0};
  int height{0};
  /// T_global_overlay (SE2), refreshed from static TF.
  float tx{0.f};
  float ty{0.f};
  float yaw{0.f};
  bool enabled{false};

  LethalOverlayGpu() = default;
  LethalOverlayGpu(const LethalOverlayGpu&) = delete;
  LethalOverlayGpu& operator=(const LethalOverlayGpu&) = delete;
  ~LethalOverlayGpu();

  void clear();

  /// Upload a host mask (CV_32FC1, 1=free / 0=lethal).
  bool upload(
      const cv::Mat& mask, float origin_x_in, float origin_y_in,
      float resolution_in, float tx_in, float ty_in, float yaw_in,
      const nvblox::CudaStream& cuda_stream);

  /// Load <stem>.lethal.yaml + sibling .f32 (see author_lethal_map.py).
  /// Load yaml + sibling .f32 in yaml frame_id. Pose is overwritten from TF.
  bool loadYaml(
      const std::string& yaml_path, float tx_in, float ty_in, float yaw_in,
      const nvblox::CudaStream& cuda_stream, std::string* err = nullptr);

  /// Rasterize yaml-frame polygons into the yaml grid (no TF bake).
  bool loadYamlPolygonsInGrid(
      const std::string& yaml_path,
      const std::vector<std::vector<cv::Point2f>>& polys_grid_xy,
      const nvblox::CudaStream& cuda_stream, std::string* err = nullptr);
};

/// frame_id from lethal yaml (default map).
std::string lethalYamlFrameId(const std::string& yaml_path);

/// Polygon vertices as stored (map-frame from RViz author). Empty if none.
std::vector<std::vector<cv::Point2f>> parseLethalPolygons(
    const std::string& yaml_path);

}  // namespace hound_mapping
