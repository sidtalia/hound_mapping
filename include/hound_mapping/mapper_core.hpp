#pragma once

#include <memory>
#include <optional>
#include <string>

#include <nvblox/nvblox.h>

namespace hound_mapping {

/// Parse Isaac/nvblox weighting names. Unknown → inverse_square, *ok = false.
nvblox::WeightingFunctionType parseWeightingMode(
    const std::string& raw, bool* ok = nullptr);

struct MapperConfig {
  float voxel_size_m = 0.05f;
  float max_integration_distance_m = 15.0f;
  float lidar_max_integration_distance_m = 15.0f;
  float map_clearing_radius_m = 32.0f;

  // Camera + lidar TSDF (+ color truncation / max_weight / weighting).
  float truncation_distance_vox = 4.0f;
  float max_weight = 5.0f;
  nvblox::WeightingFunctionType weighting_function_type =
      nvblox::WeightingFunctionType::kInverseSquareWeight;
  float invalid_depth_decay_factor = -1.0f;
  int raycast_subsampling_factor = 4;

  // Color paint. Library defaults are 7 m max / 15 m sphere-tracer.
  float color_max_integration_distance_m = 7.0f;
  float color_sphere_tracer_max_ray_length_m = 15.0f;
  int color_sphere_tracing_ray_subsampling_factor = 4;
  float color_measurement_weight = 0.8f;
};

/// Per-sensor projective settings applied immediately before integrate.
struct IntegratorSettings {
  float max_integration_distance_m = 15.0f;
  float truncation_distance_vox = 4.0f;
  float max_weight = 5.0f;
  nvblox::WeightingFunctionType weighting_function_type =
      nvblox::WeightingFunctionType::kInverseSquareWeight;
  float invalid_depth_decay_factor = -1.0f;
  int ray_subsampling_factor = 4;
  float measurement_weight = 0.8f;
  float color_max_integration_distance_m = 7.0f;
  float sphere_tracer_max_ray_length_m = 15.0f;
  int sphere_tracing_ray_subsampling_factor = 4;
};

/// ROS-free nvblox wrapper: integrate depth/lidar/color, radius-clear blocks.
class MapperCore {
 public:
  explicit MapperCore(const MapperConfig& config);
  ~MapperCore() = default;

  MapperCore(const MapperCore&) = delete;
  MapperCore& operator=(const MapperCore&) = delete;

  void integrateDepth(
      nvblox::DepthImage& depth, const nvblox::Transform& T_L_C,
      const nvblox::Camera& camera,
      const std::optional<nvblox::MonoImage>& people_mask = std::nullopt);

  void integrateLidar(
      const nvblox::Pointcloud& cloud, const nvblox::Transform& T_L_S,
      const nvblox::Lidar& lidar, bool motion_compensation = false,
      const std::optional<nvblox::Transform>& T_L_S_scan_end = std::nullopt,
      const std::optional<nvblox::Time>& scan_duration_ms = std::nullopt);

  void integrateColor(
      const nvblox::ColorImage& color, const nvblox::Transform& T_L_C,
      const nvblox::Camera& camera,
      const std::optional<nvblox::MonoImage>& keep_mask = std::nullopt);

  void clearOutsideRadius(const nvblox::Vector3f& center_L);

  /// Drop all TSDF/color blocks (new empty mapper, same voxel/integrator config).
  void clear();

  /// Serialize TSDF/color LayerCake to an nvblox SQLite map file.
  /// Must be called from the thread that owns the mapper (no concurrent integrate).
  bool saveLayerCake(const std::string& path) const;

  /// Replace layers from a LayerCake file (clears existing map).
  bool loadLayerCake(const std::string& path);

  /// Full color mesh from TSDF(+color) → PLY. Call after load or integrate.
  bool exportColorMeshPly(const std::string& ply_path);

  /// Cheap: enum/float writes on the live integrator (no realloc).
  void applyTsdfSettings(const IntegratorSettings& s, bool lidar);
  void applyColorSettings(const IntegratorSettings& s);

  nvblox::Mapper& mapper() { return *mapper_; }
  const nvblox::Mapper& mapper() const { return *mapper_; }
  const MapperConfig& config() const { return config_; }

 private:
  void applyIntegratorParams();

  MapperConfig config_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  std::unique_ptr<nvblox::Mapper> mapper_;
};

}  // namespace hound_mapping
