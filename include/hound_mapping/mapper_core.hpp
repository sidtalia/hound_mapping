#pragma once

#include <memory>
#include <optional>
#include <string>

#include <nvblox/nvblox.h>

namespace hound_mapping {

struct MapperConfig {
  float voxel_size_m = 0.05f;
  float max_integration_distance_m = 15.0f;
  float lidar_max_integration_distance_m = 15.0f;
  float map_clearing_radius_m = 32.0f;
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
      const nvblox::Camera& camera);

  void clearOutsideRadius(const nvblox::Vector3f& center_L);

  /// Serialize TSDF/color LayerCake to an nvblox SQLite map file.
  /// Must be called from the thread that owns the mapper (no concurrent integrate).
  bool saveLayerCake(const std::string& path) const;

  /// Replace layers from a LayerCake file (clears existing map).
  bool loadLayerCake(const std::string& path);

  /// Full color mesh from TSDF(+color) → PLY. Call after load or integrate.
  bool exportColorMeshPly(const std::string& ply_path);

  nvblox::Mapper& mapper() { return *mapper_; }
  const nvblox::Mapper& mapper() const { return *mapper_; }
  const MapperConfig& config() const { return config_; }

 private:
  MapperConfig config_;
  std::shared_ptr<nvblox::CudaStream> cuda_stream_;
  std::unique_ptr<nvblox::Mapper> mapper_;
};

}  // namespace hound_mapping
