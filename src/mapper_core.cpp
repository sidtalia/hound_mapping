#include "hound_mapping/mapper_core.hpp"

#include <utility>

namespace hound_mapping {

MapperCore::MapperCore(const MapperConfig& config)
: config_(config),
  cuda_stream_(std::make_shared<nvblox::CudaStreamOwning>())
{
  nvblox::BlockMemoryPoolParams pool_params(nvblox::MemoryType::kDevice);
  mapper_ = std::make_unique<nvblox::Mapper>(
      config_.voxel_size_m, pool_params, nvblox::ProjectiveLayerType::kTsdf,
      cuda_stream_);

  mapper_->tsdf_integrator().max_integration_distance_m(
      config_.max_integration_distance_m);
  mapper_->lidar_tsdf_integrator().max_integration_distance_m(
      config_.lidar_max_integration_distance_m);
}

void MapperCore::integrateDepth(
    nvblox::DepthImage& depth, const nvblox::Transform& T_L_C,
    const nvblox::Camera& camera,
    const std::optional<nvblox::MonoImage>& people_mask)
{
  if (people_mask.has_value()) {
    // Non-zero people-mask pixels are inverted → inactive for static TSDF.
    const nvblox::MonoImage& mask = people_mask.value();
    std::optional<nvblox::ImageView<const uint8_t>> mask_view{
        nvblox::MonoImageConstView(mask)};
    nvblox::MaskedDepthImageConstView masked(
        depth, mask_view, nvblox::MaskMode::kInverted);
    mapper_->integrateDepth(masked, T_L_C, camera);
  } else {
    mapper_->integrateDepth(depth, T_L_C, camera);
  }
}

void MapperCore::integrateLidar(
    const nvblox::Pointcloud& cloud, const nvblox::Transform& T_L_S,
    const nvblox::Lidar& lidar, bool motion_compensation,
    const std::optional<nvblox::Transform>& T_L_S_scan_end,
    const std::optional<nvblox::Time>& scan_duration_ms)
{
  mapper_->integrateDepth(
      cloud, T_L_S, lidar, motion_compensation, T_L_S_scan_end,
      scan_duration_ms);
}

void MapperCore::integrateColor(
    const nvblox::ColorImage& color, const nvblox::Transform& T_L_C,
    const nvblox::Camera& camera)
{
  mapper_->integrateColor(color, T_L_C, camera);
}

void MapperCore::clearOutsideRadius(const nvblox::Vector3f& center_L)
{
  if (config_.map_clearing_radius_m > 0.f) {
    mapper_->clearOutsideRadius(center_L, config_.map_clearing_radius_m);
  }
}

bool MapperCore::saveLayerCake(const std::string& path) const
{
  return mapper_->saveLayerCake(path);
}

bool MapperCore::loadLayerCake(const std::string& path)
{
  nvblox::BlockMemoryPoolParams pool_params(nvblox::MemoryType::kDevice);
  return mapper_->loadMap(path, pool_params);
}

bool MapperCore::exportColorMeshPly(const std::string& ply_path)
{
  mapper_->updateColorMesh(nvblox::UpdateFullLayer::kYes);
  return mapper_->saveColorMeshAsPly(ply_path);
}

}  // namespace hound_mapping
