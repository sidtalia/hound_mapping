#include "hound_mapping/mapper_core.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace hound_mapping {

namespace {

std::string normalizeWeightingName(std::string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (c == '-' || c == ' ') {
      c = '_';
    }
  }
  return s;
}

}  // namespace

nvblox::WeightingFunctionType parseWeightingMode(
    const std::string& raw, bool* ok)
{
  using W = nvblox::WeightingFunctionType;
  const std::string s = normalizeWeightingName(raw);
  if (ok) {
    *ok = true;
  }
  if (s == "constant" || s == "0" || s == "kconstantweight") {
    return W::kConstantWeight;
  }
  if (s == "constant_dropoff" || s == "1" || s == "kconstantdropoffweight") {
    return W::kConstantDropoffWeight;
  }
  if (s == "inverse_square" || s == "2" || s == "kinversesquareweight") {
    return W::kInverseSquareWeight;
  }
  if (s == "inverse_square_dropoff" || s == "3" ||
      s == "kinversesquaredropoffweight") {
    return W::kInverseSquareDropoffWeight;
  }
  if (s == "inverse_square_tsdf_distance_penalty" || s == "4" ||
      s == "kinversesquaretsdfdistancepenalty") {
    return W::kInverseSquareTsdfDistancePenalty;
  }
  if (s == "linear_with_max" || s == "5" || s == "klinearwithmax") {
    return W::kLinearWithMax;
  }
  if (ok) {
    *ok = false;
  }
  return W::kInverseSquareWeight;
}

void MapperCore::applyIntegratorParams()
{
  const int ray_sub = std::max(1, config_.raycast_subsampling_factor);
  const int color_sub =
      std::max(1, config_.color_sphere_tracing_ray_subsampling_factor);

  auto apply_tsdf = [&](nvblox::ProjectiveTsdfIntegrator& integ, float max_dist) {
    integ.max_integration_distance_m(max_dist);
    integ.truncation_distance_vox(config_.truncation_distance_vox);
    integ.max_weight(config_.max_weight);
    integ.weighting_function_type(config_.weighting_function_type);
    integ.invalid_depth_decay_factor(config_.invalid_depth_decay_factor);
    integ.view_calculator().raycast_subsampling_factor(
        static_cast<unsigned int>(ray_sub));
  };
  apply_tsdf(mapper_->tsdf_integrator(), config_.max_integration_distance_m);
  apply_tsdf(
      mapper_->lidar_tsdf_integrator(), config_.lidar_max_integration_distance_m);

  auto& color = mapper_->color_integrator();
  color.max_integration_distance_m(config_.color_max_integration_distance_m);
  color.truncation_distance_vox(config_.truncation_distance_vox);
  color.max_weight(config_.max_weight);
  color.weighting_function_type(config_.weighting_function_type);
  color.measurement_weight(config_.color_measurement_weight);
  color.sphere_tracing_ray_subsampling_factor(color_sub);
  color.sphere_tracer().maximum_ray_length_m(
      config_.color_sphere_tracer_max_ray_length_m);
  color.view_calculator().raycast_subsampling_factor(
      static_cast<unsigned int>(ray_sub));
}

MapperCore::MapperCore(const MapperConfig& config)
: config_(config),
  cuda_stream_(std::make_shared<nvblox::CudaStreamOwning>())
{
  nvblox::BlockMemoryPoolParams pool_params(nvblox::MemoryType::kDevice);
  mapper_ = std::make_unique<nvblox::Mapper>(
      config_.voxel_size_m, pool_params, nvblox::ProjectiveLayerType::kTsdf,
      cuda_stream_);
  applyIntegratorParams();
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

void MapperCore::clear()
{
  nvblox::BlockMemoryPoolParams pool_params(nvblox::MemoryType::kDevice);
  mapper_ = std::make_unique<nvblox::Mapper>(
      config_.voxel_size_m, pool_params, nvblox::ProjectiveLayerType::kTsdf,
      cuda_stream_);
  applyIntegratorParams();
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
