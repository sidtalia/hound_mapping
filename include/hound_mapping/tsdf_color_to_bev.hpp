#pragma once

#include <cstdint>

#include <nvblox/core/cuda_stream.h>
#include <nvblox/map/common_names.h>

#include "hound_mapping/prior_bev_gpu.hpp"

namespace hound_mapping {

struct TsdfColorToBevParams {
  float origin_x = 0.f;
  float origin_y = 0.f;
  float resolution = 0.05f;
  int width = 0;
  int height = 0;
  float z_min = -1.f;
  float z_max = 1.5f;
  /// Elevation written for cells with no traversable surface.
  float unknown_elev_z = 0.f;
  /// Min free height between an upper surface and a floor to count as
  /// traversable under an overhang (need not equal true robot height).
  float robot_height_m = 0.4f;
  /// Min TSDF weight to treat a voxel as observed.
  float tsdf_weight_min = 1e-3f;
  /// Luminance written when ColorVoxel missing / low weight (0 → lethal cost).
  float unobserved_luminance = 0.f;
  float color_weight_min = 1e-3f;
  /// Unused in tsdf_color_to_bev; passed through for callers that also slope.
  float lethal_slope_deg = 60.f;
};

/// Persistent device scratch for BEV extraction (realloc only when size grows).
class TsdfColorToBevWorkspace {
 public:
  TsdfColorToBevWorkspace() = default;
  TsdfColorToBevWorkspace(const TsdfColorToBevWorkspace&) = delete;
  TsdfColorToBevWorkspace& operator=(const TsdfColorToBevWorkspace&) = delete;
  ~TsdfColorToBevWorkspace();

  void ensureCells(int cells, int nz);
  void ensureBlocks(int n_blocks);

  float* elev_d = nullptr;
  float* lum_d = nullptr;
  float* cost_d = nullptr;
  /// Packed HxW×3 unit normals (nx,ny,nz).
  float* normal_d = nullptr;
  uint8_t* obs_d = nullptr;
  /// Per-cell Z occupancy / free bitsets (n_words = ceil(nz/32) each).
  uint32_t* occ_bits_d = nullptr;
  uint32_t* free_bits_d = nullptr;
  nvblox::Index3D* block_indices_d = nullptr;

  int cell_capacity = 0;
  int bit_words = 0;
  int block_capacity = 0;
};

/// Block-scatter TSDF → BEV: free/occ Z-bits → overhang floor → semantic cost.
/// Optional prior: when TSDF has no surface, sample PriorBevGpu (loop1 fill).
void tsdf_color_to_bev(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    const TsdfColorToBevParams& params, TsdfColorToBevWorkspace* workspace,
    float* elev_host, float* lum_host, float* cost_host,
    uint8_t* observed_host, const nvblox::CudaStream& cuda_stream,
    const PriorBevGpu* prior = nullptr);

/// GPU normals from elev + slope OR into cost. Unobserved cells get cost =
/// slope_lethal only (semantic lethality cleared after elev inpaint).
void apply_slope_normals_cost(
    TsdfColorToBevWorkspace* workspace, int width, int height, float resolution,
    float lethal_slope_deg, float* elev_host, float* cost_host,
    const uint8_t* observed_host, float* normal_host,
    const nvblox::CudaStream& cuda_stream);

}  // namespace hound_mapping
