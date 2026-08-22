// Block-scatter TSDF → BEV elevation / cost / luminance.
// 1) Filter allocated blocks overlapping the window
// 2) One thread per voxel: coalesced reads → set free/occ Z-bits in BEV columns
// 3) 2D pass: overhang-aware floor + color/cost

#include "hound_mapping/tsdf_color_to_bev.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <nvblox/core/indexing.h>
#include <nvblox/gpu_hash/gpu_layer_view.h>
#include <nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh>
#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>
#include <nvblox/map/blox.h>
#include <nvblox/map/voxels.h>

namespace hound_mapping {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr int kVps = nvblox::TsdfBlock::kVoxelsPerSide;  // 8
constexpr int kVoxelsPerBlock = nvblox::TsdfBlock::kNumVoxels;  // 512
constexpr int kMaxNz = 256;

__device__ inline float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

__device__ inline void set_bit(uint32_t* words, int n_words, int cell, int iz) {
  if (iz < 0) {
    return;
  }
  const int w = iz >> 5;
  if (w >= n_words) {
    return;
  }
  atomicOr(&words[cell * n_words + w], 1u << (iz & 31));
}

__device__ inline bool test_bit(
    const uint32_t* words, int n_words, int cell, int iz) {
  if (iz < 0) {
    return false;
  }
  const int w = iz >> 5;
  if (w >= n_words) {
    return false;
  }
  return (words[cell * n_words + w] >> (iz & 31)) & 1u;
}

__device__ inline float colorLuminance(const nvblox::Color& c) {
  return 0.299f * static_cast<float>(c.r()) +
         0.587f * static_cast<float>(c.g()) +
         0.114f * static_cast<float>(c.b());
}

/// Match nvblox ColorMeshIntegrator::updateAppearance: closest ColorVoxel
/// with weight (3×3×3 around the query point). Same-index TSDF voxels are
/// often unpainted; color lives on the free-side neighbor.
__device__ inline bool closestColorLuminance(
    nvblox::Index3DDeviceHashMapType<nvblox::ColorBlock> color_hash,
    float block_size, float voxel_size, const nvblox::Vector3f& p,
    float color_weight_min, float* out_L)
{
  float best_d2 = kInf;
  float best_L = 0.f;
  bool found = false;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        const nvblox::Vector3f q =
            p + nvblox::Vector3f(
                    static_cast<float>(dx) * voxel_size,
                    static_cast<float>(dy) * voxel_size,
                    static_cast<float>(dz) * voxel_size);
        nvblox::ColorVoxel* cv = nullptr;
        if (!nvblox::getVoxelAtPosition<nvblox::ColorVoxel>(
                color_hash, q, block_size, &cv) ||
            cv == nullptr || cv->weight < color_weight_min) {
          continue;
        }
        const nvblox::Vector3f d = q - p;
        const float d2 = d.x() * d.x() + d.y() * d.y() + d.z() * d.z();
        if (d2 < best_d2) {
          best_d2 = d2;
          best_L = colorLuminance(cv->color);
          found = true;
        }
      }
    }
  }
  if (found) {
    *out_L = best_L;
  }
  return found;
}

__global__ void scatter_block_voxels_kernel(
    nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> tsdf_hash,
    nvblox::Index3DDeviceHashMapType<nvblox::ColorBlock> color_hash,
    const nvblox::Index3D* blocks, int n_blocks, float block_size,
    float origin_x, float origin_y, float resolution, int width, int height,
    float z_min, float z_max, float step, int nz, float tsdf_weight_min,
    float bev_occ_tsdf_max_m, float color_weight_min, float color_band_m,
    float voxel_size, int n_words, uint32_t* occ_bits, uint32_t* free_bits,
    uint32_t* lum_iz_u) {
  const int b = static_cast<int>(blockIdx.x);
  if (b >= n_blocks) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= kVoxelsPerBlock) {
    return;
  }

  const nvblox::Index3D bi = blocks[b];
  nvblox::TsdfBlock* block = nvblox::getBlockPtr(tsdf_hash, bi);
  if (block == nullptr) {
    return;
  }

  const int vx = tid % kVps;
  const int vy = (tid / kVps) % kVps;
  const int vz = tid / (kVps * kVps);
  const nvblox::TsdfVoxel& voxel = block->voxels[vx][vy][vz];
  if (voxel.weight < tsdf_weight_min) {
    return;
  }

  const nvblox::Vector3f p =
      nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
          block_size, bi, nvblox::Index3D(vx, vy, vz));
  if (p.z() < z_min || p.z() > z_max) {
    return;
  }

  const int ix = static_cast<int>(floorf((p.x() - origin_x) / resolution));
  const int iy = static_cast<int>(floorf((p.y() - origin_y) / resolution));
  if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
    return;
  }
  // iz = 0 at z_max (top), increasing downward — matches prior sweep.
  int iz = static_cast<int>(floorf((z_max - p.z()) / step + 1e-4f));
  if (iz < 0) {
    iz = 0;
  }
  if (iz >= nz) {
    iz = nz - 1;
  }

  const int cell = iy * width + ix;
  if (voxel.distance > bev_occ_tsdf_max_m) {
    set_bit(free_bits, n_words, cell, iz);
  } else {
    set_bit(occ_bits, n_words, cell, iz);
  }

  // Color into this XY cell's Z bin — read back only if this iz becomes floor.
  if (lum_iz_u != nullptr && fabsf(voxel.distance) <= color_band_m) {
    float sampled_L = 0.f;
    if (closestColorLuminance(
            color_hash, block_size, voxel_size, p, color_weight_min,
            &sampled_L)) {
      const uint32_t L = static_cast<uint32_t>(
          fminf(fmaxf(sampled_L, 0.f), 255.f) + 0.5f);
      atomicMin(&lum_iz_u[cell * nz + iz], L);
    }
  }
}

__global__ void finalize_from_bits_kernel(
    nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> tsdf_hash,
    float block_size, float origin_x, float origin_y, float resolution,
    int width, int height, float z_max, float step, int nz, float unknown_elev_z,
    float robot_height_m, float tsdf_weight_min, float bev_occ_tsdf_max_m,
    float unobserved_luminance, int n_words, const uint32_t* occ_bits,
    const uint32_t* free_bits, const uint32_t* lum_iz_u, float* elev,
    float* lum, float* cost, uint8_t* observed, const float* prior_elev,
    const float* prior_cost, const uint8_t* prior_obs, float prior_ox,
    float prior_oy, float prior_res, int prior_w, int prior_h, float prior_tx,
    float prior_ty, float prior_tz, float prior_yaw, bool prior_enabled) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= width || iy >= height) {
    return;
  }
  const int cell = iy * width + ix;
  const float h_req = fmaxf(robot_height_m, 0.f);
  float ceiling_z = kInf;
  float surface_z = unknown_elev_z;
  bool found = false;
  int floor_iz = -1;

  for (int iz = 0; iz < nz; ++iz) {
    if (!test_bit(occ_bits, n_words, cell, iz)) {
      continue;
    }
    const float z_occ = z_max - static_cast<float>(iz) * step;
    float z_hit = z_occ;
    // Sub-voxel refine if free sample sits just above.
    if (iz > 0 && test_bit(free_bits, n_words, cell, iz - 1)) {
      const float z_free = z_max - static_cast<float>(iz - 1) * step;
      const float x = origin_x + (static_cast<float>(ix) + 0.5f) * resolution;
      const float y = origin_y + (static_cast<float>(iy) + 0.5f) * resolution;
      nvblox::TsdfVoxel* tv_o = nullptr;
      nvblox::TsdfVoxel* tv_f = nullptr;
      float d_o = 0.f;
      float d_f = 0.f;
      if (nvblox::getVoxelAtPosition<nvblox::TsdfVoxel>(
              tsdf_hash, nvblox::Vector3f(x, y, z_occ), block_size, &tv_o) &&
          tv_o && tv_o->weight >= tsdf_weight_min &&
          nvblox::getVoxelAtPosition<nvblox::TsdfVoxel>(
              tsdf_hash, nvblox::Vector3f(x, y, z_free), block_size, &tv_f) &&
          tv_f && tv_f->weight >= tsdf_weight_min) {
        d_o = tv_o->distance;
        d_f = tv_f->distance;
        if (d_f > bev_occ_tsdf_max_m && d_o <= bev_occ_tsdf_max_m) {
          const float delta = d_f - d_o;
          const float frac =
              (fabsf(delta) > 1e-12f)
                  ? clampf((bev_occ_tsdf_max_m - d_o) / delta, 0.f, 1.f)
                  : 0.f;
          z_hit = z_occ + frac * (z_free - z_occ);
        }
      }
    }

    const float clearance =
        isfinite(ceiling_z) ? (ceiling_z - z_hit) : kInf;
    if (clearance >= h_req) {
      surface_z = z_hit;
      floor_iz = iz;
      found = true;
      break;
    }
    ceiling_z = z_hit;
  }

  float L = unobserved_luminance;
  float out_elev = unknown_elev_z;
  // Cost stored as free-ness luminance [0,255]: 255 white/free, 0 black/lethal.
  float out_cost = unobserved_luminance;
  uint8_t out_obs = 0;

  if (found) {
    out_elev = surface_z;
    out_obs = 255;
    // Color of the floor bin (and free-side bin just above, where mesh color
    // usually lives). Not the whole column — that mixed ceilings/people/floor.
    if (lum_iz_u != nullptr && floor_iz >= 0) {
      uint32_t Lcode = lum_iz_u[cell * nz + floor_iz];
      if (floor_iz > 0) {
        Lcode = min(Lcode, lum_iz_u[cell * nz + (floor_iz - 1)]);
      }
      if (Lcode <= 255u) {
        L = static_cast<float>(Lcode);
      }
    }
    out_cost = L;
  } else if (
      prior_enabled && prior_elev && prior_cost && prior_obs && prior_w > 0 &&
      prior_h > 0 && prior_res > 0.f) {
    // Live cell center in global → prior frame: R(-yaw)*(p - t).
    const float x = origin_x + (static_cast<float>(ix) + 0.5f) * resolution;
    const float y = origin_y + (static_cast<float>(iy) + 0.5f) * resolution;
    const float dx = x - prior_tx;
    const float dy = y - prior_ty;
    const float c = cosf(-prior_yaw);
    const float s = sinf(-prior_yaw);
    const float xp = c * dx - s * dy;
    const float yp = s * dx + c * dy;
    const int pix =
        static_cast<int>(floorf((xp - prior_ox) / prior_res));
    const int piy =
        static_cast<int>(floorf((yp - prior_oy) / prior_res));
    if (pix >= 0 && piy >= 0 && pix < prior_w && piy < prior_h) {
      const int pcell = piy * prior_w + pix;
      if (prior_obs[pcell] != 0) {
        out_elev = prior_elev[pcell] + prior_tz;
        // Prior cost expected as free-ness [0,255] (same as live).
        out_cost = clampf(prior_cost[pcell], 0.f, 255.f);
        L = out_cost;
        out_obs = 255;
      }
    }
  }

  elev[cell] = out_elev;
  lum[cell] = L;
  if (cost) {
    cost[cell] = out_cost;
  }
  if (observed) {
    observed[cell] = out_obs;
  }
}

/// Finite-difference normals from elev; OR slope lethality into cost.
/// Cost is free-ness [0,255]; slope-lethal forces 0. Unobserved → 0 if
/// slope-lethal else leave (caller usually left unobserved_luminance).
__global__ void slope_normals_cost_kernel(
    const float* elev, float* cost, float* normals, const uint8_t* observed,
    int width, int height, float resolution, float cos_thresh) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= width || iy >= height) {
    return;
  }
  const int cell = iy * width + ix;
  const int xm = max(ix - 1, 0);
  const int xp = min(ix + 1, width - 1);
  const int ym = max(iy - 1, 0);
  const int yp = min(iy + 1, height - 1);
  const float inv_dx =
      1.f / (resolution * static_cast<float>(max(xp - xm, 1)));
  const float inv_dy =
      1.f / (resolution * static_cast<float>(max(yp - ym, 1)));
  const float dzdx =
      (elev[iy * width + xp] - elev[iy * width + xm]) * inv_dx;
  const float dzdy =
      (elev[yp * width + ix] - elev[ym * width + ix]) * inv_dy;
  float nx = -dzdx;
  float ny = -dzdy;
  float nz = 1.f;
  const float inv_n = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-12f);
  nx *= inv_n;
  ny *= inv_n;
  nz *= inv_n;
  if (normals) {
    normals[cell * 3 + 0] = nx;
    normals[cell * 3 + 1] = ny;
    normals[cell * 3 + 2] = nz;
  }
  if (cost) {
    const bool slope_lethal = (nz < cos_thresh);
    const bool known = (observed == nullptr) || (observed[cell] != 0);
    if (slope_lethal) {
      cost[cell] = 0.f;
    } else if (!known) {
      // Keep existing (typically unobserved_luminance); do not invent free.
    }
  }
}

void checkCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

bool blockOverlapsWindow(
    const nvblox::Index3D& bi, float block_size, float origin_x, float origin_y,
    float res, int width, int height, float z_min, float z_max) {
  const nvblox::Vector3f o =
      nvblox::getPositionFromBlockIndex(block_size, bi);
  const float x0 = o.x();
  const float y0 = o.y();
  const float z0 = o.z();
  const float x1 = x0 + block_size;
  const float y1 = y0 + block_size;
  const float z1 = z0 + block_size;
  const float wx0 = origin_x;
  const float wy0 = origin_y;
  const float wx1 = origin_x + width * res;
  const float wy1 = origin_y + height * res;
  if (x1 < wx0 || x0 > wx1 || y1 < wy0 || y0 > wy1) {
    return false;
  }
  if (z1 < z_min || z0 > z_max) {
    return false;
  }
  return true;
}

}  // namespace

TsdfColorToBevWorkspace::~TsdfColorToBevWorkspace() {
  if (elev_d) {
    cudaFree(elev_d);
  }
  if (lum_d) {
    cudaFree(lum_d);
  }
  if (cost_d) {
    cudaFree(cost_d);
  }
  if (normal_d) {
    cudaFree(normal_d);
  }
  if (obs_d) {
    cudaFree(obs_d);
  }
  if (lum_iz_u_d) {
    cudaFree(lum_iz_u_d);
  }
  if (occ_bits_d) {
    cudaFree(occ_bits_d);
  }
  if (free_bits_d) {
    cudaFree(free_bits_d);
  }
  if (block_indices_d) {
    cudaFree(block_indices_d);
  }
}

void TsdfColorToBevWorkspace::ensureCells(int cells, int nz) {
  const int words = std::max(1, (nz + 31) / 32);
  const bool need_cells = cells > cell_capacity;
  const bool need_nz = nz > nz_capacity;
  const bool need_bits = need_cells || words > bit_words || need_nz;
  if (!need_bits && !need_cells) {
    bit_words = words;
    return;
  }

  if (need_cells) {
    if (elev_d) {
      cudaFree(elev_d);
      elev_d = nullptr;
    }
    if (lum_d) {
      cudaFree(lum_d);
      lum_d = nullptr;
    }
    if (cost_d) {
      cudaFree(cost_d);
      cost_d = nullptr;
    }
    if (normal_d) {
      cudaFree(normal_d);
      normal_d = nullptr;
    }
    if (obs_d) {
      cudaFree(obs_d);
      obs_d = nullptr;
    }
    const size_t nbytes_f = static_cast<size_t>(cells) * sizeof(float);
    const size_t nbytes_n = nbytes_f * 3;
    const size_t nbytes_u = static_cast<size_t>(cells) * sizeof(uint8_t);
    checkCuda(cudaMalloc(&elev_d, nbytes_f), "cudaMalloc elev");
    checkCuda(cudaMalloc(&lum_d, nbytes_f), "cudaMalloc lum");
    checkCuda(cudaMalloc(&cost_d, nbytes_f), "cudaMalloc cost");
    checkCuda(cudaMalloc(&normal_d, nbytes_n), "cudaMalloc normal");
    checkCuda(cudaMalloc(&obs_d, nbytes_u), "cudaMalloc observed");
    cell_capacity = cells;
  }

  if (need_cells || need_nz) {
    if (lum_iz_u_d) {
      cudaFree(lum_iz_u_d);
      lum_iz_u_d = nullptr;
    }
    const int nz_alloc = std::max(nz, nz_capacity);
    checkCuda(
        cudaMalloc(
            &lum_iz_u_d,
            static_cast<size_t>(cell_capacity) * static_cast<size_t>(nz_alloc) *
                sizeof(uint32_t)),
        "cudaMalloc lum_iz_u");
    nz_capacity = nz_alloc;
  }

  if (occ_bits_d) {
    cudaFree(occ_bits_d);
    occ_bits_d = nullptr;
  }
  if (free_bits_d) {
    cudaFree(free_bits_d);
    free_bits_d = nullptr;
  }
  const size_t nbytes_bits =
      static_cast<size_t>(cell_capacity) * words * sizeof(uint32_t);
  checkCuda(cudaMalloc(&occ_bits_d, nbytes_bits), "cudaMalloc occ_bits");
  checkCuda(cudaMalloc(&free_bits_d, nbytes_bits), "cudaMalloc free_bits");
  bit_words = words;
}

void TsdfColorToBevWorkspace::ensureBlocks(int n_blocks) {
  if (n_blocks <= block_capacity) {
    return;
  }
  if (block_indices_d) {
    cudaFree(block_indices_d);
    block_indices_d = nullptr;
  }
  checkCuda(
      cudaMalloc(
          &block_indices_d, static_cast<size_t>(n_blocks) * sizeof(nvblox::Index3D)),
      "cudaMalloc block_indices");
  block_capacity = n_blocks;
}

void tsdf_color_to_bev(
    const nvblox::TsdfLayer& tsdf, const nvblox::ColorLayer& color,
    const TsdfColorToBevParams& params, TsdfColorToBevWorkspace* workspace,
    float* elev_host, float* lum_host, float* cost_host,
    uint8_t* observed_host, const nvblox::CudaStream& cuda_stream,
    const PriorBevGpu* prior) {
  if (params.width <= 0 || params.height <= 0 || elev_host == nullptr ||
      lum_host == nullptr || workspace == nullptr) {
    return;
  }

  const int n = params.width * params.height;
  const float voxel_size = tsdf.voxel_size();
  const float block_size = tsdf.block_size();
  const float step = fmaxf(voxel_size, 1e-4f);
  int nz = static_cast<int>(floorf((params.z_max - params.z_min) / step)) + 1;
  nz = std::max(1, std::min(nz, kMaxNz));

  workspace->ensureCells(n, nz);
  const int n_words = workspace->bit_words;
  const size_t nbytes_bits =
      static_cast<size_t>(n) * n_words * sizeof(uint32_t);
  const size_t nbytes_f = static_cast<size_t>(n) * sizeof(float);
  checkCuda(
      cudaMemsetAsync(workspace->occ_bits_d, 0, nbytes_bits, cuda_stream),
      "memset occ_bits");
  checkCuda(
      cudaMemsetAsync(workspace->free_bits_d, 0, nbytes_bits, cuda_stream),
      "memset free_bits");
  checkCuda(
      cudaMemsetAsync(
          workspace->lum_iz_u_d, 0xFF,
          static_cast<size_t>(n) * static_cast<size_t>(nz) * sizeof(uint32_t),
          cuda_stream),
      "memset lum_iz_u");
  // Costmap default = unobserved_luminance (SSoT); finalize overwrites cells.
  {
    const float fill = params.unobserved_luminance;
    std::vector<float> cost_init(static_cast<size_t>(n), fill);
    checkCuda(
        cudaMemcpyAsync(
            workspace->cost_d, cost_init.data(), nbytes_f,
            cudaMemcpyHostToDevice, cuda_stream),
        "H2D cost init unobserved");
  }

  std::vector<nvblox::Index3D> blocks;
  blocks.reserve(static_cast<size_t>(tsdf.numBlocks()));
  for (const nvblox::Index3D& bi : tsdf.getAllBlockIndices()) {
    if (blockOverlapsWindow(
            bi, block_size, params.origin_x, params.origin_y, params.resolution,
            params.width, params.height, params.z_min, params.z_max)) {
      blocks.push_back(bi);
    }
  }

  auto& tsdf_view = tsdf.getGpuLayerView(cuda_stream);
  auto& color_view = color.getGpuLayerView(cuda_stream);
  const auto tsdf_hash = tsdf_view.getHash().impl_;
  const auto color_hash = color_view.getHash().impl_;

  if (!blocks.empty()) {
    workspace->ensureBlocks(static_cast<int>(blocks.size()));
    checkCuda(
        cudaMemcpyAsync(
            workspace->block_indices_d, blocks.data(),
            blocks.size() * sizeof(nvblox::Index3D), cudaMemcpyHostToDevice,
            cuda_stream),
        "H2D block_indices");

    const float color_band = fmaxf(4.f * voxel_size, params.color_z_search_half_band_m);
    scatter_block_voxels_kernel<<<static_cast<int>(blocks.size()),
                                  kVoxelsPerBlock, 0, cuda_stream>>>(
        tsdf_hash, color_hash, workspace->block_indices_d,
        static_cast<int>(blocks.size()), block_size, params.origin_x,
        params.origin_y, params.resolution, params.width, params.height,
        params.z_min, params.z_max, step, nz, params.tsdf_weight_min,
        params.bev_occ_tsdf_max_m, params.color_weight_min, color_band,
        voxel_size, n_words, workspace->occ_bits_d, workspace->free_bits_d,
        workspace->lum_iz_u_d);
    checkCuda(cudaGetLastError(), "scatter_block_voxels_kernel");
  }

  const bool prior_ok =
      prior && prior->enabled && prior->elev_d && prior->cost_d &&
      prior->obs_d && prior->width > 0 && prior->height > 0;
  dim3 t2(16, 16);
  dim3 b2(
      (params.width + t2.x - 1) / t2.x, (params.height + t2.y - 1) / t2.y);
  finalize_from_bits_kernel<<<b2, t2, 0, cuda_stream>>>(
      tsdf_hash, block_size, params.origin_x, params.origin_y,
      params.resolution, params.width, params.height, params.z_max, step, nz,
      params.unknown_elev_z, params.robot_height_m, params.tsdf_weight_min,
      params.bev_occ_tsdf_max_m, params.unobserved_luminance, n_words,
      workspace->occ_bits_d, workspace->free_bits_d, workspace->lum_iz_u_d,
      workspace->elev_d, workspace->lum_d, workspace->cost_d, workspace->obs_d,
      prior_ok ? prior->elev_d : nullptr,
      prior_ok ? prior->cost_d : nullptr,
      prior_ok ? prior->obs_d : nullptr,
      prior_ok ? prior->origin_x : 0.f,
      prior_ok ? prior->origin_y : 0.f,
      prior_ok ? prior->resolution : 0.f,
      prior_ok ? prior->width : 0,
      prior_ok ? prior->height : 0,
      prior_ok ? prior->tx : 0.f,
      prior_ok ? prior->ty : 0.f,
      prior_ok ? prior->tz : 0.f,
      prior_ok ? prior->yaw : 0.f,
      prior_ok);
  checkCuda(cudaGetLastError(), "finalize_from_bits_kernel");

  const size_t nbytes_u = static_cast<size_t>(n) * sizeof(uint8_t);
  checkCuda(
      cudaMemcpyAsync(
          elev_host, workspace->elev_d, nbytes_f, cudaMemcpyDeviceToHost,
          cuda_stream),
      "D2H elev");
  checkCuda(
      cudaMemcpyAsync(
          lum_host, workspace->lum_d, nbytes_f, cudaMemcpyDeviceToHost,
          cuda_stream),
      "D2H lum");
  if (cost_host) {
    checkCuda(
        cudaMemcpyAsync(
            cost_host, workspace->cost_d, nbytes_f, cudaMemcpyDeviceToHost,
            cuda_stream),
        "D2H cost");
  }
  if (observed_host) {
    checkCuda(
        cudaMemcpyAsync(
            observed_host, workspace->obs_d, nbytes_u, cudaMemcpyDeviceToHost,
            cuda_stream),
        "D2H observed");
  }
  cuda_stream.synchronize();
}

void apply_slope_normals_cost(
    TsdfColorToBevWorkspace* workspace, int width, int height, float resolution,
    float lethal_slope_deg, float* elev_host, float* cost_host,
    const uint8_t* observed_host, float* normal_host,
    const nvblox::CudaStream& cuda_stream) {
  if (workspace == nullptr || elev_host == nullptr || width <= 0 || height <= 0) {
    return;
  }
  const int n = width * height;
  workspace->ensureCells(n, workspace->bit_words > 0 ? workspace->bit_words * 32 : 1);
  const size_t nbytes_f = static_cast<size_t>(n) * sizeof(float);
  const size_t nbytes_u = static_cast<size_t>(n) * sizeof(uint8_t);

  checkCuda(
      cudaMemcpyAsync(
          workspace->elev_d, elev_host, nbytes_f, cudaMemcpyHostToDevice,
          cuda_stream),
      "H2D elev for slope");
  if (cost_host) {
    checkCuda(
        cudaMemcpyAsync(
            workspace->cost_d, cost_host, nbytes_f, cudaMemcpyHostToDevice,
            cuda_stream),
        "H2D cost for slope");
  }
  if (observed_host) {
    checkCuda(
        cudaMemcpyAsync(
            workspace->obs_d, observed_host, nbytes_u, cudaMemcpyHostToDevice,
            cuda_stream),
        "H2D observed for slope");
  }

  const float cos_thresh = cosf(lethal_slope_deg * 0.017453292519943295f);
  dim3 t2(16, 16);
  dim3 b2((width + t2.x - 1) / t2.x, (height + t2.y - 1) / t2.y);
  slope_normals_cost_kernel<<<b2, t2, 0, cuda_stream>>>(
      workspace->elev_d, workspace->cost_d, workspace->normal_d,
      observed_host ? workspace->obs_d : nullptr, width, height, resolution,
      cos_thresh);
  checkCuda(cudaGetLastError(), "slope_normals_cost_kernel");

  if (cost_host) {
    checkCuda(
        cudaMemcpyAsync(
            cost_host, workspace->cost_d, nbytes_f, cudaMemcpyDeviceToHost,
            cuda_stream),
        "D2H cost after slope");
  }
  if (normal_host) {
    checkCuda(
        cudaMemcpyAsync(
            normal_host, workspace->normal_d, nbytes_f * 3,
            cudaMemcpyDeviceToHost, cuda_stream),
        "D2H normal");
  }
  cuda_stream.synchronize();
}

}  // namespace hound_mapping
