#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

#include <nvblox/core/cuda_stream.h>

namespace hound_mapping {

/// Full-coverage prior elev/cost/observed resident on GPU for loop1 Finalize.
struct PriorBevGpu {
  float* elev_d{nullptr};
  float* cost_d{nullptr};
  uint8_t* obs_d{nullptr};
  float origin_x{0.f};
  float origin_y{0.f};
  float resolution{0.05f};
  int width{0};
  int height{0};
  /// T_global_prior: p_global = R(yaw) * p_prior + (tx,ty,tz)
  float tx{0.f};
  float ty{0.f};
  float tz{0.f};
  float yaw{0.f};
  bool enabled{false};

  PriorBevGpu() = default;
  PriorBevGpu(const PriorBevGpu&) = delete;
  PriorBevGpu& operator=(const PriorBevGpu&) = delete;
  ~PriorBevGpu();

  void clear();
  /// Upload host LocalMaps-style buffers (CV_32FC1 elev/cost, CV_8UC1 obs).
  bool upload(
      const cv::Mat& elev, const cv::Mat& cost, const cv::Mat& observed,
      float origin_x_in, float origin_y_in, float resolution_in,
      float tx_in, float ty_in, float tz_in, float yaw_in,
      const nvblox::CudaStream& cuda_stream);
};

}  // namespace hound_mapping
