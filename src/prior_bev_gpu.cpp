#include "hound_mapping/prior_bev_gpu.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace hound_mapping {
namespace {

void checkCuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("PriorBevGpu ") + what + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

PriorBevGpu::~PriorBevGpu()
{
  clear();
}

void PriorBevGpu::clear()
{
  if (elev_d) {
    cudaFree(elev_d);
    elev_d = nullptr;
  }
  if (cost_d) {
    cudaFree(cost_d);
    cost_d = nullptr;
  }
  if (obs_d) {
    cudaFree(obs_d);
    obs_d = nullptr;
  }
  width = 0;
  height = 0;
  enabled = false;
}

bool PriorBevGpu::upload(
    const cv::Mat& elev, const cv::Mat& cost, const cv::Mat& observed,
    float origin_x_in, float origin_y_in, float resolution_in, float tx_in,
    float ty_in, float tz_in, float yaw_in,
    const nvblox::CudaStream& cuda_stream)
{
  if (elev.empty() || cost.empty() || observed.empty() ||
      elev.type() != CV_32FC1 || cost.type() != CV_32FC1 ||
      observed.type() != CV_8UC1 || elev.rows != cost.rows ||
      elev.cols != cost.cols || elev.rows != observed.rows ||
      elev.cols != observed.cols || !elev.isContinuous() ||
      !cost.isContinuous() || !observed.isContinuous()) {
    return false;
  }

  clear();
  width = elev.cols;
  height = elev.rows;
  origin_x = origin_x_in;
  origin_y = origin_y_in;
  resolution = resolution_in;
  tx = tx_in;
  ty = ty_in;
  tz = tz_in;
  yaw = yaw_in;

  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  checkCuda(cudaMalloc(&elev_d, n * sizeof(float)), "cudaMalloc elev");
  checkCuda(cudaMalloc(&cost_d, n * sizeof(float)), "cudaMalloc cost");
  checkCuda(cudaMalloc(&obs_d, n * sizeof(uint8_t)), "cudaMalloc obs");
  checkCuda(
      cudaMemcpyAsync(
          elev_d, elev.ptr<float>(), n * sizeof(float),
          cudaMemcpyHostToDevice, cuda_stream),
      "H2D elev");
  checkCuda(
      cudaMemcpyAsync(
          cost_d, cost.ptr<float>(), n * sizeof(float),
          cudaMemcpyHostToDevice, cuda_stream),
      "H2D cost");
  checkCuda(
      cudaMemcpyAsync(
          obs_d, observed.ptr<uint8_t>(), n * sizeof(uint8_t),
          cudaMemcpyHostToDevice, cuda_stream),
      "H2D obs");
  cuda_stream.synchronize();
  enabled = true;
  return true;
}

}  // namespace hound_mapping
