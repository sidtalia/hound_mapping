#include "hound_mapping/lethal_overlay_gpu.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

namespace hound_mapping {
namespace {

void checkCuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("LethalOverlayGpu ") + what + ": " +
        cudaGetErrorString(err));
  }
}

std::string trim(std::string s)
{
  auto is_sp = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && is_sp(static_cast<unsigned char>(s.front()))) {
    s.erase(s.begin());
  }
  while (!s.empty() && is_sp(static_cast<unsigned char>(s.back()))) {
    s.pop_back();
  }
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                        (s.front() == '\'' && s.back() == '\''))) {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

bool parseSimpleYaml(
    const std::string& path, std::unordered_map<std::string, std::string>* kv)
{
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, colon));
    const std::string val = trim(line.substr(colon + 1));
    if (key.empty() || key == "polygons") {
      continue;
    }
    (*kv)[key] = val;
  }
  return true;
}

}  // namespace

LethalOverlayGpu::~LethalOverlayGpu()
{
  clear();
}

void LethalOverlayGpu::clear()
{
  if (mask_d) {
    cudaFree(mask_d);
    mask_d = nullptr;
  }
  width = 0;
  height = 0;
  enabled = false;
}

bool LethalOverlayGpu::upload(
    const cv::Mat& mask, float origin_x_in, float origin_y_in,
    float resolution_in, float tx_in, float ty_in, float yaw_in,
    const nvblox::CudaStream& cuda_stream)
{
  if (mask.empty() || mask.type() != CV_32FC1 || !mask.isContinuous() ||
      mask.rows <= 0 || mask.cols <= 0 || !(resolution_in > 0.f)) {
    return false;
  }

  clear();
  width = mask.cols;
  height = mask.rows;
  origin_x = origin_x_in;
  origin_y = origin_y_in;
  resolution = resolution_in;
  tx = tx_in;
  ty = ty_in;
  yaw = yaw_in;

  const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
  checkCuda(cudaMalloc(&mask_d, n * sizeof(float)), "cudaMalloc mask");
  checkCuda(
      cudaMemcpyAsync(
          mask_d, mask.ptr<float>(), n * sizeof(float),
          cudaMemcpyHostToDevice, cuda_stream),
      "H2D mask");
  cuda_stream.synchronize();
  enabled = true;
  return true;
}

bool LethalOverlayGpu::loadYaml(
    const std::string& yaml_path, float tx_in, float ty_in, float yaw_in,
    const nvblox::CudaStream& cuda_stream, std::string* err)
{
  std::unordered_map<std::string, std::string> kv;
  if (!parseSimpleYaml(yaml_path, &kv)) {
    if (err) {
      *err = "cannot read " + yaml_path;
    }
    return false;
  }

  auto need = [&](const char* key) -> const std::string* {
    const auto it = kv.find(key);
    if (it == kv.end() || it->second.empty()) {
      if (err) {
        *err = std::string("missing ") + key + " in " + yaml_path;
      }
      return nullptr;
    }
    return &it->second;
  };

  const std::string* w_s = need("width");
  const std::string* h_s = need("height");
  const std::string* r_s = need("resolution");
  const std::string* ox_s = need("origin_x");
  const std::string* oy_s = need("origin_y");
  if (!w_s || !h_s || !r_s || !ox_s || !oy_s) {
    return false;
  }

  const int w = std::stoi(*w_s);
  const int h = std::stoi(*h_s);
  const float res = std::stof(*r_s);
  const float ox = std::stof(*ox_s);
  const float oy = std::stof(*oy_s);
  if (w <= 0 || h <= 0 || !(res > 0.f)) {
    if (err) {
      *err = "invalid width/height/resolution in " + yaml_path;
    }
    return false;
  }

  std::string bin_name = "lethal.f32";
  if (const auto it = kv.find("lethal_f32"); it != kv.end() && !it->second.empty()) {
    bin_name = it->second;
  }
  std::filesystem::path yaml(yaml_path);
  std::filesystem::path bin = std::filesystem::path(bin_name);
  if (bin.is_relative()) {
    bin = yaml.parent_path() / bin;
  }
  if (!std::filesystem::exists(bin)) {
    if (err) {
      *err = "lethal_f32 not found: " + bin.string();
    }
    return false;
  }

  const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
  cv::Mat mask(h, w, CV_32FC1);
  std::ifstream in(bin, std::ios::binary);
  in.read(
      reinterpret_cast<char*>(mask.ptr<float>()),
      static_cast<std::streamsize>(n * sizeof(float)));
  if (!in || static_cast<size_t>(in.gcount()) != n * sizeof(float)) {
    if (err) {
      *err = "lethal_f32 size mismatch (want " + std::to_string(n) +
          " floats): " + bin.string();
    }
    return false;
  }

  if (!upload(mask, ox, oy, res, tx_in, ty_in, yaw_in, cuda_stream)) {
    if (err) {
      *err = "GPU upload failed";
    }
    return false;
  }
  return true;
}

}  // namespace hound_mapping
