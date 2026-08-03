// Replay a rail-sim dataset through MapperCore + ElevationCostExtractor.
// Measures depth/trav → elev/cost latency and RMSE vs GT heightfield crop.
//
//   ros2 run hound_mapping smoke_rail_mapper --ros-args
//     -p dataset:=/root/colcon_ws/src/hound_mapping/data/rail_sim/race-2

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nvblox/nvblox.h>

#include "hound_mapping/elevation_cost.hpp"
#include "hound_mapping/mapper_core.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct CamCfg {
  int width = 640;
  int height = 360;
  float fx = 400.f;
  float fy = 400.f;
  float cx = 320.f;
  float cy = 180.f;
  float max_depth_m = 6.f;
  int n_frames = 0;
  float map_res = 0.1f;
  float window_radius_m = 10.f;
  float voxel_size = 0.05f;
  int elev_h = 0;
  int elev_w = 0;
};

std::unordered_map<std::string, std::string> loadKv(const std::string& path)
{
  std::unordered_map<std::string, std::string> kv;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    kv[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return kv;
}

CamCfg loadCamera(const std::string& dataset)
{
  const auto kv = loadKv(dataset + "/camera.txt");
  CamCfg c;
  auto getf = [&](const char* k, float def) {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::stof(it->second);
  };
  auto geti = [&](const char* k, int def) {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::stoi(it->second);
  };
  c.width = geti("width", c.width);
  c.height = geti("height", c.height);
  c.fx = getf("fx", c.fx);
  c.fy = getf("fy", c.fy);
  c.cx = getf("cx", c.cx);
  c.cy = getf("cy", c.cy);
  c.max_depth_m = getf("max_depth_m", c.max_depth_m);
  c.n_frames = geti("n_frames", 0);
  c.map_res = getf("map_res", c.map_res);
  c.window_radius_m = getf("window_radius_m", c.window_radius_m);
  c.voxel_size = getf("voxel_size", c.voxel_size);
  c.elev_h = geti("elev_h", 0);
  c.elev_w = geti("elev_w", 0);
  return c;
}

std::vector<nvblox::Transform> loadPoses(const std::string& path, int n)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<nvblox::Transform> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    double m[16];
    in.read(reinterpret_cast<char*>(m), sizeof(m));
    if (!in) {
      break;
    }
    Eigen::Matrix4f M;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        M(r, c) = static_cast<float>(m[r * 4 + c]);
      }
    }
    nvblox::Transform T = nvblox::Transform::Identity();
    T.matrix() = M;
    out.push_back(T);
  }
  return out;
}

}  // namespace

static void fillDepth(nvblox::DepthImage* depth, const cv::Mat& mm)
{
  for (int r = 0; r < mm.rows; ++r) {
    const uint16_t* src = mm.ptr<uint16_t>(r);
    for (int c = 0; c < mm.cols; ++c) {
      (*depth)(r, c) = src[c] > 0 ? src[c] * 0.001f : 0.f;
    }
  }
}

static void fillColor(nvblox::ColorImage* color, const cv::Mat& bgr)
{
  for (int r = 0; r < bgr.rows; ++r) {
    for (int c = 0; c < bgr.cols; ++c) {
      const cv::Vec3b& p = bgr.at<cv::Vec3b>(r, c);
      (*color)(r, c) = nvblox::Color(p[2], p[1], p[0]);
    }
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("smoke_rail_mapper");
  node->declare_parameter("dataset", "");
  node->declare_parameter("max_frames", 0);
  node->declare_parameter("integrate_every", 1);
  node->declare_parameter("clear_radius_scale", 1.0);
  node->declare_parameter("dump_frame_elev", true);

  const std::string dataset = node->get_parameter("dataset").as_string();
  if (dataset.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Set -p dataset:=/path/to/rail_sim/seq");
    return 1;
  }
  CamCfg cam = loadCamera(dataset);
  int n = cam.n_frames;
  const int max_frames = node->get_parameter("max_frames").as_int();
  if (max_frames > 0) {
    n = std::min(n, max_frames);
  }
  const int every = std::max(
      1, static_cast<int>(node->get_parameter("integrate_every").as_int()));
  const bool dump_frame_elev = node->get_parameter("dump_frame_elev").as_bool();

  auto poses = loadPoses(dataset + "/frames/poses_T_wo.bin", n);
  if (static_cast<int>(poses.size()) < n) {
    n = static_cast<int>(poses.size());
  }
  RCLCPP_INFO(
      node->get_logger(),
      "smoke_rail_mapper: %s  %d frames  cam %dx%d  window=%.1fm",
      dataset.c_str(), n, cam.width, cam.height, cam.window_radius_m);

  hound_mapping::MapperConfig mc;
  mc.voxel_size_m = cam.voxel_size;
  mc.map_clearing_radius_m = cam.window_radius_m;
  mc.max_integration_distance_m = cam.max_depth_m + 1.f;
  hound_mapping::MapperCore core(mc);

  hound_mapping::ElevationCostConfig ec;
  ec.resolution_m = cam.voxel_size;
  ec.window_radius_m = cam.window_radius_m;
  ec.elev_z_min_rel = -5.f;
  ec.elev_z_max_rel = 2.f;
  ec.do_inpaint = true;
  ec.inpaint_resolution_m = 0.25f;
  ec.inpaint_radius = 3;
  hound_mapping::ElevationCostExtractor extractor(ec);
  RCLCPP_INFO(
      node->get_logger(),
      "unknowns → low-res Telea @ %.2fm then slope/normals",
      ec.inpaint_resolution_m);

  nvblox::Camera camera(cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height);

  std::vector<float> gt_elev;
  if (cam.elev_h > 0 && cam.elev_w > 0) {
    const std::string ep = dataset + "/maps/elevation.f32";
    std::ifstream ein(ep, std::ios::binary);
    gt_elev.resize(static_cast<size_t>(cam.elev_h) * cam.elev_w);
    ein.read(
        reinterpret_cast<char*>(gt_elev.data()),
        static_cast<std::streamsize>(gt_elev.size() * sizeof(float)));
    if (!ein) {
      RCLCPP_WARN(node->get_logger(), "Failed to read %s", ep.c_str());
      gt_elev.clear();
    }
  }

  auto sample_gt = [&](float x, float y) -> float {
    if (gt_elev.empty()) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const float col = x / cam.map_res;
    const float row = y / cam.map_res;
    const int c0 = static_cast<int>(std::floor(col));
    const int r0 = static_cast<int>(std::floor(row));
    if (c0 < 0 || r0 < 0 || c0 + 1 >= cam.elev_w || r0 + 1 >= cam.elev_h) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    const float dc = col - c0;
    const float dr = row - r0;
    const auto at = [&](int r, int c) {
      return gt_elev[static_cast<size_t>(r) * cam.elev_w + c];
    };
    return (1 - dc) * (1 - dr) * at(r0, c0) + dc * (1 - dr) * at(r0, c0 + 1) +
           (1 - dc) * dr * at(r0 + 1, c0) + dc * dr * at(r0 + 1, c0 + 1);
  };

  std::vector<double> integrate_ms;
  std::vector<double> clear_ms;
  std::vector<double> extract_ms;
  std::vector<double> bev_ms;
  std::vector<double> inpaint_ms;
  std::vector<double> slope_ms;
  std::vector<double> e2e_ms;
  std::vector<double> rmse_m;
  cv::Mat last_elev;
  cv::Mat last_cost;
  float last_ox = 0.f;
  float last_oy = 0.f;
  float last_res = cam.voxel_size;

  for (int i = 0; i < n; ++i) {
    if (i % every != 0 && i + 1 < n) {
      continue;
    }
    char stem[64];
    std::snprintf(stem, sizeof(stem), "%06d", i);
    const std::string depth_path =
        dataset + "/frames/" + stem + "_depth_mm.png";
    const std::string trav_path = dataset + "/frames/" + stem + "_trav.png";
    cv::Mat depth_mm = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    cv::Mat trav_bgr = cv::imread(trav_path, cv::IMREAD_COLOR);
    if (depth_mm.empty() || trav_bgr.empty()) {
      RCLCPP_WARN(node->get_logger(), "Missing frame %s", stem);
      continue;
    }
    if (depth_mm.type() != CV_16UC1) {
      RCLCPP_ERROR(node->get_logger(), "depth not 16UC1");
      return 1;
    }

    nvblox::DepthImage depth(
        cam.height, cam.width, nvblox::MemoryType::kHost);
    nvblox::ColorImage color(
        cam.height, cam.width, nvblox::MemoryType::kHost);
    fillDepth(&depth, depth_mm);
    fillColor(&color, trav_bgr);

    const nvblox::Transform& T_L_C = poses[i];
    const auto t0 = Clock::now();
    core.integrateDepth(depth, T_L_C, camera);
    core.integrateColor(color, T_L_C, camera);
    const double t_int =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    const auto t1 = Clock::now();
    // Elev BEV reads TSDF directly (no ESDF layer).
    core.clearOutsideRadius(T_L_C.translation());
    const double t_clear =
        std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

    hound_mapping::ExtractTiming timing;
    const auto t2 = Clock::now();
    const auto maps = extractor.extract(
        core, T_L_C.translation(), T_L_C.translation().z(), &timing);
    const double t_ext =
        std::chrono::duration<double, std::milli>(Clock::now() - t2).count();
    const double t_e2e =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    integrate_ms.push_back(t_int);
    clear_ms.push_back(t_clear);
    extract_ms.push_back(t_ext);
    bev_ms.push_back(timing.bev_ms);
    inpaint_ms.push_back(timing.inpaint_ms);
    slope_ms.push_back(timing.slope_ms);
    e2e_ms.push_back(t_e2e);
    if (maps.width > 0) {
      last_elev = maps.elevation.clone();
      last_cost = maps.cost.clone();
      last_ox = maps.origin_x;
      last_oy = maps.origin_y;
      last_res = maps.resolution;

      if (dump_frame_elev) {
        const std::string elev_bin =
            dataset + "/frames/" + stem + "_elev.f32";
        const std::string elev_txt =
            dataset + "/frames/" + stem + "_elev.txt";
        std::ofstream bout(elev_bin, std::ios::binary);
        bout.write(
            reinterpret_cast<const char*>(maps.elevation.ptr<float>(0)),
            static_cast<std::streamsize>(
                maps.elevation.total() * sizeof(float)));
        std::ofstream tout(elev_txt);
        tout << "height=" << maps.height << "\n"
             << "width=" << maps.width << "\n"
             << "resolution=" << maps.resolution << "\n"
             << "origin_x=" << maps.origin_x << "\n"
             << "origin_y=" << maps.origin_y << "\n";
        const std::string cost_bin =
            dataset + "/frames/" + stem + "_cost.f32";
        std::ofstream cout_bin(cost_bin, std::ios::binary);
        cout_bin.write(
            reinterpret_cast<const char*>(maps.cost.ptr<float>(0)),
            static_cast<std::streamsize>(maps.cost.total() * sizeof(float)));
        // Observed mask for scrubber (optional).
        const std::string obs_bin =
            dataset + "/frames/" + stem + "_observed.u8";
        std::ofstream oout(obs_bin, std::ios::binary);
        oout.write(
            reinterpret_cast<const char*>(maps.observed.ptr<uint8_t>(0)),
            static_cast<std::streamsize>(maps.observed.total()));
      }
    }

    // RMSE vs GT heightfield on observed cells.
    if (!gt_elev.empty() && maps.width > 0) {
      double sse = 0.0;
      int cnt = 0;
      for (int r = 0; r < maps.height; ++r) {
        for (int c = 0; c < maps.width; ++c) {
          if (maps.observed.at<uint8_t>(r, c) == 0) {
            continue;
          }
          const float zm = maps.elevation.at<float>(r, c);
          if (!std::isfinite(zm)) {
            continue;
          }
          const float xw = maps.origin_x + (c + 0.5f) * maps.resolution;
          const float yw = maps.origin_y + (r + 0.5f) * maps.resolution;
          const float zg = sample_gt(xw, yw);
          if (!std::isfinite(zg)) {
            continue;
          }
          const double d = static_cast<double>(zm - zg);
          sse += d * d;
          ++cnt;
        }
      }
      if (cnt > 0) {
        rmse_m.push_back(std::sqrt(sse / cnt));
      }
    }

    if (i % 20 == 0 || i + 1 == n) {
      const double rmse = rmse_m.empty() ? -1.0 : rmse_m.back();
      int obs = 0;
      if (maps.width > 0) {
        obs = cv::countNonZero(maps.observed);
      }
      const int cells = maps.width * maps.height;
      RCLCPP_INFO(
          node->get_logger(),
          "[%d/%d] int=%.1f clear=%.1f extract=%.1f (bev=%.1f) "
          "e2e=%.1f ms  rmse=%.3fm  observed=%d/%d (%.1f%%)",
          i + 1, n, t_int, t_clear, t_ext, timing.bev_ms,
          t_e2e, rmse, obs, cells,
          cells > 0 ? 100.0 * obs / cells : 0.0);
    }
  }

  auto stats = [](const std::vector<double>& xs, const char* unit) {
    if (xs.empty()) {
      return std::string("n=0");
    }
    double s = 0;
    for (double x : xs) {
      s += x;
    }
    std::ostringstream oss;
    oss << "n=" << xs.size() << " mean=" << (s / xs.size()) << " " << unit;
    return oss.str();
  };
  RCLCPP_INFO(node->get_logger(), "integrate: %s", stats(integrate_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "clear:     %s", stats(clear_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "extract:   %s", stats(extract_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "  bev:     %s", stats(bev_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "  inpaint: %s", stats(inpaint_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "  slope:   %s", stats(slope_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "e2e:       %s", stats(e2e_ms, "ms").c_str());
  RCLCPP_INFO(node->get_logger(), "elev RMSE: %s", stats(rmse_m, "m").c_str());

  if (!last_elev.empty()) {
    const std::string dump = dataset + "/mapped_elev.f32";
    const std::string meta = dataset + "/mapped_elev.txt";
    std::ofstream out(dump, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(last_elev.ptr<float>(0)),
        static_cast<std::streamsize>(
            last_elev.total() * sizeof(float)));
    std::ofstream mout(meta);
    mout << "height=" << last_elev.rows << "\n"
         << "width=" << last_elev.cols << "\n"
         << "resolution=" << last_res << "\n"
         << "origin_x=" << last_ox << "\n"
         << "origin_y=" << last_oy << "\n";
    if (!last_cost.empty()) {
      const std::string cdump = dataset + "/mapped_cost.f32";
      std::ofstream coutf(cdump, std::ios::binary);
      coutf.write(
          reinterpret_cast<const char*>(last_cost.ptr<float>(0)),
          static_cast<std::streamsize>(
              last_cost.total() * sizeof(float)));
    }
    RCLCPP_INFO(
        node->get_logger(),
        "Wrote mapped elev %dx%d → %s", last_elev.rows, last_elev.cols,
        dump.c_str());
  }

  RCLCPP_INFO(
      node->get_logger(),
      "Viz: python3 .../viser_rail_compare.py --dataset %s "
      "--mapped-elev %s/mapped_elev.f32",
      dataset.c_str(), dataset.c_str());

  rclcpp::shutdown();
  return 0;
}
