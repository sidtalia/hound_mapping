// Oneshot: LayerCake → full BEV elev/cost yaml (same grid as prior fill).
//
//   layercake_to_bev --layercake /root/colcon_ws/maps/hound_tsdf.layercake
//   layercake_to_bev --layercake cake.layercake --out cake.elev.yaml

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <glog/logging.h>

#include "hound_mapping/elevation_cost.hpp"

namespace {

void printUsage(const char* argv0)
{
  std::fprintf(
      stderr,
      "Usage: %s --layercake PATH [--out PATH] [--resolution M] "
      "[--voxel-size M]\n",
      argv0);
}

bool writeLocalMapYaml(
    const hound_mapping::LocalMaps& maps, const std::string& yaml_path,
    std::string* err)
{
  namespace fs = std::filesystem;
  const fs::path yaml(yaml_path);
  try {
    if (!yaml.parent_path().empty()) {
      fs::create_directories(yaml.parent_path());
    }
  } catch (const std::exception& ex) {
    if (err) {
      *err = std::string("mkdir failed: ") + ex.what();
    }
    return false;
  }

  fs::path base = yaml;
  if (base.extension() == ".yaml" || base.extension() == ".yml") {
    base = base.replace_extension();
  }
  if (base.extension() == ".elev") {
    base = base.replace_extension();
  }
  const fs::path elev_bin = fs::path(base.string() + ".elev.f32");
  const fs::path cost_bin = fs::path(base.string() + ".cost.f32");

  if (maps.elevation.empty() || maps.cost.empty() ||
      maps.elevation.type() != CV_32FC1 || maps.cost.type() != CV_32FC1) {
    if (err) {
      *err = "empty elev/cost";
    }
    return false;
  }

  {
    std::ofstream out(elev_bin, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(maps.elevation.ptr<float>(0)),
        static_cast<std::streamsize>(maps.elevation.total() * sizeof(float)));
    if (!out) {
      if (err) {
        *err = "write failed: " + elev_bin.string();
      }
      return false;
    }
  }
  {
    std::ofstream out(cost_bin, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(maps.cost.ptr<float>(0)),
        static_cast<std::streamsize>(maps.cost.total() * sizeof(float)));
    if (!out) {
      if (err) {
        *err = "write failed: " + cost_bin.string();
      }
      return false;
    }
  }

  std::ofstream y(yaml);
  y << "# LocalMap snapshot from LayerCake BEV (same grid as prior fill).\n"
    << "frame_id: map\n"
    << "width: " << maps.width << "\n"
    << "height: " << maps.height << "\n"
    << "resolution: " << maps.resolution << "\n"
    << "origin_x: " << maps.origin_x << "\n"
    << "origin_y: " << maps.origin_y << "\n"
    << "origin_z: 0.0\n"
    << "elevation_f32: " << elev_bin.filename().string() << "\n"
    << "cost_f32: " << cost_bin.filename().string() << "\n";
  if (!y) {
    if (err) {
      *err = "write failed: " + yaml.string();
    }
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  FLAGS_minloglevel = google::GLOG_WARNING;

  std::string cake;
  std::string out;
  float resolution = 0.05f;
  float voxel_size = 0.05f;
  float robot_height = 0.4f;
  float bev_occ = 0.025f;
  float unobserved = 252.f;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto take = [&](float* dst) {
      if (i + 1 >= argc) {
        return false;
      }
      *dst = std::strtof(argv[++i], nullptr);
      return true;
    };
    if (std::strcmp(a, "--layercake") == 0 && i + 1 < argc) {
      cake = argv[++i];
    } else if (std::strcmp(a, "--out") == 0 && i + 1 < argc) {
      out = argv[++i];
    } else if (std::strcmp(a, "--resolution") == 0) {
      if (!take(&resolution)) {
        printUsage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(a, "--voxel-size") == 0) {
      if (!take(&voxel_size)) {
        printUsage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(a, "--robot-height") == 0) {
      if (!take(&robot_height)) {
        printUsage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(a, "--bev-occ-tsdf-max") == 0) {
      if (!take(&bev_occ)) {
        printUsage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(a, "--unobserved-luminance") == 0) {
      if (!take(&unobserved)) {
        printUsage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a);
      printUsage(argv[0]);
      return 2;
    }
  }

  if (cake.empty()) {
    printUsage(argv[0]);
    return 2;
  }
  if (out.empty()) {
    out = std::filesystem::path(cake).replace_extension(".elev.yaml").string();
  }

  hound_mapping::ElevationCostConfig ec;
  ec.resolution_m = resolution;
  ec.robot_height_m = robot_height;
  ec.bev_occ_tsdf_max_m = bev_occ;
  ec.unobserved_luminance = unobserved;
  ec.do_inpaint = false;

  hound_mapping::LocalMaps maps;
  std::string err;
  if (!hound_mapping::extractLayerCakeBev(cake, voxel_size, ec, &maps, &err)) {
    std::fprintf(stderr, "layercake_to_bev: %s\n", err.c_str());
    return 1;
  }
  if (!writeLocalMapYaml(maps, out, &err)) {
    std::fprintf(stderr, "layercake_to_bev: %s\n", err.c_str());
    return 1;
  }
  std::printf(
      "wrote %s (%dx%d @ %.3fm origin=(%.2f,%.2f))\n", out.c_str(), maps.width,
      maps.height, maps.resolution, maps.origin_x, maps.origin_y);
  return 0;
}
