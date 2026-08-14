#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <nvblox/nvblox.h>

#include "hound_mapping/mapper_core.hpp"

namespace hound_mapping {
namespace {

class LayercakeToMeshNode : public rclcpp::Node
{
public:
  LayercakeToMeshNode()
  : Node("layercake_to_mesh")
  {
    declare_parameter(
        "layer_cake_path", "/root/colcon_ws/maps/hound_tsdf.layercake");
    declare_parameter(
        "mesh_ply_path", "/root/colcon_ws/maps/hound_tsdf_mesh.ply");
    declare_parameter("oneshot", true);
    declare_parameter("voxel_size", 0.05);

    layer_cake_path_ = get_parameter("layer_cake_path").as_string();
    mesh_ply_path_ = get_parameter("mesh_ply_path").as_string();
    oneshot_ = get_parameter("oneshot").as_bool();
    voxel_size_ = static_cast<float>(get_parameter("voxel_size").as_double());

    export_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/export_mesh",
        [this](
            const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          std::string msg;
          const bool ok = exportOnce(&msg);
          res->success = ok;
          res->message = msg;
        });

    if (oneshot_) {
      std::string msg;
      const bool ok = exportOnce(&msg);
      if (ok) {
        RCLCPP_INFO(get_logger(), "%s", msg.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "%s", msg.c_str());
      }
      // Defer shutdown so constructor can finish cleanly.
      shutdown_timer_ = create_wall_timer(
          std::chrono::milliseconds(50), [this, ok]() {
            shutdown_timer_->cancel();
            rclcpp::shutdown();
            (void)ok;
          });
    } else {
      RCLCPP_INFO(
          get_logger(),
          "Ready. Call ~/export_mesh (layer_cake=%s → ply=%s)",
          layer_cake_path_.c_str(), mesh_ply_path_.c_str());
    }
  }

private:
  bool exportOnce(std::string* message)
  {
    if (layer_cake_path_.empty() || mesh_ply_path_.empty()) {
      if (message) {
        *message = "layer_cake_path or mesh_ply_path empty";
      }
      return false;
    }
    if (!std::filesystem::exists(layer_cake_path_)) {
      if (message) {
        *message = "layercake not found: " + layer_cake_path_;
      }
      return false;
    }

    try {
      const auto parent =
          std::filesystem::path(mesh_ply_path_).parent_path();
      if (!parent.empty()) {
        std::filesystem::create_directories(parent);
      }
    } catch (const std::exception& ex) {
      if (message) {
        *message = std::string("mkdir failed: ") + ex.what();
      }
      return false;
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    MapperConfig mc;
    mc.voxel_size_m = voxel_size_;
    MapperCore core(mc);
    if (!core.loadLayerCake(layer_cake_path_)) {
      if (message) {
        *message = "loadLayerCake failed: " + layer_cake_path_;
      }
      return false;
    }
    const size_t tsdf_blocks = core.mapper().tsdf_layer().numBlocks();
    if (!core.exportColorMeshPly(mesh_ply_path_)) {
      if (message) {
        *message = "exportColorMeshPly failed: " + mesh_ply_path_;
      }
      return false;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::uintmax_t bytes = 0;
    try {
      if (std::filesystem::exists(mesh_ply_path_)) {
        bytes = std::filesystem::file_size(mesh_ply_path_);
      }
    } catch (...) {
    }
    if (message) {
      char buf[512];
      std::snprintf(
          buf, sizeof(buf),
          "OK %.1fms → %s (tsdf_blocks=%zu file=%.2f MiB)", ms,
          mesh_ply_path_.c_str(), tsdf_blocks,
          static_cast<double>(bytes) / (1024.0 * 1024.0));
      *message = buf;
    }
    return true;
  }

  std::string layer_cake_path_;
  std::string mesh_ply_path_;
  bool oneshot_{true};
  float voxel_size_{0.05f};
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr export_srv_;
  rclcpp::TimerBase::SharedPtr shutdown_timer_;
};

}  // namespace
}  // namespace hound_mapping

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hound_mapping::LayercakeToMeshNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
