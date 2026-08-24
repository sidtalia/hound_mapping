#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <nvblox/nvblox.h>
#include <nvblox/sensors/pointcloud_to_depth_conversion.h>
#include <glog/logging.h>

#include "hound_mapping/elevation_cost.hpp"
#include "hound_mapping/mapper_core.hpp"
#include "hound_mapping/msg/local_map.hpp"
#include "hound_mapping/prior_bev_gpu.hpp"

namespace hound_mapping {
namespace {

bool rateOk(rclcpp::Time* last, double hz, const rclcpp::Time& now)
{
  if (hz <= 0.0) {
    return false;
  }
  if (last->nanoseconds() == 0) {
    *last = now;
    return true;
  }
  const double min_dt = 1.0 / hz;
  if ((now - *last).seconds() + 1e-9 < min_dt) {
    return false;
  }
  *last = now;
  return true;
}

nvblox::Transform toNvblox(const Eigen::Isometry3d& T)
{
  nvblox::Transform out;
  out.matrix() = T.matrix().cast<float>();
  return out;
}

std::optional<nvblox::Camera> cameraFromInfo(
    const sensor_msgs::msg::CameraInfo& info)
{
  if (info.width == 0 || info.height == 0) {
    return std::nullopt;
  }
  const float fu = static_cast<float>(info.k[0]);
  const float fv = static_cast<float>(info.k[4]);
  const float cu = static_cast<float>(info.k[2]);
  const float cv = static_cast<float>(info.k[5]);
  return nvblox::Camera(fu, fv, cu, cv, static_cast<int>(info.width),
                        static_cast<int>(info.height));
}

std::string expandNameTemplate(std::string tmpl, const std::string& name)
{
  const std::string token = "{name}";
  auto pos = tmpl.find(token);
  if (pos != std::string::npos) {
    tmpl.replace(pos, token.size(), name);
  }
  return tmpl;
}

bool depthToNvblox(
    const sensor_msgs::msg::Image& msg, nvblox::DepthImage* out)
{
  const int rows = static_cast<int>(msg.height);
  const int cols = static_cast<int>(msg.width);
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  *out = nvblox::DepthImage(rows, cols, nvblox::MemoryType::kHost);
  if (msg.encoding == "32FC1") {
    for (int r = 0; r < rows; ++r) {
      const float* src = reinterpret_cast<const float*>(
          msg.data.data() + r * msg.step);
      for (int c = 0; c < cols; ++c) {
        (*out)(r, c) = src[c];
      }
    }
    return true;
  }
  if (msg.encoding == "16UC1") {
    for (int r = 0; r < rows; ++r) {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(
          msg.data.data() + r * msg.step);
      for (int c = 0; c < cols; ++c) {
        const uint16_t mm = src[c];
        (*out)(r, c) = (mm == 0) ? 0.f : static_cast<float>(mm) * 0.001f;
      }
    }
    return true;
  }
  return false;
}

/// Zero depth in the bottom ``fraction`` of rows (optical: row0 = top).
/// Invalid depth (0) is skipped by projective integrate.
void zeroDepthBottomFraction(nvblox::DepthImage* depth, double fraction)
{
  if (!depth || fraction <= 0.0) {
    return;
  }
  fraction = std::clamp(fraction, 0.0, 1.0);
  const int rows = depth->rows();
  const int cols = depth->cols();
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const int r0 = static_cast<int>(std::lround((1.0 - fraction) * rows));
  for (int r = std::max(0, r0); r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      (*depth)(r, c) = 0.f;
    }
  }
}

bool colorToNvblox(
    const sensor_msgs::msg::Image& msg, nvblox::ColorImage* out)
{
  const int rows = static_cast<int>(msg.height);
  const int cols = static_cast<int>(msg.width);
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  *out = nvblox::ColorImage(rows, cols, nvblox::MemoryType::kHost);
  const bool bgr = (msg.encoding == "bgr8" || msg.encoding == "bgra8");
  const bool rgb = (msg.encoding == "rgb8" || msg.encoding == "rgba8");
  if (!bgr && !rgb) {
    return false;
  }
  const int channels = (msg.encoding == "rgba8" || msg.encoding == "bgra8") ? 4 : 3;
  for (int r = 0; r < rows; ++r) {
    const uint8_t* src = msg.data.data() + r * msg.step;
    for (int c = 0; c < cols; ++c) {
      const uint8_t* p = src + c * channels;
      if (rgb) {
        (*out)(r, c) = nvblox::Color(p[0], p[1], p[2]);
      } else {
        (*out)(r, c) = nvblox::Color(p[2], p[1], p[0]);
      }
    }
  }
  return true;
}

/// All-white RGB = fully traversable (cost stores L=255).
bool whiteColorImage(
    int rows, int cols, nvblox::ColorImage* out)
{
  if (rows <= 0 || cols <= 0 || out == nullptr) {
    return false;
  }
  *out = nvblox::ColorImage(rows, cols, nvblox::MemoryType::kHost);
  const nvblox::Color white(255, 255, 255);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      (*out)(r, c) = white;
    }
  }
  return true;
}

bool maskToNvblox(
    const sensor_msgs::msg::Image& msg, nvblox::MonoImage* out)
{
  const int rows = static_cast<int>(msg.height);
  const int cols = static_cast<int>(msg.width);
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  *out = nvblox::MonoImage(rows, cols, nvblox::MemoryType::kHost);
  if (msg.encoding == "mono8" || msg.encoding == "8UC1") {
    for (int r = 0; r < rows; ++r) {
      const uint8_t* src = msg.data.data() + r * msg.step;
      for (int c = 0; c < cols; ++c) {
        (*out)(r, c) = src[c];
      }
    }
    return true;
  }
  return false;
}

bool cloudToNvblox(
    const sensor_msgs::msg::PointCloud2& msg, nvblox::Pointcloud* out,
    std::shared_ptr<nvblox::CudaStream> stream,
    nvblox::Time* max_time_ms = nullptr)
{
  bool has_time = false;
  for (const auto& f : msg.fields) {
    if (f.name == "time" || f.name == "t" || f.name == "timestamp") {
      has_time = true;
      break;
    }
  }

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
  std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iter_t;
  if (has_time) {
    // Prefer "time" (composite/Unitree seconds relative to stamp).
    const char* field = "time";
    for (const auto& f : msg.fields) {
      if (f.name == "time") {
        field = "time";
        break;
      }
      if (f.name == "t") {
        field = "t";
      } else if (f.name == "timestamp" && std::string(field) != "t") {
        field = "timestamp";
      }
    }
    iter_t = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(
        msg, field);
  }

  std::vector<nvblox::Vector3f> pts;
  std::vector<nvblox::Time> times_ms;
  const size_t n_est = static_cast<size_t>(msg.width) * msg.height;
  pts.reserve(n_est);
  if (has_time) {
    times_ms.reserve(n_est);
  }
  int64_t max_ms = 0;
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    const float x = *iter_x;
    const float y = *iter_y;
    const float z = *iter_z;
    float t_s = 0.f;
    if (iter_t) {
      t_s = **iter_t;
      ++(*iter_t);
    }
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    pts.emplace_back(x, y, z);
    if (has_time) {
      // Float "time" from our composite is seconds relative to header.
      // Values >> 10 are treated as already-nanoseconds (nvblox_ros style).
      int64_t ms = 0;
      if (std::isfinite(t_s)) {
        if (std::fabs(t_s) > 10.f) {
          ms = static_cast<int64_t>(t_s * 1e-6);  // ns → ms
        } else {
          ms = static_cast<int64_t>(std::llround(t_s * 1000.0));
        }
        if (ms < 0) {
          ms = 0;
        }
      }
      times_ms.emplace_back(ms);
      if (ms > max_ms) {
        max_ms = ms;
      }
    }
  }
  if (pts.empty()) {
    return false;
  }
  *out = nvblox::Pointcloud(nvblox::MemoryType::kDevice);
  out->copyPointsFromAsync(pts, *stream);
  if (has_time && times_ms.size() == pts.size()) {
    out->copyTimestampsFromAsync(times_ms, *stream);
    if (max_time_ms) {
      *max_time_ms = nvblox::Time(max_ms);
    }
  } else if (max_time_ms) {
    *max_time_ms = nvblox::Time(0);
  }
  stream->synchronize();
  return true;
}

sensor_msgs::msg::Image matToImageMsg(
    const cv::Mat& mat, const std_msgs::msg::Header& header,
    const std::string& encoding)
{
  const cv::Mat cont = mat.isContinuous() ? mat : mat.clone();
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.height = static_cast<uint32_t>(cont.rows);
  msg.width = static_cast<uint32_t>(cont.cols);
  msg.encoding = encoding;
  msg.is_bigendian = false;
  msg.step = static_cast<uint32_t>(cont.step);
  msg.data.assign(cont.datastart, cont.dataend);
  return msg;
}

}  // namespace

/// Single worker thread drains latest sensor slots at mapper_rate_hz:
/// integrate → BEV on live TSDF/color → inpaint/slope → LocalMap (elev+cost).
/// ROS callbacks only overwrite latest message pointers (no mapper access).
class MappingNode : public rclcpp::Node {
 public:
  MappingNode()
  : Node("hound_mapping")
  {
    FLAGS_minloglevel = google::GLOG_WARNING;

    declareParams();
    loadParams();

    MapperConfig mc;
    mc.voxel_size_m = voxel_size_;
    mc.max_integration_distance_m = max_integration_distance_m_;
    mc.lidar_max_integration_distance_m = lidar_max_integration_distance_m_;
    mc.map_clearing_radius_m = map_clearing_radius_m_;
    mapper_ = std::make_unique<MapperCore>(mc);

    ElevationCostConfig ec;
    ec.resolution_m = elevation_resolution_m_ > 0.f
        ? elevation_resolution_m_
        : voxel_size_;
    // LocalMap full side length (m); ElevationCost uses half-extent internally.
    ec.window_radius_m = local_map_size_m_ > 0.f
        ? local_map_size_m_ * 0.5f
        : map_clearing_radius_m_;
    ec.unobserved_luminance = unobserved_luminance_;
    ec.color_weight_min = color_weight_min_;
    ec.color_z_search_half_band_m = color_z_search_half_band_m_;
    ec.color_z_column_above_m = color_z_column_above_m_;
    ec.bev_occ_tsdf_max_m = bev_occ_tsdf_max_m_;
    ec.robot_height_m = robot_height_m_;
    ec.elev_z_min_rel = elev_z_min_rel_;
    ec.elev_z_max_rel = elev_z_max_rel_;
    ec.lethal_slope_deg = lethal_slope_deg_;
    ec.do_inpaint = do_inpaint_;
    ec.inpaint_resolution_m = inpaint_resolution_m_;
    ec.inpaint_radius = inpaint_radius_;
    ec.elev_median_ksize = elev_median_ksize_;
    extractor_ = std::make_unique<ElevationCostExtractor>(ec);
    // Separate CUDA workspace for loop2 (inpaint/slope) so it never races loop1 BEV.
    finish_extractor_ = std::make_unique<ElevationCostExtractor>(ec);
    prior_bev_ = std::make_unique<PriorBevGpu>();
    maybeLoadPriorBev(ec);

    lidar_ = nvblox::Lidar(
        lidar_width_, lidar_height_, lidar_min_valid_range_m_,
        lidar_vertical_fov_rad_);
    if (lidar_elevation_below_deg_ > 0.f && lidar_elevation_above_deg_ > 0.f) {
      constexpr float kDegToRad = static_cast<float>(M_PI / 180.0);
      const float below_rad = lidar_elevation_below_deg_ * kDegToRad;
      const float above_rad = lidar_elevation_above_deg_ * kDegToRad;
      lidar_ = nvblox::Lidar(
          lidar_width_, lidar_height_, lidar_min_valid_range_m_, below_rad,
          above_rad);
      RCLCPP_INFO(
          get_logger(),
          "nvblox Lidar FoV asymmetric: -%.1f° … +%.1f° (%dx%d)",
          lidar_elevation_below_deg_, lidar_elevation_above_deg_,
          lidar_width_, lidar_height_);
    } else {
      RCLCPP_INFO(
          get_logger(),
          "nvblox Lidar FoV symmetric: vertical_fov=%.3f rad (%dx%d)",
          lidar_vertical_fov_rad_, lidar_width_, lidar_height_);
    }
    lidar_stream_ = std::make_shared<nvblox::CudaStreamOwning>();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Latest-only, best-effort: extra readers (RViz, ros2 topic hz) must not
    // stall the mapper with reliable backpressure.
    local_map_pub_ =
        create_publisher<hound_mapping::msg::LocalMap>(
            "~/local_map", rclcpp::QoS(1).best_effort());
    tsdf_color_cloud_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("~/tsdf_color_cloud", 1);
    tsdf_color_mesh_pub_ =
        create_publisher<visualization_msgs::msg::Marker>("~/tsdf_color_mesh", 1);
    lidar_depth_pub_ =
        create_publisher<sensor_msgs::msg::Image>("~/lidar_depth_image", 1);
    lidar_depth_viz_pub_ =
        create_publisher<sensor_msgs::msg::Image>("~/lidar_depth_viz", 1);
    lidar_depth_cloud_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("~/lidar_depth_cloud", 1);
    lidar_tsdf_probe_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("~/lidar_tsdf_probe", 1);
    tsdf_voxels_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("~/tsdf_voxels", 1);
    extract_timing_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "~/extract_timing_ms", 10);

    auto sensor_qos = rclcpp::SensorDataQoS();

    auto bind_cam = [this, sensor_qos](const std::string& name,
                                        const std::string& depth_topic,
                                        const std::string& depth_info,
                                        const std::string& color_topic,
                                        const std::string& color_info) {
      auto slot = std::make_shared<CamSlot>();
      slot->name = name;
      if (use_depth_) {
        slot->depth_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
            depth_info, sensor_qos,
            [slot](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
              std::lock_guard<std::mutex> lock(slot->mu);
              slot->depth_info = *msg;
              slot->depth_camera = cameraFromInfo(*msg);
            });
        slot->depth_sub = create_subscription<sensor_msgs::msg::Image>(
            depth_topic, sensor_qos,
            [this, slot](sensor_msgs::msg::Image::ConstSharedPtr msg) {
              {
                std::lock_guard<std::mutex> lock(slot->mu);
                slot->pending_depth = msg;
              }
              requestDrainSoon();
            });
      }
      if (use_color_) {
        slot->color_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
            color_info, sensor_qos,
            [slot](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
              std::lock_guard<std::mutex> lock(slot->mu);
              slot->color_info = *msg;
              slot->color_camera = cameraFromInfo(*msg);
            });
        slot->color_sub = create_subscription<sensor_msgs::msg::Image>(
            color_topic, sensor_qos,
            [this, slot](sensor_msgs::msg::Image::ConstSharedPtr msg) {
              RCLCPP_INFO_ONCE(
                  get_logger(),
                  "first color on %s (%s)",
                  slot->name.c_str(),
                  msg->header.frame_id.c_str());
              {
                std::lock_guard<std::mutex> lock(slot->mu);
                slot->pending_color = msg;
              }
              requestDrainSoon();
            });
      }
      cam_slots_.push_back(slot);
      RCLCPP_INFO(
          get_logger(), "camera[%s] depth=%s color=%s", name.c_str(),
          depth_topic.c_str(), color_topic.c_str());
    };

    if (!camera_names_.empty()) {
      for (const auto& name : camera_names_) {
        std::string depth_topic = depth_topic_template_;
        std::string depth_info = depth_info_topic_template_;
        if (depth_topic.empty()) {
          depth_topic = "/" + name + "/depth/image_rect_raw";
        } else {
          depth_topic = expandNameTemplate(std::move(depth_topic), name);
        }
        if (depth_info.empty()) {
          depth_info = "/" + name + "/depth/camera_info";
        } else {
          depth_info = expandNameTemplate(std::move(depth_info), name);
        }
        std::string color_topic = color_topic_template_;
        if (color_topic.empty()) {
          color_topic = "/" + name + "/color/image_raw";
        } else {
          color_topic = expandNameTemplate(std::move(color_topic), name);
        }
        std::string color_info = color_info_topic_template_;
        if (color_info.empty()) {
          color_info = "/" + name + "/color/camera_info";
        } else {
          color_info = expandNameTemplate(std::move(color_info), name);
        }
        bind_cam(name, depth_topic, depth_info, color_topic, color_info);
      }
    } else {
      bind_cam(
          "camera", depth_topic_, depth_info_topic_, color_topic_,
          color_info_topic_);
    }
    if (use_lidar_) {
      lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          lidar_topic_, sensor_qos,
          [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
            RCLCPP_INFO_ONCE(
                get_logger(),
                "first lidar on %s frame=%s",
                lidar_topic_.c_str(),
                msg->header.frame_id.c_str());
            {
              std::lock_guard<std::mutex> lock(inbox_mutex_);
              pending_lidar_ = msg;
            }
            requestDrainSoon();
          });
    }
    if (use_people_mask_) {
      mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
          people_mask_topic_, sensor_qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            std::lock_guard<std::mutex> lock(inbox_mutex_);
            pending_mask_ = msg;
          });
    }

    map_clear_sub_ = create_subscription<std_msgs::msg::Empty>(
        map_clear_topic_, 1,
        [this](const std_msgs::msg::Empty::SharedPtr) {
          {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            clear_requested_ = true;
            drain_soon_ = true;
          }
          wake_cv_.notify_one();
        });

    save_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/save_layer_cake",
        [this](
            const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          if (layer_cake_path_.empty()) {
            res->success = false;
            res->message = "layer_cake_path is empty";
            return;
          }
          {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            save_requested_ = true;
            drain_soon_ = true;
          }
          wake_cv_.notify_one();
          res->success = true;
          res->message =
              "queued save on mapper worker → " + layer_cake_path_;
        });

    worker_running_ = true;
    publish_running_ = true;
    worker_thread_ = std::thread([this]() { workerLoop(); });
    publish_thread_ = std::thread([this]() { publishLoop(); });

    RCLCPP_INFO(
        get_logger(),
        "hound_mapping ready: depth=%d color=%d lidar=%d voxel=%.3f "
        "local_map=%.1fm clear_r=%.1fm loop1=integrate+BEV on new data "
        "loop2=inpaint/publish extract_on_depth=%d "
        "debug_semantic_all_traversable=%d "
        "save_layer_cake=~/save_layer_cake path=%s "
        "map_clear=%s "
        "tsdf_color_mesh=%d@%.1fHz tsdf_voxels=%d@%.1fHz "
        "(~/local_map + ~/tsdf_color_cloud|mesh|~/tsdf_voxels)",
        use_depth_, use_color_, use_lidar_, voxel_size_,
        local_map_size_m_ > 0.f ? local_map_size_m_
                                : (2.f * map_clearing_radius_m_),
        map_clearing_radius_m_, extract_on_depth_ ? 1 : 0,
        debug_semantic_all_traversable_ ? 1 : 0,
        layer_cake_path_.empty() ? "(unset)" : layer_cake_path_.c_str(),
        map_clear_topic_.c_str(),
        publish_tsdf_color_mesh_ ? 1 : 0, tsdf_color_mesh_rate_hz_,
        publish_tsdf_voxels_ ? 1 : 0, tsdf_voxel_rate_hz_);
  }

  ~MappingNode() override
  {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      worker_running_ = false;
      drain_soon_ = true;
    }
    wake_cv_.notify_one();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    {
      std::lock_guard<std::mutex> lock(pub_mutex_);
      publish_running_ = false;
      pending_pub_.reset();
    }
    pub_cv_.notify_one();
    if (publish_thread_.joinable()) {
      publish_thread_.join();
    }
  }

 private:
  struct CamSlot {
    std::string name;
    std::mutex mu;
    std::optional<nvblox::Camera> depth_camera;
    std::optional<nvblox::Camera> color_camera;
    sensor_msgs::msg::CameraInfo depth_info;
    sensor_msgs::msg::CameraInfo color_info;
    sensor_msgs::msg::Image::ConstSharedPtr pending_depth;
    sensor_msgs::msg::Image::ConstSharedPtr pending_color;
    int64_t last_depth_ns{-1};
    int64_t last_color_ns{-1};
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_sub;
  };

  /** Contiguous BEV buffers + metadata for loop2 (inpaint/publish). */
  struct PublishJob {
    LocalMaps maps;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    float clearing_z{0.f};
    double integrate_ms{0.0};
    double bev_ms{0.0};
    double handoff_ms{0.0};
    double loop1_ms{0.0};
  };

  void declareParams()
  {
    declare_parameter("global_frame", "odom");
    declare_parameter("map_clearing_frame_id", "camera_link");
    declare_parameter("voxel_size", 0.05);
    declare_parameter("elevation_resolution", 0.05);
    declare_parameter("map_clearing_radius_m", 32.0);
    // Empty TSDF (worker thread). Same pattern as FCU ~/ekf_reset.
    declare_parameter("map_clear_topic", "~/map_clear");
    // Full LocalMap side length (m). <=0 → 2 * map_clearing_radius_m.
    declare_parameter("local_map_size_m", 0.0);
    declare_parameter("max_integration_distance_m", 15.0);
    declare_parameter("lidar_max_integration_distance_m", 15.0);
    declare_parameter("lethal_slope_deg", 60.0);
    declare_parameter("do_inpaint", true);
    declare_parameter("inpaint_resolution_m", 0.25);
    declare_parameter("inpaint_radius", 3);
    declare_parameter("elev_median_ksize", 5);
    declare_parameter("unobserved_luminance", 128.0);
    declare_parameter("color_weight_min", 0.001);
    // 0 = auto (2 × voxel_size). Vertical band when sampling ColorVoxel at elev.
    declare_parameter("color_z_search_half_band_m", 0.0);
    // 0 → full BEV Z window above elev (min luminance in column).
    declare_parameter("color_z_column_above_m", 0.0);
    // BEV: distance <= this counts as occupied (0 = classic TSDF ≤0).
    declare_parameter("bev_occ_tsdf_max_m", 0.0);
    declare_parameter("robot_height_m", 0.4);
    declare_parameter("elev_z_min_rel", -1.0);
    declare_parameter("elev_z_max_rel", 1.5);

    declare_parameter("use_depth", true);
    declare_parameter("use_color", true);
    // Debug: paint TSDF color all-white (semantic free in FOV).
    // Unobserved still uses unobserved_luminance (grey outside FOV).
    declare_parameter("debug_semantic_all_traversable", false);
    declare_parameter("use_lidar", true);
    declare_parameter("use_people_mask", false);
    // Drop bottom of depth image before integrate (0 = keep all, 0.5 = bottom half).
    declare_parameter("depth_ignore_bottom_fraction", 0.0);

    declare_parameter("depth_topic", "/camera/depth/image_rect_raw");
    declare_parameter("depth_info_topic", "/camera/depth/camera_info");
    declare_parameter("color_topic", "/camera/color/image_raw");
    declare_parameter("color_info_topic", "/camera/color/camera_info");
    declare_parameter("lidar_topic", "/unilidar/cloud");
    declare_parameter("people_mask_topic", "/segmentation/coarse_people_mask");
    declare_parameter("camera_names", std::vector<std::string>{});
    declare_parameter("depth_topic_template", "");
    declare_parameter("depth_info_topic_template", "");
    declare_parameter("color_topic_template", "");
    declare_parameter("color_info_topic_template", "");

    declare_parameter("lidar_width", 360);
    declare_parameter("lidar_height", 90);
    declare_parameter("lidar_vertical_fov_rad", 2.094395);
    // Asymmetric elevation (deg, positive). Mid-360: below=7, above=52.
    // When both > 0, used instead of symmetric lidar_vertical_fov_rad.
    declare_parameter("lidar_elevation_below_deg", 0.0);
    declare_parameter("lidar_elevation_above_deg", 0.0);
    declare_parameter("lidar_min_valid_range_m", 0.3);
    declare_parameter("use_lidar_motion_compensation", false);

    // Legacy: worker is event-driven on new depth/color/lidar (latest-only).
    declare_parameter("mapper_rate_hz", 20.0);
    // Legacy alias: if mapper_rate_hz unset in older launch, still honored via
    // loadParams fallback when publish_map_rate_hz is the only rate set.
    declare_parameter("publish_map_rate_hz", 20.0);
    declare_parameter("integrate_depth_rate_hz", 20.0);
    declare_parameter("integrate_color_rate_hz", 20.0);
    declare_parameter("integrate_lidar_rate_hz", 10.0);
    declare_parameter("map_clear_rate_hz", 20.0);
    declare_parameter("log_extract_timing", false);
    // Debug: publish colored TSDF isosurface (mesh vertices / triangles).
    declare_parameter("publish_tsdf_color_mesh", false);
    declare_parameter("tsdf_color_mesh_rate_hz", 1.0);
    declare_parameter("tsdf_color_mesh_max_triangles", 500000);
    // Near-surface TSDF voxels as PointCloud2 (xyz+rgb from ColorLayer).
    declare_parameter("publish_tsdf_voxels", false);
    declare_parameter("tsdf_voxel_rate_hz", 1.0);
    declare_parameter("tsdf_voxel_min_weight", 0.1);
    // 0 → 2×voxel_size. Only |distance| ≤ this are emitted (surface shell).
    declare_parameter("tsdf_voxel_max_distance_m", 0.0);
    declare_parameter("tsdf_voxel_max_points", 400000);
    // Debug: publish nvblox spherical depth image from lidar pointcloud.
    declare_parameter("publish_lidar_depth_image", false);
    declare_parameter("publish_lidar_tsdf_probe", false);
    declare_parameter("lidar_tsdf_probe_max_points", 4000);
    // When true: wake worker ASAP on each new depth (still single-threaded).
    declare_parameter("extract_on_depth", false);
    declare_parameter("dump_elev_dir", "");
    declare_parameter("dump_elev_rate_hz", 20.0);
    declare_parameter("dump_elev_n_frames", 0);
    // nvblox SQLite LayerCake path. Trigger via ~/save_layer_cake (runs on
    // the mapper worker — blocks that tick; logs wall time + block counts).
    declare_parameter(
        "layer_cake_path", "/root/colcon_ws/maps/hound_tsdf.layercake");
    // Boot-load LayerCake → full BEV on GPU; Finalize fills live holes.
    declare_parameter("prior_layer_cake_path", "");
    declare_parameter(
        "prior_xyz_yaw", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    declare_parameter("prior_fill_enabled", true);
    // TF: bag replay often has sensor stamps a few ms ahead of /tf.
    declare_parameter("tf_lookup_timeout_s", 0.05);
    declare_parameter("tf_fallback_to_latest", true);
  }

  void loadParams()
  {
    global_frame_ = get_parameter("global_frame").as_string();
    map_clearing_frame_ = get_parameter("map_clearing_frame_id").as_string();
    voxel_size_ = static_cast<float>(get_parameter("voxel_size").as_double());
    elevation_resolution_m_ =
        static_cast<float>(get_parameter("elevation_resolution").as_double());
    map_clearing_radius_m_ = static_cast<float>(
        get_parameter("map_clearing_radius_m").as_double());
    map_clear_topic_ = get_parameter("map_clear_topic").as_string();
    local_map_size_m_ = static_cast<float>(
        get_parameter("local_map_size_m").as_double());
    max_integration_distance_m_ = static_cast<float>(
        get_parameter("max_integration_distance_m").as_double());
    lidar_max_integration_distance_m_ = static_cast<float>(
        get_parameter("lidar_max_integration_distance_m").as_double());
    lethal_slope_deg_ =
        static_cast<float>(get_parameter("lethal_slope_deg").as_double());
    do_inpaint_ = get_parameter("do_inpaint").as_bool();
    inpaint_resolution_m_ = static_cast<float>(
        get_parameter("inpaint_resolution_m").as_double());
    inpaint_radius_ = static_cast<int>(get_parameter("inpaint_radius").as_int());
    elev_median_ksize_ =
        static_cast<int>(get_parameter("elev_median_ksize").as_int());
    unobserved_luminance_ = static_cast<float>(
        get_parameter("unobserved_luminance").as_double());
    color_weight_min_ = static_cast<float>(
        get_parameter("color_weight_min").as_double());
    color_z_search_half_band_m_ = static_cast<float>(
        get_parameter("color_z_search_half_band_m").as_double());
    color_z_column_above_m_ = static_cast<float>(
        get_parameter("color_z_column_above_m").as_double());
    bev_occ_tsdf_max_m_ = static_cast<float>(
        get_parameter("bev_occ_tsdf_max_m").as_double());
    robot_height_m_ = static_cast<float>(
        get_parameter("robot_height_m").as_double());
    elev_z_min_rel_ = static_cast<float>(
        get_parameter("elev_z_min_rel").as_double());
    elev_z_max_rel_ = static_cast<float>(
        get_parameter("elev_z_max_rel").as_double());

    use_depth_ = get_parameter("use_depth").as_bool();
    use_color_ = get_parameter("use_color").as_bool();
    debug_semantic_all_traversable_ =
        get_parameter("debug_semantic_all_traversable").as_bool();
    use_lidar_ = get_parameter("use_lidar").as_bool();
    use_people_mask_ = get_parameter("use_people_mask").as_bool();
    depth_ignore_bottom_fraction_ =
        get_parameter("depth_ignore_bottom_fraction").as_double();
    depth_ignore_bottom_fraction_ =
        std::clamp(depth_ignore_bottom_fraction_, 0.0, 1.0);

    depth_topic_ = get_parameter("depth_topic").as_string();
    depth_info_topic_ = get_parameter("depth_info_topic").as_string();
    color_topic_ = get_parameter("color_topic").as_string();
    color_info_topic_ = get_parameter("color_info_topic").as_string();
    lidar_topic_ = get_parameter("lidar_topic").as_string();
    people_mask_topic_ = get_parameter("people_mask_topic").as_string();
    camera_names_ = get_parameter("camera_names").as_string_array();
    depth_topic_template_ = get_parameter("depth_topic_template").as_string();
    depth_info_topic_template_ =
        get_parameter("depth_info_topic_template").as_string();
    color_topic_template_ = get_parameter("color_topic_template").as_string();
    color_info_topic_template_ =
        get_parameter("color_info_topic_template").as_string();

    lidar_width_ = get_parameter("lidar_width").as_int();
    lidar_height_ = get_parameter("lidar_height").as_int();
    lidar_vertical_fov_rad_ = static_cast<float>(
        get_parameter("lidar_vertical_fov_rad").as_double());
    lidar_elevation_below_deg_ = static_cast<float>(
        get_parameter("lidar_elevation_below_deg").as_double());
    lidar_elevation_above_deg_ = static_cast<float>(
        get_parameter("lidar_elevation_above_deg").as_double());
    lidar_min_valid_range_m_ = static_cast<float>(
        get_parameter("lidar_min_valid_range_m").as_double());
    use_lidar_motion_compensation_ =
        get_parameter("use_lidar_motion_compensation").as_bool();

    mapper_rate_hz_ = get_parameter("mapper_rate_hz").as_double();
    publish_map_rate_hz_ = get_parameter("publish_map_rate_hz").as_double();
    // Older configs only set publish_map_rate_hz; treat it as the drain rate.
    if (mapper_rate_hz_ <= 0.0) {
      mapper_rate_hz_ = publish_map_rate_hz_ > 0.0 ? publish_map_rate_hz_ : 20.0;
    }

    integrate_depth_rate_hz_ =
        get_parameter("integrate_depth_rate_hz").as_double();
    integrate_color_rate_hz_ =
        get_parameter("integrate_color_rate_hz").as_double();
    integrate_lidar_rate_hz_ =
        get_parameter("integrate_lidar_rate_hz").as_double();
    (void)integrate_depth_rate_hz_;
    (void)integrate_color_rate_hz_;
    (void)integrate_lidar_rate_hz_;
    map_clear_rate_hz_ = get_parameter("map_clear_rate_hz").as_double();
    log_extract_timing_ = get_parameter("log_extract_timing").as_bool();
    publish_tsdf_color_mesh_ =
        get_parameter("publish_tsdf_color_mesh").as_bool();
    tsdf_color_mesh_rate_hz_ =
        get_parameter("tsdf_color_mesh_rate_hz").as_double();
    tsdf_color_mesh_max_triangles_ =
        get_parameter("tsdf_color_mesh_max_triangles").as_int();
    publish_tsdf_voxels_ = get_parameter("publish_tsdf_voxels").as_bool();
    tsdf_voxel_rate_hz_ = get_parameter("tsdf_voxel_rate_hz").as_double();
    tsdf_voxel_min_weight_ =
        static_cast<float>(get_parameter("tsdf_voxel_min_weight").as_double());
    tsdf_voxel_max_distance_m_ =
        get_parameter("tsdf_voxel_max_distance_m").as_double();
    tsdf_voxel_max_points_ =
        get_parameter("tsdf_voxel_max_points").as_int();
    publish_lidar_depth_image_ =
        get_parameter("publish_lidar_depth_image").as_bool();
    publish_lidar_tsdf_probe_ =
        get_parameter("publish_lidar_tsdf_probe").as_bool();
    lidar_tsdf_probe_max_points_ =
        get_parameter("lidar_tsdf_probe_max_points").as_int();
    extract_on_depth_ = get_parameter("extract_on_depth").as_bool();
    dump_elev_dir_ = get_parameter("dump_elev_dir").as_string();
    dump_elev_rate_hz_ = get_parameter("dump_elev_rate_hz").as_double();
    dump_elev_n_frames_ = get_parameter("dump_elev_n_frames").as_int();
    layer_cake_path_ = get_parameter("layer_cake_path").as_string();
    prior_layer_cake_path_ =
        get_parameter("prior_layer_cake_path").as_string();
    prior_xyz_yaw_ = get_parameter("prior_xyz_yaw").as_double_array();
    if (prior_xyz_yaw_.size() < 4) {
      prior_xyz_yaw_ = {0.0, 0.0, 0.0, 0.0};
    }
    prior_fill_enabled_ = get_parameter("prior_fill_enabled").as_bool();
    tf_lookup_timeout_s_ = get_parameter("tf_lookup_timeout_s").as_double();
    tf_fallback_to_latest_ = get_parameter("tf_fallback_to_latest").as_bool();
    if (!dump_elev_dir_.empty()) {
      RCLCPP_INFO(
          get_logger(), "dump_elev_dir=%s rate=%.3fHz n_frames=%d",
          dump_elev_dir_.c_str(), dump_elev_rate_hz_, dump_elev_n_frames_);
    }
  }

  void maybeLoadPriorBev(const ElevationCostConfig& ec)
  {
    if (!prior_fill_enabled_ || prior_layer_cake_path_.empty()) {
      extractor_->setPrior(nullptr);
      return;
    }
    if (!std::filesystem::exists(prior_layer_cake_path_)) {
      RCLCPP_ERROR(
          get_logger(),
          "prior_layer_cake_path not found (%s); continuing without prior",
          prior_layer_cake_path_.c_str());
      extractor_->setPrior(nullptr);
      return;
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    try {
      MapperConfig mc;
      mc.voxel_size_m = voxel_size_;
      MapperCore temp(mc);
      if (!temp.loadLayerCake(prior_layer_cake_path_)) {
        RCLCPP_ERROR(
            get_logger(), "prior loadLayerCake failed: %s",
            prior_layer_cake_path_.c_str());
        extractor_->setPrior(nullptr);
        return;
      }
      const nvblox::TsdfLayer& tsdf = temp.mapper().tsdf_layer();
      const nvblox::ColorLayer& color = temp.mapper().color_layer();
      const float pad = tsdf.block_size();
      const TsdfAabb aabb = aabbFromTsdfLayer(tsdf, pad);
      if (!aabb.valid) {
        RCLCPP_ERROR(
            get_logger(), "prior LayerCake has no TSDF blocks: %s",
            prior_layer_cake_path_.c_str());
        extractor_->setPrior(nullptr);
        return;
      }

      const float res = ec.resolution_m > 0.f ? ec.resolution_m : voxel_size_;
      const int width = std::max(
          1, static_cast<int>(std::ceil((aabb.max_x - aabb.min_x) / res)));
      const int height = std::max(
          1, static_cast<int>(std::ceil((aabb.max_y - aabb.min_y) / res)));
      // Cap absurd grids (e.g. corrupt cake) — ~200 m @ 5 cm ≈ 4000².
      constexpr int kMaxDim = 8192;
      if (width > kMaxDim || height > kMaxDim) {
        RCLCPP_ERROR(
            get_logger(),
            "prior BEV too large (%dx%d); increase elevation_resolution or "
            "check cake AABB",
            width, height);
        extractor_->setPrior(nullptr);
        return;
      }

      ElevationCostExtractor prior_ext(ec);
      LocalMaps maps = prior_ext.extractBevGrid(
          tsdf, color, aabb.min_x, aabb.min_y, width, height, aabb.min_z,
          aabb.max_z, 0.5f * (aabb.min_z + aabb.max_z), nullptr,
          /*use_prior=*/false);
      if (maps.width <= 0 || maps.elevation.empty()) {
        RCLCPP_ERROR(get_logger(), "prior extractBevGrid returned empty");
        extractor_->setPrior(nullptr);
        return;
      }

      const float tx = static_cast<float>(prior_xyz_yaw_[0]);
      const float ty = static_cast<float>(prior_xyz_yaw_[1]);
      const float tz = static_cast<float>(prior_xyz_yaw_[2]);
      const float yaw = static_cast<float>(prior_xyz_yaw_[3]);
      nvblox::CudaStreamOwning upload_stream;
      if (!prior_bev_->upload(
              maps.elevation, maps.cost, maps.observed, maps.origin_x,
              maps.origin_y, maps.resolution, tx, ty, tz, yaw,
              upload_stream)) {
        RCLCPP_ERROR(get_logger(), "prior GPU upload failed");
        extractor_->setPrior(nullptr);
        return;
      }
      extractor_->setPrior(prior_bev_.get());
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0)
              .count();
      const int observed = cv::countNonZero(maps.observed);
      RCLCPP_INFO(
          get_logger(),
          "prior BEV ready in %.1fms: %dx%d @ %.3fm origin=(%.2f,%.2f) "
          "observed=%d/%d T_global_prior=[%.2f,%.2f,%.2f,yaw=%.3f] from %s",
          ms, maps.width, maps.height, maps.resolution, maps.origin_x,
          maps.origin_y, observed, maps.width * maps.height, tx, ty, tz, yaw,
          prior_layer_cake_path_.c_str());
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(
          get_logger(), "prior BEV load exception: %s", ex.what());
      prior_bev_->clear();
      extractor_->setPrior(nullptr);
    }
  }

  bool lookupTf(
      const std::string& target, const std::string& source,
      const rclcpp::Time& stamp, nvblox::Transform* T_target_source)
  {
    const auto timeout =
        rclcpp::Duration::from_seconds(std::max(0.0, tf_lookup_timeout_s_));
    try {
      const auto tf = tf_buffer_->lookupTransform(
          target, source, stamp, timeout);
      *T_target_source = toNvblox(tf2::transformToEigen(tf.transform));
      return true;
    } catch (const tf2::TransformException& ex) {
      if (tf_fallback_to_latest_) {
        try {
          // TimePointZero = latest; tolerates bag TF lag / slight stamp skew.
          const auto tf = tf_buffer_->lookupTransform(
              target, source, rclcpp::Time(0, 0, RCL_ROS_TIME));
          *T_target_source = toNvblox(tf2::transformToEigen(tf.transform));
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 5000,
              "TF %s <- %s: using latest (stamp lookup failed: %s)",
              target.c_str(), source.c_str(), ex.what());
          return true;
        } catch (const tf2::TransformException& ex2) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000, "TF %s <- %s failed: %s",
              target.c_str(), source.c_str(), ex2.what());
          return false;
        }
      }
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "TF %s <- %s failed: %s",
          target.c_str(), source.c_str(), ex.what());
      return false;
    }
  }

  void requestDrainSoon()
  {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      drain_soon_ = true;
    }
    wake_cv_.notify_one();
  }

  void workerLoop()
  {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(wake_mutex_);
        if (!worker_running_) {
          return;
        }
        wake_cv_.wait(lock, [this]() {
          return drain_soon_ || !worker_running_;
        });
        drain_soon_ = false;
        if (!worker_running_) {
          return;
        }
      }
      drainOnce();
    }
  }

  void drainOnce()
  {
    using Clock = std::chrono::steady_clock;
    maybeSaveLayerCake();
    const bool map_cleared = maybeClearMap();

    const auto t_all = Clock::now();

    sensor_msgs::msg::Image::ConstSharedPtr mask_msg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr lidar_msg;
    {
      std::lock_guard<std::mutex> lock(inbox_mutex_);
      mask_msg = pending_mask_;
      lidar_msg = pending_lidar_;
    }

    double integrate_ms = 0.0;
    rclcpp::Time stamp_for_map{0, 0, RCL_ROS_TIME};
    bool any_integrate = false;

    // Order: depth (coarse) → lidar (correct geometry) → color (paint TSDF).
    struct PendingCam {
      std::shared_ptr<CamSlot> slot;
      sensor_msgs::msg::Image::ConstSharedPtr depth_msg;
      sensor_msgs::msg::Image::ConstSharedPtr color_msg;
      std::optional<nvblox::Camera> depth_cam;
      std::optional<nvblox::Camera> color_cam;
    };
    std::vector<PendingCam> pending_cams;
    pending_cams.reserve(cam_slots_.size());
    for (auto& slot : cam_slots_) {
      PendingCam pc;
      pc.slot = slot;
      {
        std::lock_guard<std::mutex> lock(slot->mu);
        pc.depth_msg = slot->pending_depth;
        pc.color_msg = slot->pending_color;
        slot->pending_depth.reset();
        slot->pending_color.reset();
        pc.depth_cam = slot->depth_camera;
        pc.color_cam = slot->color_camera;
        if (!pc.color_cam && pc.depth_cam) {
          pc.color_cam = pc.depth_cam;
        }
      }
      pending_cams.push_back(std::move(pc));
    }

    for (auto& pc : pending_cams) {
      if (!(pc.depth_msg && pc.depth_cam && use_depth_)) {
        continue;
      }
      const int64_t ns =
          rclcpp::Time(pc.depth_msg->header.stamp).nanoseconds();
      if (ns == pc.slot->last_depth_ns) {
        continue;
      }
      const auto t0 = Clock::now();
      nvblox::Transform T_L_C;
      if (lookupTf(
              global_frame_, pc.depth_msg->header.frame_id,
              pc.depth_msg->header.stamp, &T_L_C)) {
        nvblox::DepthImage depth(nvblox::MemoryType::kHost);
        if (depthToNvblox(*pc.depth_msg, &depth)) {
          zeroDepthBottomFraction(&depth, depth_ignore_bottom_fraction_);
          std::optional<nvblox::MonoImage> mask;
          if (mask_msg) {
            nvblox::MonoImage m(nvblox::MemoryType::kHost);
            if (maskToNvblox(*mask_msg, &m) && m.rows() == depth.rows() &&
                m.cols() == depth.cols()) {
              mask = std::move(m);
            }
          }
          mapper_->integrateDepth(depth, T_L_C, *pc.depth_cam, mask);
          pc.slot->last_depth_ns = ns;
          stamp_for_map = pc.depth_msg->header.stamp;
          any_integrate = true;
        } else {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 5000,
              "Unsupported depth encoding: %s",
              pc.depth_msg->encoding.c_str());
        }
      }
      integrate_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    if (lidar_msg && use_lidar_) {
      const auto t0 = Clock::now();
      nvblox::Transform T_L_S;
      if (lookupTf(
              global_frame_, lidar_msg->header.frame_id,
              lidar_msg->header.stamp, &T_L_S)) {
        nvblox::Pointcloud cloud(nvblox::MemoryType::kDevice);
        nvblox::Time scan_duration_ms(0);
        if (cloudToNvblox(
                *lidar_msg, &cloud, lidar_stream_, &scan_duration_ms)) {
          bool do_mc = use_lidar_motion_compensation_;
          std::optional<nvblox::Transform> T_end;
          std::optional<nvblox::Time> duration;
          if (do_mc) {
            if (static_cast<int64_t>(scan_duration_ms) <= 0 ||
                !cloud.timestamps_ms().has_value()) {
              RCLCPP_WARN_THROTTLE(
                  get_logger(), *get_clock(), 10000,
                  "Lidar MC requested but pointcloud has no usable "
                  "per-point times; integrating without MC");
              do_mc = false;
            } else {
              constexpr int64_t kMsToNs = 1000000;
              const int64_t start_ns =
                  rclcpp::Time(lidar_msg->header.stamp).nanoseconds();
              const rclcpp::Time scan_end(
                  start_ns +
                  static_cast<int64_t>(scan_duration_ms) * kMsToNs);
              nvblox::Transform T_L_S_end;
              if (lookupTf(
                      global_frame_, lidar_msg->header.frame_id, scan_end,
                      &T_L_S_end)) {
                T_end = T_L_S_end;
                duration = scan_duration_ms;
              } else {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 5000,
                    "Lidar MC: TF at scan end failed; integrating without MC");
                do_mc = false;
              }
            }
          }
          mapper_->integrateLidar(
              cloud, T_L_S, lidar_, do_mc, T_end, duration);
          maybePublishLidarDepthImage(
              cloud, T_L_S, do_mc, T_end, duration, lidar_msg->header);
          if (stamp_for_map.nanoseconds() == 0) {
            stamp_for_map = lidar_msg->header.stamp;
          }
          any_integrate = true;
          std::lock_guard<std::mutex> lock(inbox_mutex_);
          if (pending_lidar_ == lidar_msg) {
            pending_lidar_.reset();
          }
        }
      }
      integrate_ms += std::chrono::duration<double, std::milli>(
                          Clock::now() - t0)
                          .count();
    }

    for (auto& pc : pending_cams) {
      if (!use_color_) {
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skip color integrate [%s]: use_color=false",
            pc.slot ? pc.slot->name.c_str() : "?");
        continue;
      }
      if (!pc.color_msg) {
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skip color integrate [%s]: no pending color image",
            pc.slot ? pc.slot->name.c_str() : "?");
        continue;
      }
      if (!pc.color_cam) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skip color integrate [%s]: no color camera model "
            "(missing/invalid camera_info)",
            pc.slot ? pc.slot->name.c_str() : "?");
        continue;
      }
      const int64_t ns =
          rclcpp::Time(pc.color_msg->header.stamp).nanoseconds();
      if (ns == pc.slot->last_color_ns) {
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skip color integrate [%s]: duplicate color stamp",
            pc.slot ? pc.slot->name.c_str() : "?");
        continue;
      }
      const auto t0 = Clock::now();
      nvblox::Transform T_L_C;
      if (lookupTf(
              global_frame_, pc.color_msg->header.frame_id,
              pc.color_msg->header.stamp, &T_L_C)) {
        nvblox::ColorImage color(nvblox::MemoryType::kHost);
        bool ok = false;
        if (debug_semantic_all_traversable_) {
          ok = whiteColorImage(
              static_cast<int>(pc.color_msg->height),
              static_cast<int>(pc.color_msg->width), &color);
        } else {
          ok = colorToNvblox(*pc.color_msg, &color);
        }
        if (ok) {
          mapper_->integrateColor(color, T_L_C, *pc.color_cam);
          pc.slot->last_color_ns = ns;
          if (stamp_for_map.nanoseconds() == 0) {
            stamp_for_map = pc.color_msg->header.stamp;
          }
          any_integrate = true;
        } else {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 5000,
              "Skip color integrate [%s]: unsupported color encoding: %s",
              pc.slot ? pc.slot->name.c_str() : "?",
              pc.color_msg->encoding.c_str());
        }
      } else {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Skip color integrate [%s]: TF lookup failed %s <- %s @ %.3f",
            pc.slot ? pc.slot->name.c_str() : "?",
            global_frame_.c_str(), pc.color_msg->header.frame_id.c_str(),
            rclcpp::Time(pc.color_msg->header.stamp).seconds());
      }
      integrate_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    const rclcpp::Time stamp =
        stamp_for_map.nanoseconds() != 0 ? stamp_for_map : now();

    nvblox::Transform T_L_clear;
    if (!lookupTf(
            global_frame_, map_clearing_frame_, stamp, &T_L_clear)) {
      return;
    }

    if (rateOk(&last_clear_, map_clear_rate_hz_, stamp)) {
      mapper_->clearOutsideRadius(T_L_clear.translation());
    }

    // Event-driven: skip BEV when this wake had no new integrate (save-only).
    if (!any_integrate && !map_cleared) {
      return;
    }

    ExtractTiming bev_timing;
    LocalMaps maps = extractor_->extractBev(
        mapper_->mapper().tsdf_layer(), mapper_->mapper().color_layer(),
        T_L_clear.translation(), T_L_clear.translation().z(), &bev_timing);
    const int grid_w = maps.width;
    const int grid_h = maps.height;

    const auto t_handoff = Clock::now();
    PublishJob job;
    job.maps = std::move(maps);
    job.stamp = stamp;
    job.clearing_z = T_L_clear.translation().z();
    job.integrate_ms = integrate_ms;
    job.bev_ms = bev_timing.bev_ms;
    const double handoff_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_handoff)
            .count();
    job.handoff_ms = handoff_ms;
    job.loop1_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_all)
            .count();
    {
      std::lock_guard<std::mutex> lock(pub_mutex_);
      // Latest-only: drop stale BEV if loop2 is behind.
      pending_pub_ = std::move(job);
    }
    pub_cv_.notify_one();

    maybePublishTsdfColorMesh(stamp);
    maybePublishTsdfVoxels(stamp);
  }

  void publishLoop()
  {
    while (true) {
      PublishJob job;
      {
        std::unique_lock<std::mutex> lock(pub_mutex_);
        pub_cv_.wait(lock, [this]() {
          return pending_pub_.has_value() || !publish_running_;
        });
        if (!publish_running_ && !pending_pub_) {
          return;
        }
        if (!pending_pub_) {
          continue;
        }
        job = std::move(*pending_pub_);
        pending_pub_.reset();
      }

      using Clock = std::chrono::steady_clock;
      const auto t0 = Clock::now();
      ExtractTiming fin;
      finish_extractor_->finishLocalMap(&job.maps, &fin);
      const double finish_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

      {
        std_msgs::msg::Float64MultiArray tmsg;
        // [integrate, bev, handoff, inpaint, slope, colorize, loop1, loop2]
        tmsg.data = {
            job.integrate_ms, job.bev_ms, job.handoff_ms, fin.inpaint_ms,
            fin.slope_ms, fin.colorize_ms, job.loop1_ms, finish_ms};
        extract_timing_pub_->publish(tmsg);
      }

      std_msgs::msg::Header header;
      header.stamp = job.stamp;
      header.frame_id = global_frame_;

      // Skip serialize/publish when nobody is listening (RViz/nav off).
      const size_t local_subs =
          local_map_pub_->get_subscription_count() +
          local_map_pub_->get_intra_process_subscription_count();
      if (local_subs > 0) {
        hound_mapping::msg::LocalMap local_map;
        local_map.header = header;
        local_map.info.map_load_time = job.stamp;
        local_map.info.resolution = job.maps.resolution;
        local_map.info.width = static_cast<uint32_t>(job.maps.width);
        local_map.info.height = static_cast<uint32_t>(job.maps.height);
        local_map.info.origin.position.x = job.maps.origin_x;
        local_map.info.origin.position.y = job.maps.origin_y;
        local_map.info.origin.position.z = job.clearing_z;
        local_map.info.origin.orientation.w = 1.0;
        local_map.elevation =
            matToImageMsg(job.maps.elevation, header, "32FC1");
        local_map.costmap = matToImageMsg(job.maps.cost, header, "32FC1");
        local_map_pub_->publish(std::move(local_map));
      }
      maybeDumpElev(job.stamp, job.maps);
    }
  }

  void maybePublishLidarDepthImage(
      const nvblox::Pointcloud& cloud, const nvblox::Transform& T_L_S,
      bool do_mc, const std::optional<nvblox::Transform>& T_end,
      const std::optional<nvblox::Time>& duration,
      const std_msgs::msg::Header& cloud_header)
  {
    if (!publish_lidar_depth_image_ && !publish_lidar_tsdf_probe_) {
      return;
    }
    const bool need_depth =
        publish_lidar_depth_image_ &&
        lidar_depth_pub_->get_subscription_count() > 0;
    const bool need_viz =
        publish_lidar_depth_image_ &&
        lidar_depth_viz_pub_->get_subscription_count() > 0;
    const bool need_cloud =
        publish_lidar_depth_image_ &&
        lidar_depth_cloud_pub_->get_subscription_count() > 0;
    const bool need_probe =
        publish_lidar_tsdf_probe_ &&
        lidar_tsdf_probe_pub_->get_subscription_count() > 0;
    if (!need_depth && !need_viz && !need_cloud && !need_probe) {
      return;
    }

    nvblox::DepthImage depth_device(nvblox::MemoryType::kDevice);
    nvblox::depthImageFromPointcloudGPU(
        cloud, T_L_S, lidar_, do_mc, T_end, duration, &depth_device,
        *lidar_stream_);
    lidar_stream_->synchronize();

    nvblox::DepthImage depth_host(nvblox::MemoryType::kHost);
    depth_host.copyFromAsync(depth_device, *lidar_stream_);
    lidar_stream_->synchronize();

    const int rows = depth_host.rows();
    const int cols = depth_host.cols();
    if (rows <= 0 || cols <= 0) {
      return;
    }

    std_msgs::msg::Header header = cloud_header;
    header.frame_id = cloud_header.frame_id;

    if (need_depth) {
      sensor_msgs::msg::Image msg;
      msg.header = header;
      msg.height = static_cast<uint32_t>(rows);
      msg.width = static_cast<uint32_t>(cols);
      msg.encoding = "32FC1";
      msg.is_bigendian = false;
      msg.step = static_cast<uint32_t>(cols * sizeof(float));
      msg.data.resize(static_cast<size_t>(rows) * msg.step);
      for (int r = 0; r < rows; ++r) {
        std::memcpy(
            msg.data.data() + static_cast<size_t>(r) * msg.step,
            depth_host.dataConstPtr() + static_cast<size_t>(r) * cols,
            static_cast<size_t>(cols) * sizeof(float));
      }
      lidar_depth_pub_->publish(msg);
    }

    if (need_viz) {
      // Normalize valid depths to mono8; invalid/empty → 0 (black gaps).
      float dmin = std::numeric_limits<float>::max();
      float dmax = 0.f;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const float d = depth_host(r, c);
          if (d > 0.f && std::isfinite(d)) {
            dmin = std::min(dmin, d);
            dmax = std::max(dmax, d);
          }
        }
      }
      const float span = (dmax > dmin) ? (dmax - dmin) : 1.f;
      sensor_msgs::msg::Image msg;
      msg.header = header;
      msg.height = static_cast<uint32_t>(rows);
      msg.width = static_cast<uint32_t>(cols);
      msg.encoding = "mono8";
      msg.is_bigendian = false;
      msg.step = static_cast<uint32_t>(cols);
      msg.data.resize(static_cast<size_t>(rows) * cols);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const float d = depth_host(r, c);
          uint8_t v = 0;
          if (d > 0.f && std::isfinite(d)) {
            v = static_cast<uint8_t>(
                std::clamp((d - dmin) / span * 255.f, 0.f, 255.f));
          }
          msg.data[static_cast<size_t>(r) * cols + c] = v;
        }
      }
      lidar_depth_viz_pub_->publish(msg);
    }

    if (need_cloud) {
      // Unproject filled spherical pixels → pointcloud in lidar frame
      // (same frame as raw /livox/cloud). Empty bins produce no points.
      std::vector<float> xyz;
      xyz.reserve(static_cast<size_t>(rows) * cols * 3 / 4);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const float d = depth_host(r, c);
          if (!(d > 0.f) || !std::isfinite(d)) {
            continue;
          }
          const nvblox::Vector3f p =
              lidar_.unprojectFromPixelIndices(nvblox::Index2D(c, r), d);
          xyz.push_back(p.x());
          xyz.push_back(p.y());
          xyz.push_back(p.z());
        }
      }
      const size_t n = xyz.size() / 3;
      sensor_msgs::msg::PointCloud2 out;
      out.header = header;
      out.height = 1;
      out.width = static_cast<uint32_t>(n);
      out.is_dense = true;
      out.is_bigendian = false;
      sensor_msgs::PointCloud2Modifier mod(out);
      mod.setPointCloud2FieldsByString(1, "xyz");
      mod.resize(n);
      sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");
      for (size_t i = 0; i < n; ++i, ++iter_x, ++iter_y, ++iter_z) {
        *iter_x = xyz[i * 3 + 0];
        *iter_y = xyz[i * 3 + 1];
        *iter_z = xyz[i * 3 + 2];
      }
      lidar_depth_cloud_pub_->publish(out);
    }

    if (need_probe) {
      // Query TSDF at reprojected hits + 6 face neighbors (layer frame).
      // RGB focuses on what marching cubes needs (local sign change):
      //   red=no block, yellow=weight≈0,
      //   green=local +/- straddles 0 (meshable),
      //   magenta=weight>0 but no local zero-crossing (won't mesh).
      constexpr float kMinWeight = 1e-4f;
      const float vs = std::max(voxel_size_, 1e-3f);
      const nvblox::Vector3f neighbor_off[7] = {
          {0.f, 0.f, 0.f}, {vs, 0.f, 0.f},  {-vs, 0.f, 0.f},
          {0.f, vs, 0.f},  {0.f, -vs, 0.f}, {0.f, 0.f, vs},
          {0.f, 0.f, -vs}};

      std::vector<nvblox::Vector3f> hit_L;
      hit_L.reserve(static_cast<size_t>(
          std::max(1, lidar_tsdf_probe_max_points_)));
      const int max_pts = std::max(1, lidar_tsdf_probe_max_points_);
      int n_valid = 0;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const float d = depth_host(r, c);
          if (d > 0.f && std::isfinite(d)) {
            ++n_valid;
          }
        }
      }
      const int stride =
          std::max(1, (n_valid + max_pts - 1) / max_pts);
      int seen = 0;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          const float d = depth_host(r, c);
          if (!(d > 0.f) || !std::isfinite(d)) {
            continue;
          }
          if ((seen++ % stride) != 0) {
            continue;
          }
          const nvblox::Vector3f p_S =
              lidar_.unprojectFromPixelIndices(nvblox::Index2D(c, r), d);
          hit_L.push_back(T_L_S * p_S);
          if (static_cast<int>(hit_L.size()) >= max_pts) {
            break;
          }
        }
        if (static_cast<int>(hit_L.size()) >= max_pts) {
          break;
        }
      }

      std::vector<nvblox::Vector3f> query_L;
      query_L.reserve(hit_L.size() * 7);
      for (const auto& p : hit_L) {
        for (const auto& o : neighbor_off) {
          query_L.push_back(p + o);
        }
      }

      std::vector<nvblox::TsdfVoxel> voxels;
      std::vector<bool> ok;
      mapper_->mapper().tsdf_layer().getVoxels(
          query_L, &voxels, &ok, lidar_stream_.get());

      size_t n_none = 0;
      size_t n_w0 = 0;
      size_t n_cross = 0;
      size_t n_all_pos = 0;
      size_t n_all_neg = 0;
      size_t n_isolated = 0;
      double sum_abs_cross = 0.0;
      double sum_abs_nocross = 0.0;
      double sum_w_nocross = 0.0;

      sensor_msgs::msg::PointCloud2 out;
      out.header.stamp = cloud_header.stamp;
      out.header.frame_id = global_frame_;
      out.height = 1;
      out.width = static_cast<uint32_t>(hit_L.size());
      out.is_dense = true;
      out.is_bigendian = false;
      sensor_msgs::PointCloud2Modifier mod(out);
      mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
      mod.resize(hit_L.size());
      sensor_msgs::PointCloud2Iterator<float> iter_x(out, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(out, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(out, "z");
      sensor_msgs::PointCloud2Iterator<float> iter_rgb(out, "rgb");

      auto pack_rgb = [](uint8_t r, uint8_t g, uint8_t b) {
        const uint32_t packed =
            (static_cast<uint32_t>(r) << 16) |
            (static_cast<uint32_t>(g) << 8) |
            static_cast<uint32_t>(b);
        float f = 0.f;
        std::memcpy(&f, &packed, sizeof(float));
        return f;
      };

      for (size_t i = 0; i < hit_L.size();
           ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
        *iter_x = hit_L[i].x();
        *iter_y = hit_L[i].y();
        *iter_z = hit_L[i].z();
        const size_t base = i * 7;
        float rgb = 0.f;
        if (!ok[base]) {
          ++n_none;
          rgb = pack_rgb(255, 40, 40);  // red: no block at hit
        } else if (voxels[base].weight < kMinWeight) {
          ++n_w0;
          rgb = pack_rgb(255, 200, 0);  // yellow: weight ≈ 0
        } else {
          bool has_pos = false;
          bool has_neg = false;
          int n_weighted = 0;
          for (size_t k = 0; k < 7; ++k) {
            if (!ok[base + k] || voxels[base + k].weight < kMinWeight) {
              continue;
            }
            ++n_weighted;
            if (voxels[base + k].distance >= 0.f) {
              has_pos = true;
            } else {
              has_neg = true;
            }
          }
          const float abs_d = std::fabs(voxels[base].distance);
          if (has_pos && has_neg) {
            ++n_cross;
            sum_abs_cross += abs_d;
            rgb = pack_rgb(40, 220, 40);  // green: local zero-crossing
          } else if (n_weighted <= 1) {
            ++n_isolated;
            sum_abs_nocross += abs_d;
            sum_w_nocross += voxels[base].weight;
            rgb = pack_rgb(255, 140, 0);  // orange: only this voxel weighted
          } else if (has_pos) {
            ++n_all_pos;
            sum_abs_nocross += abs_d;
            sum_w_nocross += voxels[base].weight;
            rgb = pack_rgb(255, 0, 255);  // magenta: all free (+)
          } else {
            ++n_all_neg;
            sum_abs_nocross += abs_d;
            sum_w_nocross += voxels[base].weight;
            rgb = pack_rgb(0, 160, 255);  // blue: all occupied (-)
          }
        }
        *iter_rgb = rgb;
      }
      lidar_tsdf_probe_pub_->publish(out);

      const size_t n_nocross = n_all_pos + n_all_neg + n_isolated;
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "lidar_tsdf_probe: samples=%zu none=%zu(red) w0=%zu(yellow) "
          "cross=%zu(green) all+=%zu(magenta) all-=%zu(blue) "
          "isolated=%zu(orange) |d|_cross=%.3f |d|_nocross=%.3f "
          "w_nocross=%.2f stride=%d bins=%d",
          hit_L.size(), n_none, n_w0, n_cross, n_all_pos, n_all_neg,
          n_isolated,
          n_cross ? sum_abs_cross / static_cast<double>(n_cross) : 0.0,
          n_nocross ? sum_abs_nocross / static_cast<double>(n_nocross)
                    : 0.0,
          n_nocross ? sum_w_nocross / static_cast<double>(n_nocross) : 0.0,
          stride, n_valid);
    }
  }

  void maybePublishTsdfVoxels(const rclcpp::Time& stamp)
  {
    if (!publish_tsdf_voxels_) {
      return;
    }
    if (!rateOk(&last_tsdf_voxels_, tsdf_voxel_rate_hz_, stamp)) {
      return;
    }
    if (tsdf_voxels_pub_->get_subscription_count() == 0) {
      return;
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const nvblox::TsdfLayer& tsdf = mapper_->mapper().tsdf_layer();
    const nvblox::ColorLayer& color = mapper_->mapper().color_layer();
    const float block_size = tsdf.block_size();
    const float voxel_size = tsdf.voxel_size();
    const float max_abs_d =
        tsdf_voxel_max_distance_m_ > 0.0
            ? static_cast<float>(tsdf_voxel_max_distance_m_)
            : (2.f * voxel_size);
    const size_t max_pts = static_cast<size_t>(
        std::max(1, tsdf_voxel_max_points_));

    // ColorLayer is usually device memory; clone blocks once for host lookup.
    nvblox::Index3DHashMapType<nvblox::ColorBlock::ConstPtr>::type color_host;
    const bool clone_color =
        (color.memory_type() == nvblox::MemoryType::kDevice);
    for (const nvblox::Index3D& bi : color.getAllBlockIndices()) {
      nvblox::ColorBlock::ConstPtr blk = color.getBlockAtIndex(bi);
      if (!blk) {
        continue;
      }
      color_host.emplace(
          bi, clone_color ? blk.clone(nvblox::MemoryType::kHost) : blk);
    }

    std::vector<nvblox::Vector3f> pts;
    std::vector<uint32_t> packed_rgb;
    pts.reserve(std::min(max_pts, size_t{65536}));
    packed_rgb.reserve(std::min(max_pts, size_t{65536}));

    size_t n_geom = 0;
    size_t n_colored = 0;
    size_t n_trunc = 0;
    nvblox::callFunctionOnAllVoxels<nvblox::TsdfVoxel>(
        tsdf,
        [&](const nvblox::Index3D& bi, const nvblox::Index3D& vi,
            const nvblox::TsdfVoxel* vox) {
          if (pts.size() >= max_pts) {
            ++n_trunc;
            return;
          }
          if (vox->weight < tsdf_voxel_min_weight_) {
            return;
          }
          if (std::fabs(vox->distance) > max_abs_d) {
            return;
          }
          ++n_geom;
          pts.push_back(nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
              block_size, bi, vi));

          // Magenta = TSDF surface with no ColorVoxel weight (integration miss).
          uint8_t r = 255, g = 0, b = 255;
          const auto it = color_host.find(bi);
          if (it != color_host.end() && it->second) {
            const nvblox::ColorVoxel& cv =
                it->second->voxels[vi.x()][vi.y()][vi.z()];
            if (cv.weight > 1e-4f) {
              r = cv.color.r();
              g = cv.color.g();
              b = cv.color.b();
              ++n_colored;
            }
          }
          packed_rgb.push_back(
              (static_cast<uint32_t>(r) << 16) |
              (static_cast<uint32_t>(g) << 8) |
              static_cast<uint32_t>(b));
        });

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = global_frame_;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(pts.size());
    cloud.is_dense = true;
    cloud.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
    mod.resize(pts.size());
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud, "rgb");
    for (size_t i = 0; i < pts.size();
         ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
      *iter_x = pts[i].x();
      *iter_y = pts[i].y();
      *iter_z = pts[i].z();
      std::memcpy(&(*iter_rgb), &packed_rgb[i], sizeof(float));
    }
    tsdf_voxels_pub_->publish(cloud);

    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "tsdf_voxels: pts=%zu colored=%zu geom=%zu trunc_extra=%zu "
        "color_blocks=%zu |d|<=%.3f in %.1fms → ~/tsdf_voxels",
        pts.size(), n_colored, n_geom, n_trunc, color_host.size(), max_abs_d,
        ms);
  }

  void maybePublishTsdfColorMesh(const rclcpp::Time& stamp)
  {
    if (!publish_tsdf_color_mesh_) {
      return;
    }
    if (!rateOk(&last_tsdf_color_mesh_, tsdf_color_mesh_rate_hz_, stamp)) {
      return;
    }
    // Only pay for mesh update if someone is listening.
    const bool need_cloud = tsdf_color_cloud_pub_->get_subscription_count() > 0;
    const bool need_mesh = tsdf_color_mesh_pub_->get_subscription_count() > 0;
    if (!need_cloud && !need_mesh) {
      return;
    }

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    // Mesh blocks are not reliably host-accessible; serialize to host_vector.
    mapper_->mapper().updateColorMesh(nvblox::UpdateFullLayer::kYes);
    const std::vector<nvblox::Index3D> block_ids =
        mapper_->mapper().color_mesh_layer().getAllBlockIndices();
    if (block_ids.empty()) {
      return;
    }
    mapper_->mapper().serializeSelectedLayers(
        nvblox::LayerTypeBitMask(nvblox::LayerType::kColorMesh),
        /*bandwidth_limit_mbps=*/-1.f, nvblox::BlockExclusionParams(),
        block_ids);
    const std::shared_ptr<nvblox::SerializedColorMeshLayer> serialized =
        mapper_->mapper().serializedColorMeshLayer();
    if (!serialized || serialized->vertices.empty()) {
      return;
    }

    const size_t n_verts = serialized->vertices.size();
    const size_t n_tri_idx = serialized->triangle_indices.size();
    const size_t n_tris = n_tri_idx / 3;
    const bool have_c = serialized->vertex_appearances.size() == n_verts;

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = global_frame_;

    if (need_cloud) {
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header = header;
      cloud.height = 1;
      cloud.width = static_cast<uint32_t>(n_verts);
      cloud.is_dense = false;
      cloud.is_bigendian = false;
      sensor_msgs::PointCloud2Modifier mod(cloud);
      mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
      mod.resize(n_verts);
      sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
      sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud, "rgb");
      for (size_t i = 0; i < n_verts; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
        const auto& v = serialized->vertices[i];
        *iter_x = v.x();
        *iter_y = v.y();
        *iter_z = v.z();
        uint8_t r = 200, g = 200, b = 200;
        if (have_c) {
          const auto& c = serialized->vertex_appearances[i];
          r = c.r();
          g = c.g();
          b = c.b();
        }
        const uint32_t packed =
            (static_cast<uint32_t>(r) << 16) |
            (static_cast<uint32_t>(g) << 8) |
            static_cast<uint32_t>(b);
        std::memcpy(&(*iter_rgb), &packed, sizeof(float));
      }
      tsdf_color_cloud_pub_->publish(cloud);
    }

    if (need_mesh && n_tris > 0 &&
        serialized->vertex_block_offsets.size() >= 2 &&
        serialized->triangle_index_block_offsets.size() >= 2) {
      const size_t max_tris =
          tsdf_color_mesh_max_triangles_ > 0
              ? static_cast<size_t>(tsdf_color_mesh_max_triangles_)
              : n_tris;
      visualization_msgs::msg::Marker marker;
      marker.header = header;
      marker.ns = "tsdf_color_mesh";
      marker.id = 0;
      marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 1.0;
      marker.scale.y = 1.0;
      marker.scale.z = 1.0;
      marker.points.reserve(std::min(n_tris, max_tris) * 3);
      marker.colors.reserve(std::min(n_tris, max_tris) * 3);

      size_t emitted = 0;
      const size_t n_blocks = serialized->block_indices.size();
      for (size_t bi = 0; bi < n_blocks && emitted < max_tris; ++bi) {
        if (bi + 1 >= serialized->vertex_block_offsets.size() ||
            bi + 1 >= serialized->triangle_index_block_offsets.size()) {
          break;
        }
        const size_t v_base =
            static_cast<size_t>(serialized->vertex_block_offsets[bi]);
        const size_t t0 =
            static_cast<size_t>(serialized->triangle_index_block_offsets[bi]);
        const size_t t1 = static_cast<size_t>(
            serialized->triangle_index_block_offsets[bi + 1]);
        for (size_t t = t0; t + 2 < t1 && emitted < max_tris; t += 3, ++emitted) {
          bool ok = true;
          geometry_msgs::msg::Point pts[3];
          std_msgs::msg::ColorRGBA cols[3];
          for (int k = 0; k < 3; ++k) {
            const int local =
                serialized->triangle_indices[t + static_cast<size_t>(k)];
            if (local < 0) {
              ok = false;
              break;
            }
            const size_t vi = v_base + static_cast<size_t>(local);
            if (vi >= n_verts) {
              ok = false;
              break;
            }
            const auto& v = serialized->vertices[vi];
            pts[k].x = v.x();
            pts[k].y = v.y();
            pts[k].z = v.z();
            cols[k].a = 1.f;
            if (have_c) {
              const auto& c = serialized->vertex_appearances[vi];
              cols[k].r = c.r() / 255.f;
              cols[k].g = c.g() / 255.f;
              cols[k].b = c.b() / 255.f;
            } else {
              cols[k].r = cols[k].g = cols[k].b = 0.8f;
            }
          }
          if (!ok) {
            continue;
          }
          for (int k = 0; k < 3; ++k) {
            marker.points.push_back(pts[k]);
            marker.colors.push_back(cols[k]);
          }
        }
      }
      if (!marker.points.empty()) {
        tsdf_color_mesh_pub_->publish(marker);
      }
    }

    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "tsdf_color_mesh: blocks=%zu verts=%zu tris=%zu in %.1fms",
        block_ids.size(), n_verts, n_tris, ms);
  }

  void maybeSaveLayerCake()
  {
    bool do_save = false;
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      do_save = save_requested_;
      save_requested_ = false;
    }
    if (!do_save || layer_cake_path_.empty()) {
      return;
    }

    namespace fs = std::filesystem;
    try {
      const fs::path out(layer_cake_path_);
      if (out.has_parent_path()) {
        fs::create_directories(out.parent_path());
      }
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(
          get_logger(), "save_layer_cake mkdir failed: %s", ex.what());
      return;
    }

    const size_t tsdf_blocks =
        mapper_->mapper().tsdf_layer().numBlocks();
    const size_t color_blocks =
        mapper_->mapper().color_layer().numBlocks();

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const bool ok = mapper_->saveLayerCake(layer_cake_path_);
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    std::uintmax_t bytes = 0;
    try {
      if (ok && fs::exists(layer_cake_path_)) {
        bytes = fs::file_size(layer_cake_path_);
      }
    } catch (...) {
    }

    if (ok) {
      RCLCPP_INFO(
          get_logger(),
          "saveLayerCake OK in %.1fms → %s (tsdf_blocks=%zu color_blocks=%zu "
          "file=%.2f MiB) — this tick blocked integrate/BEV",
          ms, layer_cake_path_.c_str(), tsdf_blocks, color_blocks,
          static_cast<double>(bytes) / (1024.0 * 1024.0));
    } else {
      RCLCPP_ERROR(
          get_logger(),
          "saveLayerCake FAILED after %.1fms → %s (tsdf_blocks=%zu "
          "color_blocks=%zu)",
          ms, layer_cake_path_.c_str(), tsdf_blocks, color_blocks);
    }
  }

  bool maybeClearMap()
  {
    bool do_clear = false;
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
      do_clear = clear_requested_;
      clear_requested_ = false;
    }
    if (!do_clear) {
      return false;
    }

    const size_t tsdf_blocks = mapper_->mapper().tsdf_layer().numBlocks();
    const size_t color_blocks = mapper_->mapper().color_layer().numBlocks();

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    mapper_->clear();
    {
      std::lock_guard<std::mutex> lock(pub_mutex_);
      pending_pub_.reset();
    }
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    RCLCPP_WARN(
        get_logger(),
        "map_clear: empty TSDF in %.1fms (dropped tsdf_blocks=%zu "
        "color_blocks=%zu). Prior BEV unchanged. Live integrate continues.",
        ms, tsdf_blocks, color_blocks);
    return true;
  }

  void maybeDumpElev(const rclcpp::Time& stamp, const LocalMaps& maps)
  {
    if (dump_elev_dir_.empty() || maps.width <= 0 || maps.height <= 0) {
      return;
    }
    if (maps.elevation.empty()) {
      return;
    }
    {
      double absmax = 0.0;
      const float* p = maps.elevation.ptr<float>(0);
      const int n = maps.elevation.rows * maps.elevation.cols;
      for (int i = 0; i < n; ++i) {
        absmax = std::max(absmax, static_cast<double>(std::fabs(p[i])));
      }
      if (absmax < 1e-6) {
        return;
      }
    }
    const double hz = std::max(dump_elev_rate_hz_, 1e-3);
    int fi =
        static_cast<int>(std::llround(stamp.seconds() * hz)) - 1;
    if (fi < 0) {
      return;
    }
    if (dump_elev_n_frames_ > 0) {
      fi = fi % dump_elev_n_frames_;
    }
    std::ostringstream stem;
    stem << dump_elev_dir_ << '/' << std::setw(6) << std::setfill('0') << fi;
    const std::string base = stem.str();
    {
      std::ofstream bout(base + "_elev.f32", std::ios::binary);
      bout.write(
          reinterpret_cast<const char*>(maps.elevation.ptr<float>(0)),
          static_cast<std::streamsize>(maps.elevation.total() * sizeof(float)));
    }
    {
      std::ofstream tout(base + "_elev.txt");
      tout << "height=" << maps.height << "\n"
           << "width=" << maps.width << "\n"
           << "resolution=" << maps.resolution << "\n"
           << "origin_x=" << maps.origin_x << "\n"
           << "origin_y=" << maps.origin_y << "\n";
    }
    {
      std::ofstream cout_bin(base + "_cost.f32", std::ios::binary);
      cout_bin.write(
          reinterpret_cast<const char*>(maps.cost.ptr<float>(0)),
          static_cast<std::streamsize>(maps.cost.total() * sizeof(float)));
    }
  }

  // Params
  std::string global_frame_;
  std::string map_clearing_frame_;
  float voxel_size_{0.05f};
  float elevation_resolution_m_{0.05f};
  float map_clearing_radius_m_{32.f};
  float local_map_size_m_{0.f};
  float max_integration_distance_m_{15.f};
  float lidar_max_integration_distance_m_{15.f};
  float lethal_slope_deg_{60.f};
  bool do_inpaint_{true};
  float inpaint_resolution_m_{0.25f};
  int inpaint_radius_{3};
  int elev_median_ksize_{5};
  float unobserved_luminance_{128.f};
  float color_weight_min_{1e-3f};
  float color_z_search_half_band_m_{0.f};
  float color_z_column_above_m_{0.f};
  float bev_occ_tsdf_max_m_{0.f};
  float robot_height_m_{0.4f};
  float elev_z_min_rel_{-1.f};
  float elev_z_max_rel_{1.5f};
  bool use_depth_{true};
  bool use_color_{true};
  bool debug_semantic_all_traversable_{false};
  bool use_lidar_{true};
  bool use_people_mask_{false};
  double depth_ignore_bottom_fraction_{0.0};
  std::string depth_topic_;
  std::string depth_info_topic_;
  std::string color_topic_;
  std::string color_info_topic_;
  std::string lidar_topic_;
  std::string people_mask_topic_;
  std::vector<std::string> camera_names_;
  std::string depth_topic_template_;
  std::string depth_info_topic_template_;
  std::string color_topic_template_;
  std::string color_info_topic_template_;
  int lidar_width_{360};
  int lidar_height_{90};
  float lidar_vertical_fov_rad_{2.094395f};
  float lidar_elevation_below_deg_{0.f};
  float lidar_elevation_above_deg_{0.f};
  float lidar_min_valid_range_m_{0.3f};
  bool use_lidar_motion_compensation_{false};
  double mapper_rate_hz_{20.0};
  double publish_map_rate_hz_{20.0};
  double integrate_depth_rate_hz_{20.0};
  double integrate_color_rate_hz_{20.0};
  double integrate_lidar_rate_hz_{10.0};
  bool log_extract_timing_{false};
  bool publish_tsdf_color_mesh_{false};
  double tsdf_color_mesh_rate_hz_{1.0};
  int tsdf_color_mesh_max_triangles_{500000};
  bool publish_tsdf_voxels_{false};
  double tsdf_voxel_rate_hz_{1.0};
  float tsdf_voxel_min_weight_{0.1f};
  double tsdf_voxel_max_distance_m_{0.0};
  int tsdf_voxel_max_points_{400000};
  bool publish_lidar_depth_image_{false};
  bool publish_lidar_tsdf_probe_{false};
  int lidar_tsdf_probe_max_points_{4000};
  double map_clear_rate_hz_{20.0};
  bool extract_on_depth_{false};
  std::string dump_elev_dir_;
  double dump_elev_rate_hz_{20.0};
  int dump_elev_n_frames_{0};
  std::string layer_cake_path_;
  std::string map_clear_topic_{"~/map_clear"};
  std::string prior_layer_cake_path_;
  std::vector<double> prior_xyz_yaw_{0.0, 0.0, 0.0, 0.0};
  bool prior_fill_enabled_{true};
  double tf_lookup_timeout_s_{0.05};
  bool tf_fallback_to_latest_{false};

  std::unique_ptr<MapperCore> mapper_;
  std::unique_ptr<ElevationCostExtractor> extractor_;
  std::unique_ptr<ElevationCostExtractor> finish_extractor_;
  std::unique_ptr<PriorBevGpu> prior_bev_;
  nvblox::Lidar lidar_;
  std::shared_ptr<nvblox::CudaStream> lidar_stream_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::vector<std::shared_ptr<CamSlot>> cam_slots_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<hound_mapping::msg::LocalMap>::SharedPtr local_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      tsdf_color_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      tsdf_color_mesh_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lidar_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lidar_depth_viz_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      lidar_depth_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      lidar_tsdf_probe_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      tsdf_voxels_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      extract_timing_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr map_clear_sub_;

  std::mutex inbox_mutex_;
  sensor_msgs::msg::Image::ConstSharedPtr pending_mask_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr pending_lidar_;

  rclcpp::Time last_lidar_integrate_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_clear_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_tsdf_color_mesh_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_tsdf_voxels_{0, 0, RCL_ROS_TIME};

  std::thread worker_thread_;
  std::mutex wake_mutex_;
  std::condition_variable wake_cv_;
  bool worker_running_{false};
  bool drain_soon_{false};
  bool save_requested_{false};
  bool clear_requested_{false};

  std::thread publish_thread_;
  std::mutex pub_mutex_;
  std::condition_variable pub_cv_;
  bool publish_running_{false};
  std::optional<PublishJob> pending_pub_;
};

}  // namespace hound_mapping

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hound_mapping::MappingNode>());
  rclcpp::shutdown();
  return 0;
}
