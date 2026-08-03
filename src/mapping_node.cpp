#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
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
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core.hpp>

#include <nvblox/nvblox.h>

#include "hound_mapping/elevation_cost.hpp"
#include "hound_mapping/mapper_core.hpp"
#include "hound_mapping/msg/local_map.hpp"

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
    std::shared_ptr<nvblox::CudaStream> stream)
{
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
  std::vector<nvblox::Vector3f> pts;
  pts.reserve(static_cast<size_t>(msg.width) * msg.height);
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    const float x = *iter_x;
    const float y = *iter_y;
    const float z = *iter_z;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    pts.emplace_back(x, y, z);
  }
  if (pts.empty()) {
    return false;
  }
  *out = nvblox::Pointcloud(nvblox::MemoryType::kDevice);
  out->copyPointsFromAsync(pts, *stream);
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

class MappingNode : public rclcpp::Node {
 public:
  MappingNode()
  : Node("hound_mapping")
  {
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
    ec.window_radius_m = map_clearing_radius_m_;
    ec.unobserved_luminance = unobserved_luminance_;
    ec.color_weight_min = color_weight_min_;
    ec.robot_height_m = robot_height_m_;
    ec.elev_z_min_rel = elev_z_min_rel_;
    ec.elev_z_max_rel = elev_z_max_rel_;
    ec.lethal_slope_deg = lethal_slope_deg_;
    ec.do_inpaint = do_inpaint_;
    ec.inpaint_resolution_m = inpaint_resolution_m_;
    ec.inpaint_radius = inpaint_radius_;
    extractor_ = std::make_unique<ElevationCostExtractor>(ec);

    // Snapshots for extract-off-lock (copy under map_mutex_, BEV unlocked).
    {
      nvblox::BlockMemoryPoolParams pool(nvblox::MemoryType::kDevice);
      tsdf_snap_ = std::make_unique<nvblox::TsdfLayer>(voxel_size_, pool);
      color_snap_ = std::make_unique<nvblox::ColorLayer>(voxel_size_, pool);
      snap_stream_ = std::make_shared<nvblox::CudaStreamOwning>();
    }

    lidar_ = nvblox::Lidar(
        lidar_width_, lidar_height_, lidar_min_valid_range_m_,
        lidar_vertical_fov_rad_);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    local_map_pub_ =
        create_publisher<hound_mapping::msg::LocalMap>("~/local_map", 1);
    elev_color_pub_ =
        create_publisher<sensor_msgs::msg::Image>("~/elev_color", 1);
    extract_timing_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
            "~/extract_timing_ms", 10);

    auto sensor_qos = rclcpp::SensorDataQoS();

    if (use_depth_) {
      depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
          depth_topic_, sensor_qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            onDepth(msg);
          });
      depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
          depth_info_topic_, sensor_qos,
          [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            depth_info_ = *msg;
            depth_camera_ = cameraFromInfo(*msg);
          });
    }
    if (use_color_) {
      color_sub_ = create_subscription<sensor_msgs::msg::Image>(
          color_topic_, sensor_qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            onColor(msg);
          });
      color_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
          color_info_topic_, sensor_qos,
          [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            color_info_ = *msg;
            color_camera_ = cameraFromInfo(*msg);
          });
    }
    if (use_lidar_) {
      lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          lidar_topic_, sensor_qos,
          [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
            onLidar(msg);
          });
    }
    if (use_people_mask_) {
      mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
          people_mask_topic_, sensor_qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            latest_mask_ = msg;
          });
    }

    const double publish_period =
        1.0 / std::max(publish_map_rate_hz_, 1.0);
    map_timer_ = create_wall_timer(
        std::chrono::duration<double>(publish_period),
        [this]() { requestExtract(); });

    extract_running_ = true;
    extract_thread_ = std::thread([this]() { extractLoop(); });

    RCLCPP_INFO(
        get_logger(),
        "hound_mapping ready: depth=%d color=%d lidar=%d voxel=%.3f "
        "window=%.1fm elev=TSDF→BEV (no ESDF) extract=worker+snapshot",
        use_depth_, use_color_, use_lidar_, voxel_size_,
        map_clearing_radius_m_);
  }

  ~MappingNode() override
  {
    {
      std::lock_guard<std::mutex> lock(extract_wake_mutex_);
      extract_running_ = false;
      extract_pending_ = true;
    }
    extract_cv_.notify_one();
    if (extract_thread_.joinable()) {
      extract_thread_.join();
    }
  }

 private:
  void declareParams()
  {
    declare_parameter("global_frame", "odom");
    declare_parameter("map_clearing_frame_id", "camera_link");
    declare_parameter("voxel_size", 0.05);
    declare_parameter("elevation_resolution", 0.05);
    declare_parameter("map_clearing_radius_m", 32.0);
    declare_parameter("max_integration_distance_m", 15.0);
    declare_parameter("lidar_max_integration_distance_m", 15.0);
    declare_parameter("lethal_slope_deg", 60.0);
    declare_parameter("do_inpaint", true);
    declare_parameter("inpaint_resolution_m", 0.25);
    declare_parameter("inpaint_radius", 3);
    declare_parameter("unobserved_luminance", 0.0);
    declare_parameter("color_weight_min", 0.001);
    declare_parameter("robot_height_m", 0.4);
    declare_parameter("elev_z_min_rel", -1.0);
    declare_parameter("elev_z_max_rel", 1.5);

    declare_parameter("use_depth", true);
    declare_parameter("use_color", true);
    declare_parameter("use_lidar", true);
    declare_parameter("use_people_mask", false);

    declare_parameter("depth_topic", "/camera/depth/image_rect_raw");
    declare_parameter("depth_info_topic", "/camera/depth/camera_info");
    declare_parameter("color_topic", "/camera/color/image_raw");
    declare_parameter("color_info_topic", "/camera/color/camera_info");
    declare_parameter("lidar_topic", "/unilidar/cloud");
    declare_parameter("people_mask_topic", "/segmentation/coarse_people_mask");

    declare_parameter("lidar_width", 360);
    declare_parameter("lidar_height", 90);
    declare_parameter("lidar_vertical_fov_rad", 2.094395);
    declare_parameter("lidar_min_valid_range_m", 0.3);
    declare_parameter("use_lidar_motion_compensation", false);

    declare_parameter("integrate_depth_rate_hz", 20.0);
    declare_parameter("integrate_color_rate_hz", 20.0);
    declare_parameter("integrate_lidar_rate_hz", 10.0);
    declare_parameter("map_clear_rate_hz", 20.0);
    declare_parameter("log_extract_timing", true);
    declare_parameter("publish_map_rate_hz", 5.0);
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
    unobserved_luminance_ = static_cast<float>(
        get_parameter("unobserved_luminance").as_double());
    color_weight_min_ = static_cast<float>(
        get_parameter("color_weight_min").as_double());
    robot_height_m_ = static_cast<float>(
        get_parameter("robot_height_m").as_double());
    elev_z_min_rel_ = static_cast<float>(
        get_parameter("elev_z_min_rel").as_double());
    elev_z_max_rel_ = static_cast<float>(
        get_parameter("elev_z_max_rel").as_double());

    use_depth_ = get_parameter("use_depth").as_bool();
    use_color_ = get_parameter("use_color").as_bool();
    use_lidar_ = get_parameter("use_lidar").as_bool();
    use_people_mask_ = get_parameter("use_people_mask").as_bool();

    depth_topic_ = get_parameter("depth_topic").as_string();
    depth_info_topic_ = get_parameter("depth_info_topic").as_string();
    color_topic_ = get_parameter("color_topic").as_string();
    color_info_topic_ = get_parameter("color_info_topic").as_string();
    lidar_topic_ = get_parameter("lidar_topic").as_string();
    people_mask_topic_ = get_parameter("people_mask_topic").as_string();

    lidar_width_ = get_parameter("lidar_width").as_int();
    lidar_height_ = get_parameter("lidar_height").as_int();
    lidar_vertical_fov_rad_ = static_cast<float>(
        get_parameter("lidar_vertical_fov_rad").as_double());
    lidar_min_valid_range_m_ = static_cast<float>(
        get_parameter("lidar_min_valid_range_m").as_double());
    use_lidar_motion_compensation_ =
        get_parameter("use_lidar_motion_compensation").as_bool();

    integrate_depth_rate_hz_ =
        get_parameter("integrate_depth_rate_hz").as_double();
    integrate_color_rate_hz_ =
        get_parameter("integrate_color_rate_hz").as_double();
    integrate_lidar_rate_hz_ =
        get_parameter("integrate_lidar_rate_hz").as_double();
    map_clear_rate_hz_ = get_parameter("map_clear_rate_hz").as_double();
    log_extract_timing_ = get_parameter("log_extract_timing").as_bool();
    publish_map_rate_hz_ = get_parameter("publish_map_rate_hz").as_double();
  }

  bool lookupTf(
      const std::string& target, const std::string& source,
      const rclcpp::Time& stamp, nvblox::Transform* T_target_source)
  {
    try {
      const auto tf = tf_buffer_->lookupTransform(
          target, source, stamp, rclcpp::Duration::from_seconds(0.05));
      *T_target_source = toNvblox(tf2::transformToEigen(tf.transform));
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "TF %s <- %s failed: %s",
          target.c_str(), source.c_str(), ex.what());
      return false;
    }
  }

  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    nvblox::Camera camera;
    sensor_msgs::msg::Image::ConstSharedPtr mask_msg;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (!rateOk(
              &last_depth_integrate_, integrate_depth_rate_hz_,
              msg->header.stamp)) {
        return;
      }
      if (!depth_camera_.has_value()) {
        return;
      }
      camera = *depth_camera_;
      if (use_people_mask_) {
        mask_msg = latest_mask_;
      }
    }

    nvblox::Transform T_L_C;
    if (!lookupTf(
            global_frame_, msg->header.frame_id, msg->header.stamp, &T_L_C)) {
      return;
    }
    nvblox::DepthImage depth(nvblox::MemoryType::kHost);
    if (!depthToNvblox(*msg, &depth)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Unsupported depth encoding: %s", msg->encoding.c_str());
      return;
    }
    std::optional<nvblox::MonoImage> mask;
    if (mask_msg) {
      nvblox::MonoImage m(nvblox::MemoryType::kHost);
      if (maskToNvblox(*mask_msg, &m) && m.rows() == depth.rows() &&
          m.cols() == depth.cols()) {
        mask = std::move(m);
      }
    }

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      mapper_->integrateDepth(depth, T_L_C, camera, mask);
      maybeClearLocked(msg->header.stamp);
    }
  }

  void onColor(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    nvblox::Camera camera;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (!rateOk(
              &last_color_integrate_, integrate_color_rate_hz_,
              msg->header.stamp)) {
        return;
      }
      if (!color_camera_.has_value()) {
        return;
      }
      camera = *color_camera_;
    }

    nvblox::Transform T_L_C;
    if (!lookupTf(
            global_frame_, msg->header.frame_id, msg->header.stamp, &T_L_C)) {
      return;
    }
    nvblox::ColorImage color(nvblox::MemoryType::kHost);
    if (!colorToNvblox(*msg, &color)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Unsupported color encoding: %s", msg->encoding.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      mapper_->integrateColor(color, T_L_C, camera);
    }
  }

  void onLidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
  {
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (!rateOk(
              &last_lidar_integrate_, integrate_lidar_rate_hz_,
              msg->header.stamp)) {
        return;
      }
    }

    nvblox::Transform T_L_S;
    if (!lookupTf(
            global_frame_, msg->header.frame_id, msg->header.stamp, &T_L_S)) {
      return;
    }
    auto stream = std::make_shared<nvblox::CudaStreamOwning>();
    nvblox::Pointcloud cloud(nvblox::MemoryType::kDevice);
    if (!cloudToNvblox(*msg, &cloud, stream)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      mapper_->integrateLidar(
          cloud, T_L_S, lidar_, use_lidar_motion_compensation_);
      maybeClearLocked(msg->header.stamp);
    }
  }

  /// Caller must hold map_mutex_. Radius clear only (elev reads TSDF).
  void maybeClearLocked(const rclcpp::Time& stamp)
  {
    if (!rateOk(&last_clear_, map_clear_rate_hz_, stamp)) {
      return;
    }
    nvblox::Transform T_L_clear;
    if (lookupTf(
            global_frame_, map_clearing_frame_, stamp, &T_L_clear)) {
      mapper_->clearOutsideRadius(T_L_clear.translation());
    }
  }

  void requestExtract()
  {
    {
      std::lock_guard<std::mutex> lock(extract_wake_mutex_);
      extract_pending_ = true;
    }
    extract_cv_.notify_one();
  }

  void extractLoop()
  {
    while (true) {
      {
        std::unique_lock<std::mutex> wake(extract_wake_mutex_);
        extract_cv_.wait(wake, [this]() {
          return extract_pending_ || !extract_running_;
        });
        if (!extract_running_) {
          return;
        }
        extract_pending_ = false;
      }

      const auto stamp = now();
      nvblox::Transform T_L_clear;
      if (!lookupTf(
              global_frame_, map_clearing_frame_, stamp, &T_L_clear)) {
        continue;
      }

      double snapshot_ms = 0.0;
      {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (rateOk(&last_clear_, map_clear_rate_hz_, stamp)) {
          mapper_->clearOutsideRadius(T_L_clear.translation());
        }
        tsdf_snap_->copyFromAsync(
            mapper_->mapper().tsdf_layer(), *snap_stream_);
        color_snap_->copyFromAsync(
            mapper_->mapper().color_layer(), *snap_stream_);
        snap_stream_->synchronize();
        snapshot_ms = std::chrono::duration<double, std::milli>(
                          Clock::now() - t0)
                          .count();
      }
      // map_mutex_ released — integrate can proceed while we BEV+publish.

      ExtractTiming timing;
      const LocalMaps maps = extractor_->extract(
          *tsdf_snap_, *color_snap_, T_L_clear.translation(),
          T_L_clear.translation().z(), &timing);

      {
        std_msgs::msg::Float64MultiArray tmsg;
        // [bev, snapshot, inpaint, slope, colorize, total] ms
        tmsg.data = {
            timing.bev_ms, snapshot_ms, timing.inpaint_ms, timing.slope_ms,
            timing.colorize_ms, timing.total_ms};
        extract_timing_pub_->publish(tmsg);
      }
      if (log_extract_timing_) {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "extract %.1fms (snapshot %.1f bev %.1f inpaint %.1f slope %.1f "
            "colorize %.1f) grid %dx%d",
            timing.total_ms, snapshot_ms, timing.bev_ms, timing.inpaint_ms,
            timing.slope_ms, timing.colorize_ms, maps.width, maps.height);
      }

      std_msgs::msg::Header header;
      header.stamp = stamp;
      header.frame_id = global_frame_;

      hound_mapping::msg::LocalMap local_map;
      local_map.header = header;
      local_map.info.map_load_time = stamp;
      local_map.info.resolution = maps.resolution;
      local_map.info.width = static_cast<uint32_t>(maps.width);
      local_map.info.height = static_cast<uint32_t>(maps.height);
      local_map.info.origin.position.x = maps.origin_x;
      local_map.info.origin.position.y = maps.origin_y;
      local_map.info.origin.position.z = T_L_clear.translation().z();
      local_map.info.origin.orientation.w = 1.0;
      local_map.elevation = matToImageMsg(maps.elevation, header, "32FC1");
      local_map.costmap = matToImageMsg(maps.cost, header, "32FC1");
      local_map.normals = matToImageMsg(maps.normals, header, "32FC3");
      local_map.observed = matToImageMsg(maps.observed, header, "mono8");
      local_map_pub_->publish(local_map);

      elev_color_pub_->publish(matToImageMsg(maps.elev_color, header, "bgr8"));
    }
  }

  // Params
  std::string global_frame_;
  std::string map_clearing_frame_;
  float voxel_size_{0.05f};
  float elevation_resolution_m_{0.05f};
  float map_clearing_radius_m_{32.f};
  float max_integration_distance_m_{15.f};
  float lidar_max_integration_distance_m_{15.f};
  float lethal_slope_deg_{60.f};
  bool do_inpaint_{true};
  float inpaint_resolution_m_{0.25f};
  int inpaint_radius_{3};
  float unobserved_luminance_{0.f};
  float color_weight_min_{1e-3f};
  float robot_height_m_{0.4f};
  float elev_z_min_rel_{-1.f};
  float elev_z_max_rel_{1.5f};
  bool use_depth_{true};
  bool use_color_{true};
  bool use_lidar_{true};
  bool use_people_mask_{false};
  std::string depth_topic_;
  std::string depth_info_topic_;
  std::string color_topic_;
  std::string color_info_topic_;
  std::string lidar_topic_;
  std::string people_mask_topic_;
  int lidar_width_{360};
  int lidar_height_{90};
  float lidar_vertical_fov_rad_{2.094395f};
  float lidar_min_valid_range_m_{0.3f};
  bool use_lidar_motion_compensation_{false};
  double integrate_depth_rate_hz_{20.0};
  double integrate_color_rate_hz_{20.0};
  double integrate_lidar_rate_hz_{10.0};
  bool log_extract_timing_{true};
  double map_clear_rate_hz_{20.0};
  double publish_map_rate_hz_{5.0};

  std::unique_ptr<MapperCore> mapper_;
  std::unique_ptr<ElevationCostExtractor> extractor_;
  std::unique_ptr<nvblox::TsdfLayer> tsdf_snap_;
  std::unique_ptr<nvblox::ColorLayer> color_snap_;
  std::shared_ptr<nvblox::CudaStream> snap_stream_;
  nvblox::Lidar lidar_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<hound_mapping::msg::LocalMap>::SharedPtr local_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr elev_color_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
      extract_timing_pub_;
  rclcpp::TimerBase::SharedPtr map_timer_;

  std::mutex map_mutex_;
  std::optional<nvblox::Camera> depth_camera_;
  std::optional<nvblox::Camera> color_camera_;
  sensor_msgs::msg::CameraInfo depth_info_;
  sensor_msgs::msg::CameraInfo color_info_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_mask_;

  rclcpp::Time last_depth_integrate_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_color_integrate_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_lidar_integrate_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_clear_{0, 0, RCL_ROS_TIME};

  std::thread extract_thread_;
  std::mutex extract_wake_mutex_;
  std::condition_variable extract_cv_;
  bool extract_running_{false};
  bool extract_pending_{false};
};

}  // namespace hound_mapping

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hound_mapping::MappingNode>());
  rclcpp::shutdown();
  return 0;
}
