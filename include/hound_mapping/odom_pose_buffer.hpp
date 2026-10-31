#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>

#include <Eigen/Geometry>

namespace hound_mapping
{

struct StampedOdomPose
{
  int64_t stamp_ns{0};
  Eigen::Isometry3d T_odom_base{Eigen::Isometry3d::Identity()};
};

inline Eigen::Isometry3d interpolate_se3(
    const Eigen::Isometry3d& T0, const Eigen::Isometry3d& T1, double alpha)
{
  alpha = std::clamp(alpha, 0.0, 1.0);
  const Eigen::Quaterniond q0(T0.rotation());
  const Eigen::Quaterniond q1(T1.rotation());
  const Eigen::Quaterniond q = q0.slerp(alpha, q1).normalized();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = q.toRotationMatrix();
  T.translation() = (1.0 - alpha) * T0.translation() + alpha * T1.translation();
  return T;
}

/** Ring of T_odom_base from an odometry topic (lidar-composite deskew pattern). */
class OdomPoseBuffer
{
 public:
  void set_horizon_s(double seconds)
  {
    const double s = seconds > 0.0 ? seconds : 1.0;
    horizon_ns_ = static_cast<int64_t>(s * 1e9);
  }

  void push(StampedOdomPose sample)
  {
    if (sample.stamp_ns <= 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!poses_.empty() && sample.stamp_ns < poses_.back().stamp_ns) {
      return;
    }
    poses_.push_back(std::move(sample));
    const int64_t keep_ns = poses_.back().stamp_ns - horizon_ns_;
    while (!poses_.empty() && poses_.front().stamp_ns < keep_ns) {
      poses_.pop_front();
    }
  }

  bool latest(Eigen::Isometry3d* T) const
  {
    if (T == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (poses_.empty()) {
      return false;
    }
    *T = poses_.back().T_odom_base;
    return true;
  }

  /**
   * Interpolate T_odom_base at stamp_ns.
   * clamp=true uses the nearest end sample when the stamp is outside the buffer
   * (same as deskew poseAt). clamp=false fails outside [first, last].
   */
  bool pose_at(int64_t stamp_ns, Eigen::Isometry3d* T, bool clamp) const
  {
    if (T == nullptr || stamp_ns <= 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (poses_.empty()) {
      return false;
    }
    if (stamp_ns <= poses_.front().stamp_ns) {
      if (!clamp && stamp_ns < poses_.front().stamp_ns) {
        return false;
      }
      *T = poses_.front().T_odom_base;
      return true;
    }
    if (stamp_ns >= poses_.back().stamp_ns) {
      if (!clamp && stamp_ns > poses_.back().stamp_ns) {
        return false;
      }
      *T = poses_.back().T_odom_base;
      return true;
    }
    auto it = std::lower_bound(
        poses_.begin(), poses_.end(), stamp_ns,
        [](const StampedOdomPose& p, int64_t t) { return p.stamp_ns < t; });
    if (it == poses_.begin()) {
      *T = poses_.front().T_odom_base;
      return true;
    }
    const StampedOdomPose& b = *it;
    const StampedOdomPose& a = *std::prev(it);
    if (b.stamp_ns <= a.stamp_ns) {
      *T = b.T_odom_base;
      return true;
    }
    const double alpha = static_cast<double>(stamp_ns - a.stamp_ns) /
        static_cast<double>(b.stamp_ns - a.stamp_ns);
    *T = interpolate_se3(a.T_odom_base, b.T_odom_base, alpha);
    return true;
  }

 private:
  mutable std::mutex mu_;
  std::deque<StampedOdomPose> poses_;
  int64_t horizon_ns_{1000000000};
};

}  // namespace hound_mapping
