#pragma once
#include "cloud_segmentation.h"
#include "lidar_odometry.h"
#include "lidar_refination.h"

namespace ceres_loam {
class LoamOdometry {
 public:
  typedef std::shared_ptr<LoamOdometry> LoamOdometryPtr;
  typedef std::shared_ptr<const LoamOdometry> LoamOdometryConstPtr;

 public:
  LoamOdometry() = default;
  ~LoamOdometry() = default;

  void Process(const PointCloudPtr cloud, const double stamp) {
    cloud_segmentation_.CloudHandler(cloud, stamp);
    lidar_odom_.Process();
    lidar_refination_.Process();
  }

  const PointCloudPtr GetMapCloud() {
    return lidar_refination_.GenerateWholeMap();
  }
  const PointCloudPtr GetSurroundMap() {
    return lidar_refination_.GenerateSurroundMap();
  }
  // odom coarse pose
  const Eigen::Affine3d GetOdomCoarsePose() {
    auto odom_q_ = lidar_odom_.GetQuaternion();
    auto odom_t_ = lidar_odom_.GetPos();

    Eigen::Affine3d result =
        Eigen::Translation3d(odom_t_.x(), odom_t_.y(), odom_t_.z()) *
        odom_q_.normalized();
    return result;
  }
  // mapping refined pose
  const Eigen::Affine3d GetRefinedPose() { return lidar_refination_.GetPose(); }
  // odom refined pose
  const Eigen::Affine3d GetPose() { return lidar_odom_.GetRefinedPose(); }
  // odom refined tf
  const Eigen::Affine3d GetTransfromation() {
    static Eigen::Affine3d last_pose = Eigen::Affine3d::Identity();
    Eigen::Affine3d pose = lidar_odom_.GetRefinedPose();
    Eigen::Affine3d trans = last_pose.inverse() * pose;
    last_pose = pose;
    return trans;
  }

 private:
  CloudSegmentation cloud_segmentation_;
  LidarOdometry lidar_odom_;
  LidarRefination lidar_refination_;
};
}  // namespace ceres_loam