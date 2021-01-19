#pragma once
#include "data_center.h"

namespace ceres_loam {
#define DISTORTION 0

class LidarOdometry {
 public:
  std::shared_ptr<LidarOdometry> LidarOdometryPtr;
  std::shared_ptr<const LidarOdometry> LidarOdometryConstPtr;
  LidarOdometry();
  ~LidarOdometry() = default;

 private:
  void allocateMemory();
  void accessData();

  void transformToStart(PointT const *const pi, PointT *const po);
  void calculateTransformation();

 public:
  void Process();

  const Eigen::Quaterniond GetQuaternion() { return odom_q_; }
  const Eigen::Vector3d GetPos() { return odom_t_; }
  const Eigen::Affine3d GetRefinedPose() {
    auto odom_correct_msg = DataCenter::Instance().GetOdomCorrectMsg();
    odom_q_ = odom_correct_msg->correct_q * odom_q_;
    odom_t_ =
        odom_correct_msg->correct_q * odom_t_ + odom_correct_msg->correct_t;

    Eigen::Affine3d result =
        Eigen::Translation3d(odom_t_.x(), odom_t_.y(), odom_t_.z()) *
        odom_q_.normalized();
    return result;
  }

 private:
  Config config_;
  bool init_;
  double para_q[4];          // = {0.0, 0.0, 0.0, 1.0};
  double para_t[3];          // = {0.0, 0.0, 0.0};
  Eigen::Quaterniond tf_q_;  //(para_q);
  Eigen::Vector3d tf_t_;     //(para_t);

  // transformation from current frame to world frame
  Eigen::Quaterniond odom_q_;
  Eigen::Vector3d odom_t_;

  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;
  PointCloudPtr laserCloudFullRes;
  PointCloudPtr cornerPointsSharp;
  PointCloudPtr cornerPointsLessSharp;
  PointCloudPtr surfPointsFlat;
  PointCloudPtr surfPointsLessFlat;

  PointCloudPtr laserCloudCornerLast;
  PointCloudPtr laserCloudSurfLast;
};
}  // namespace ceres_loam