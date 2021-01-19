#pragma once
#include "data_center.h"

namespace ceres_loam {

class LidarRefination {
 public:
  std::shared_ptr<LidarRefination> LidarMappingPtr;
  std::shared_ptr<const LidarRefination> LidarMappingConstPtr;

 public:
  LidarRefination();
  ~LidarRefination() = default;

  void Process();
  const Eigen::Quaterniond GetQuaternion() { return q_w_curr_; }
  const Eigen::Vector3d GetPos() { return t_w_curr_; }
  const Eigen::Affine3d GetPose() {
    Eigen::Affine3d result =
        Eigen::Translation3d(t_w_curr_.x(), t_w_curr_.y(), t_w_curr_.z()) *
        q_w_curr_.normalized();
    // return result;
    return refined_pose;
  }

  const PointCloudPtr GenerateWholeMap();
  const PointCloudPtr GenerateSurroundMap();

 private:
  void allocateMemory();
  void accessData();
  void accessAvailableCubicNum(int &scan_valid_num, int &sub_scan_valid_num);

  // use q_w_curr_ t_w_curr_即parameters to convert the point
  void pointAssociateToMap(PointT const *const pi, PointT *const po);

  void addFeatureCloudtoPool();

  void prepareData(const int featuremap_scans);
  void calculateTransformation();
  void calculateTransformationSurf(ceres::Problem &problem,
                                   ceres::LossFunction *loss_function);
  void calculateTransformationCorner(ceres::Problem &problem,
                                     ceres::LossFunction *loss_function);
  void setInitialGuess();
  void updateOptimizedResult();

  void downSampleCornerSurfArray(const int scan_valid_num);

 private:
  Config config_;
  int laser_cloud_num_;
  Eigen::Affine3d odom_pose, correct_pose, refined_pose;

  // global pose of lidar-odom assign by odom
  Eigen::Vector3d odom_t_;
  Eigen::Quaterniond odom_q_;
  // intermediate variable
  Eigen::Vector3d correct_t_;
  Eigen::Quaterniond correct_q_;

  pcl::VoxelGrid<PointT> voxel_filter_corner_;
  pcl::VoxelGrid<PointT> voxel_filter_surf_;
  PointCloudPtr feature_scan_corner_;
  PointCloudPtr feature_scan_corner_filtered_;
  PointCloudPtr feature_scan_surf_;
  PointCloudPtr feature_scan_surf_filtered_;
  PointCloudPtr cloud_full_res;  // useless

  // surround points in map to build tree
  int scans_valid_indices[125];
  int subscans_valid_indices[125];
  PointCloudPtr corner_map_;
  PointCloudPtr surf_map_;
  PointCloudPtr surround_map_;
  // kd-tree
  pcl::KdTreeFLANN<PointT>::Ptr kdtree_corner_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kdtree_surf_map_;
  std::vector<int> neighbor_pts_indices;
  std::vector<float> neighbor_pts_dist;

  // final optimized global pose
  double parameters[7];
  Eigen::Vector3d t_w_curr_;     //(parameters);
  Eigen::Quaterniond q_w_curr_;  //(parameters);

  // points in every cube
  std::vector<PointCloudPtr> corners_pool;
  std::vector<PointCloudPtr> surfs_pool;
  PointCloudPtr globis_map;
};
void LogOutputAffine3dPose(const std::string &msgs, const Eigen::Affine3d pose);
}  // namespace ceres_loam