#pragma once

#include "data_center.h"
namespace ceres_loam {

class CloudSegmentation {
 public:
  std::shared_ptr<CloudSegmentation> CloudSegmentationPtr;
  std::shared_ptr<const CloudSegmentation> CloudSegmentationConstPtr;

 public:
  CloudSegmentation();
  ~CloudSegmentation() = default;

 private:
  void allocateMemory();
  void resetParameters();

  // image projection
  void projectPointCloud();
  void groundRemoval();
  void cloudSegmentation();

  // segmentation

  template <typename T>
  void removeClosedPointCloud(const typename pcl::PointCloud<T> &cloud_in,
                              typename pcl::PointCloud<T> &cloud_out,
                              float threshold);
  void labelComponents(int row, int col);
  void divideSurfEdge();

 public:
  void CloudHandler(const PointCloudPtr cloud, const double stamp);

 private:
  // lidar
  float angle_res_x_;  // horizontal
  float angle_res_y_;  // vertical

  double timestamp_;
  PointCloudPtr lidar_cloud_in_;
  PointCloudPtr full_cloud_;
  PointCloudPtr full_cloud_with_range_;
  PointCloudPtr ground_cloud_;
  PointCloudPtr segmented_cloud_with_ground_;
  PointCloudPtr segmented_cloud_pure_;
  PointCloudPtr outlier_cloud_;

  // PointCloudPtr corner_cloud_sharp_;
  // PointCloudPtr corner_cloud_lesssharp_;
  // PointCloudPtr surf_cloud_flat_;
  // PointCloudPtr surf_cloud_lessflat_;

  std::vector<float> cloud_curvature_;
  std::vector<int> cloud_sort_idx_;
  std::vector<int> cloud_neighbor_picked_;
  std::vector<int> cloud_label_;

  Eigen::MatrixXd range_mat_;
  Eigen::MatrixXi label_mat_;

  // ground_mat: ground matrix for ground cloud marking
  // -1, no valid info to check if ground of not
  //  0, initial value, after validation, means not ground
  //  1, ground
  Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> ground_mat_;
  int label_count_;
};
}  // namespace ceres_loam