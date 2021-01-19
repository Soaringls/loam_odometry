#pragma once
#include <float.h>
#include <glog/logging.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <Eigen/Dense>
#include <Eigen/QR>
#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cmath>
#include <ctime>

#include "config.h"
namespace ceres_loam {
const double DEG_TO_RAD = M_PI / 180.0;
const double RAD_TO_DEG = 180.0 / M_PI;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloudI;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;
typedef pcl::PointCloud<pcl::PointXYZI>::ConstPtr PointCloudConstPtr;

class Timer {
 public:
  Timer() { tic(); }

  void tic() { start_ = std::chrono::system_clock::now(); }

  double end(bool restart = false) {
    end_ = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_ - start_;
    if (restart) {
      start_ = std::chrono::system_clock::now();
    }
    return elapsed_seconds.count() * 1000;  // ms
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_, end_;
};

}  // namespace ceres_loam