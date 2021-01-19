#pragma once

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace ceres_loam {

struct Config {
  // lidar
  bool use_cloud_ring = false;
  int num_scans = 16;                 // 16 vlps:16   32vlps:32
  int num_horizontal_scans = 1800;    // 16 vlps:1800 32vlps:1800
  int ground_scan_index = 7;          // 16 vlps:7    32vlps:20
  float vertical_angle_bottom = -15;  // 16 vlps:-15  32vlps:-25
  float vertical_angle_top = 15;      // 16 vlps:15   32vlps:15
  float sensor_mount_angle = 0;       // 16 vlps:0
  float scan_period = 0.1;            // 16 vlps:0.1

  //   float angle_res_x;   // 16 vlps:0.2
  //   float angle_res_y;   // 16 vlps:2
  //   float angle_bottom;  // 16 vlps:15.1

  // segmentation
  int segmented_cluster_num = 30;     // default:30 yc:120
  int segmented_valid_point_num = 5;  // default:5 yc:4
  int segmented_valid_line_num = 3;   // default:3, yc:4
  float segment_theta = 60;  // default:60.0 degrees, decrease the value can
                             // improve accuracy
  float surf_segmentation_threshold = 10.0;  // default:10.0
  // segment to surf and corner
  float minimum_range = 0.3;  // default:0.3

  // feature registration
  float dist_sq_threshold = 25;
  float nearby_scan = 2.5;

  // lidar-refination
  float corner_voxel_fiter_size = 0.2;
  float surf_voxel_fiter_size = 0.4;
  float cube_size = 14.0;  // default:50m
  int cloud_width = 21;    // default:21
  int cloud_height = 21;   // default:21
  int cloud_depth = 11;    // default:11

  int cloud_cen_width = 10;
  int cloud_cen_height = 10;
  int cloud_cen_depth = 5;
};

template <typename T>
class SingleInstance {
 public:
  template <typename... Args>
  static T& GetInstance(Args&&... args) {
    if (instance_ == nullptr) {
      instance_.reset(new T(std::forward<Args>(args)...));
    }
    return *instance_.get();
  }

 private:
  static std::unique_ptr<T> instance_;
  static std::once_flag& get_once_flag() {
    static std::once_flag once_;
    return once_;
  }
};
template <class T>
std::unique_ptr<T> SingleInstance<T>::instance_ = nullptr;

class LoamConfig : public SingleInstance<LoamConfig> {
  friend class SingleInstance<LoamConfig>;

 public:
  LoamConfig(const LoamConfig& rhs) = delete;
  LoamConfig& operator=(const LoamConfig& rhs) = delete;

  const bool init = false;
  Config config;

 private:
  LoamConfig() : instance_cnt_(0) {
    instance_cnt_++;
    if (!init) {
      LOG(ERROR) << "Attempting to use loam's config without initlization..";
    }
  }
  LoamConfig(const Config& config) : config(config), init(true) {}
  int instance_cnt_;
};
bool LoadConfig(const std::string& yaml_file);
}  // namespace ceres_loam