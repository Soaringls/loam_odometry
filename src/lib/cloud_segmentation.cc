#include "cloud_segmentation.h"

namespace ceres_loam {
CloudSegmentation::CloudSegmentation() {
  LoadConfig("asdf");
  const auto& config = LoamConfig::GetInstance().config;
  angle_res_x_ = M_PI * 2 / config.num_horizontal_scans;
  angle_res_y_ = (config.vertical_angle_top - config.vertical_angle_bottom) /
                 static_cast<float>(config.num_scans - 1) * DEG_TO_RAD;
  const std::size_t cloud_size = config.num_horizontal_scans * config.num_scans;

  full_cloud_.reset(new pcl::PointCloud<PointT>());
  full_cloud_with_range_.reset(new pcl::PointCloud<PointT>());
  full_cloud_->points.resize(cloud_size);
  full_cloud_with_range_->points.resize(cloud_size);

  lidar_cloud_in_.reset(new pcl::PointCloud<PointT>());
  ground_cloud_.reset(new pcl::PointCloud<PointT>());
  segmented_cloud_with_ground_.reset(new pcl::PointCloud<PointT>());
  segmented_cloud_pure_.reset(new pcl::PointCloud<PointT>());
  outlier_cloud_.reset(new pcl::PointCloud<PointT>());

  cloud_curvature_.reserve(cloud_size);
  cloud_sort_idx_.reserve(cloud_size);
  cloud_neighbor_picked_.reserve(cloud_size);
  cloud_label_.reserve(cloud_size);

  //   corner_cloud_sharp_.reset(new pcl::PointCloud<PointT>());
  //   corner_cloud_lesssharp_.reset(new pcl::PointCloud<PointT>());
  //   surf_cloud_flat_.reset(new pcl::PointCloud<PointT>());
  //   surf_cloud_lessflat_.reset(new pcl::PointCloud<PointT>());
}

void CloudSegmentation::resetParameters() {
  const auto& config = LoamConfig::GetInstance().config;
  const std::size_t cloud_size = config.num_horizontal_scans * config.num_scans;

  PointT nan_point;
  nan_point.x = std::numeric_limits<float>::quiet_NaN();
  nan_point.y = std::numeric_limits<float>::quiet_NaN();
  nan_point.z = std::numeric_limits<float>::quiet_NaN();
  nan_point.intensity = -1;
  std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nan_point);
  std::fill(full_cloud_with_range_->points.begin(),
            full_cloud_with_range_->points.end(), nan_point);

  timestamp_ = 0.0;
  lidar_cloud_in_->clear();
  ground_cloud_->clear();
  segmented_cloud_with_ground_->clear();
  segmented_cloud_pure_->clear();
  outlier_cloud_->clear();

  cloud_curvature_.clear();
  cloud_sort_idx_.clear();
  cloud_neighbor_picked_.clear();
  cloud_label_.clear();
  //   corner_cloud_sharp_->clear();
  //   corner_cloud_lesssharp_->clear();
  //   surf_cloud_flat_->clear();
  //   surf_cloud_lessflat_->clear();

  range_mat_.resize(config.num_scans, config.num_horizontal_scans);
  range_mat_.fill(FLT_MAX);
  ground_mat_.resize(config.num_scans, config.num_horizontal_scans);
  ground_mat_.setZero();
  label_mat_.resize(config.num_scans, config.num_horizontal_scans);
  label_mat_.setZero();

  label_count_ = 1;
}

void CloudSegmentation::CloudHandler(const PointCloudPtr cloud_in,
                                     const double stamp) {
  resetParameters();
  Timer elapsed_time;
  timestamp_ = stamp;

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_in, *lidar_cloud_in_, indices);
  LOG(INFO) << __FUNCTION__ << std::fixed << std::setprecision(6)
            << "::stamp:" << timestamp_
            << ", pts'num:" << lidar_cloud_in_->size();

  projectPointCloud();
  groundRemoval();
  cloudSegmentation();
  divideSurfEdge();
  LOG(INFO) << __func__ << std::fixed << std::setprecision(6)
            << "::cloud-segmentation elapsed time:" << elapsed_time.end()
            << " [ms]";
}

void CloudSegmentation::projectPointCloud() {
  const auto& config = LoamConfig::GetInstance().config;
  const std::size_t cloud_size = lidar_cloud_in_->size();
  for (size_t i = 0; i < cloud_size; ++i) {
    PointT pt = lidar_cloud_in_->points[i];
    // float range =
    //     std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2), std::pow(pt.z, 2));
    float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (range < 0.1) continue;

    float vertical_angle = std::asin(pt.z / range);
    int row_idx = (vertical_angle - config.vertical_angle_bottom * DEG_TO_RAD) /
                  angle_res_y_;
    if (row_idx < 0 || row_idx >= config.num_scans) continue;
    float horizontal_angle = std::atan2(pt.x, pt.y);
    int col_idx = -std::round((horizontal_angle - M_PI_2) / angle_res_x_) +
                  config.num_horizontal_scans * 0.5;
    if (col_idx >= config.num_horizontal_scans) {
      col_idx -= config.num_horizontal_scans;
    }
    if (col_idx < 0 || col_idx >= config.num_horizontal_scans) continue;
    range_mat_(row_idx, col_idx) = range;
    pt.intensity =
        static_cast<float>(row_idx) + static_cast<float>(col_idx) / 10000.0;
    size_t index = col_idx + row_idx * config.num_horizontal_scans;

    full_cloud_->points[index] = pt;
    pt.intensity = range;
    full_cloud_with_range_->points[index] = pt;
  }
}
void CloudSegmentation::groundRemoval() {
  const auto& config = LoamConfig::GetInstance().config;
  for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
    for (size_t i = 0; i < config.ground_scan_index; ++i) {
      size_t lower_idx = j + i * config.num_horizontal_scans;
      size_t upper_idx = j + (i + 1) * config.num_horizontal_scans;
      if (full_cloud_with_range_->points[lower_idx].intensity == -1 ||
          full_cloud_with_range_->points[upper_idx].intensity == -1) {
        ground_mat_(i, j) = -1;
        continue;
      }

      PointT pt_lower = full_cloud_->points[lower_idx];
      PointT pt_upper = full_cloud_->points[upper_idx];
      float dx = pt_upper.x - pt_lower.x;
      float dy = pt_upper.y - pt_lower.y;
      float dz = pt_upper.z - pt_lower.z;
      float vertical_angle =
          std::atan2(dz, std::sqrt(dx * dx + dy * dy + dz * dz));
      // sensorMountAngle:0.0,
      // 地面点的简单判别，通过查询相邻环上的两个点俯仰角是否超过阈值(10°)来判定。
      if (std::fabs(vertical_angle - config.sensor_mount_angle) <=
          config.surf_segmentation_threshold * DEG_TO_RAD) {
        ground_mat_(i, j) = 1;
        ground_mat_(i + 1, j) = 1;
      }
    }
  }

  for (size_t i = 0; i < config.num_scans; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (ground_mat_(i, j) == 1 || range_mat_(i, j) == FLT_MAX) {
        label_mat_(i, j) = -1;
      }
    }
  }

  for (size_t i = 0; i <= config.ground_scan_index; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (ground_mat_(i, j) == 1) {
        ground_cloud_->push_back(
            full_cloud_->points[j + i * config.num_horizontal_scans]);
      }
    }
  }
}

void CloudSegmentation::cloudSegmentation() {
  const auto& config = LoamConfig::GetInstance().config;

  for (size_t i = 0; i < config.num_scans; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (label_mat_(i, j) == 0) labelComponents(i, j);
    }
  }
  // extract segmented cloud for lidar odometry
  for (std::size_t i = 0; i < config.num_scans; ++i) {
    for (std::size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (label_mat_(i, j) > 0 || ground_mat_(i, j) == 1) {
        // outliers that will not be used for optimization(always continue)
        if (label_mat_(i, j) == FLT_MAX) {
          if (i > config.ground_scan_index && j % 5 == 0) {
            outlier_cloud_->push_back(
                full_cloud_->points[j + i * config.num_horizontal_scans]);
            continue;
          } else {
            continue;
          }
        }
        // majority of ground points are skipped
        if (ground_mat_(i, j) == 1) {
          if (j % 5 == 0 && j > 5 && j < config.num_horizontal_scans - 5)
            continue;
        }
        // save seg cloud
        auto pt = full_cloud_->points[j + i * config.num_horizontal_scans];
        segmented_cloud_with_ground_->push_back(pt);
      }
    }
  }
}

void CloudSegmentation::labelComponents(int row, int col) {
  const auto& config = LoamConfig::GetInstance().config;
  std::vector<bool> line_count_flag(config.num_horizontal_scans, false);
  const size_t cloud_size = config.num_horizontal_scans * config.num_scans;

  using Coord2D = Eigen::Vector2i;
  boost::circular_buffer<Coord2D> queue(cloud_size);
  boost::circular_buffer<Coord2D> all_pushed(cloud_size);
  queue.push_back({row, col});
  all_pushed.push_back({row, col});

  const Coord2D neighbor_iterator[4] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
  while (queue.size() > 0) {
    Coord2D from_ind = queue.front();
    queue.pop_front();
    // Mark popped point
    label_mat_(from_ind.x(), from_ind.y()) = label_count_;
    // Loop through all the neighboring grids of popped grid
    for (const auto iter : neighbor_iterator) {
      // new index
      int ind_x = from_ind.x() + iter.x();
      int ind_y = from_ind.y() + iter.y();
      // index should be within the boundary
      if (ind_x < 0 || ind_x >= config.num_scans) continue;
      // at range image margin(left or right side)
      if (ind_y < 0) ind_y = config.num_horizontal_scans - 1;
      if (ind_y >= config.num_horizontal_scans) ind_y = 0;

      // prevent infinite loop(caused by put already examined point back)
      if (label_mat_(ind_x, ind_y) != 0) continue;

      float d1 = std::max(range_mat_(from_ind.x(), from_ind.y()),
                          range_mat_(ind_x, ind_y));
      float d2 = std::min(range_mat_(from_ind.x(), from_ind.y()),
                          range_mat_(ind_x, ind_y));
      float alpha = iter.x() == 0 ? angle_res_x_ : angle_res_y_;

      float angle =
          std::atan2(d2 * std::sin(alpha), (d1 - d2 * std::cos(alpha)));

      if (angle > config.segment_theta * DEG_TO_RAD) {
        queue.push_back({ind_x, ind_y});
        label_mat_(ind_x, ind_y) = label_count_;
        line_count_flag[ind_x] = true;
        all_pushed.push_back({ind_x, ind_y});
      }
    }
  }
  // check if the segment is valid
  bool feasible_segment = false;
  if (all_pushed.size() >= config.segmented_cluster_num) {
    feasible_segment = true;
  } else if (all_pushed.size() >= config.segmented_valid_point_num) {
    // amount of points do not reach30, three lines with 5 points must be marked
    int line_count = 0;
    for (std::size_t i = 0; i < config.num_scans; ++i) {
      if (line_count_flag[i] == true) line_count++;
    }
    if (line_count >= config.segmented_valid_line_num) feasible_segment = true;
  }
  // segment is valid,mark these points
  if (feasible_segment == true) {
    label_count_++;
  } else {
    for (std::size_t i = 0; i < all_pushed.size(); ++i) {
      label_mat_(all_pushed[i].x(), all_pushed[i].y()) = FLT_MAX;
    }
  }
}

void CloudSegmentation::divideSurfEdge() {
  const auto& config = LoamConfig::GetInstance().config;
  removeClosedPointCloud(*segmented_cloud_with_ground_,
                         *segmented_cloud_with_ground_, config.minimum_range);

  int cloud_size = segmented_cloud_with_ground_->size();

  PointT first_pt = segmented_cloud_with_ground_->points[0];
  PointT last_pt = segmented_cloud_with_ground_->points[cloud_size - 1];
  float start_azimuth = -std::atan2(first_pt.y, first_pt.x);
  float end_azimuth = -std::atan2(last_pt.y, last_pt.x) + M_PI * 2;

  if (end_azimuth - start_azimuth > 3 * M_PI) {
    end_azimuth -= 2 * M_PI;
  } else if (end_azimuth - start_azimuth < M_PI) {
    end_azimuth += 2 * M_PI;
  }
  bool half_passed = false;
  int count = cloud_size;
  int num_scans = config.num_scans;
  std::vector<pcl::PointCloud<PointT>> laser_cloud_scans(num_scans);
  for (std::size_t i = 0; i < cloud_size; ++i) {
    auto pt = segmented_cloud_with_ground_->points[i];
    float range = std::sqrt(pt.x * pt.x + pt.y * pt.y);
    float vertical_angle = std::atan(pt.z / range) * RAD_TO_DEG;

    int scan_id = 0;
    if (num_scans == 16) {
      scan_id = static_cast<int>(
          (vertical_angle - config.vertical_angle_bottom) / 2 + 0.5);

    } else if (num_scans == 32) {
      scan_id = static_cast<int>((vertical_angle + 92.0 / 3) * 3 / 4.0);
    } else if (num_scans == 64) {
      scan_id = vertical_angle >= -8.83
                    ? static_cast<int>((2 - vertical_angle) * 3 + 0.5)
                    : num_scans / 2 +
                          static_cast<int>((-8.83 - vertical_angle) * 2 + 0.5);
    } else {
      LOG(FATAL) << "wrong scan number!";
    }

    if (num_scans == 16 || num_scans == 32) {
      if (scan_id > num_scans - 1 || scan_id < 0) {
        count--;
        continue;
      }
    }

    float azimuth = -std::atan2(pt.y, pt.x);
    if (!half_passed) {
      if (azimuth < start_azimuth - M_PI / 2) {
        azimuth += 2 * M_PI;
      } else if (azimuth > start_azimuth + M_PI * 3 / 2) {
        azimuth -= 2 * M_PI;
      }

      if (azimuth - start_azimuth > M_PI) half_passed = true;
    } else {
      azimuth += 2 * M_PI;
      if (azimuth < end_azimuth - M_PI * 3 / 2) {
        azimuth += 2 * M_PI;
      } else if (azimuth > end_azimuth + M_PI / 2) {
        azimuth -= 2 * M_PI;
      }
    }
    float stamp = (azimuth - start_azimuth) / (end_azimuth - start_azimuth);
    pt.intensity = scan_id + config.scan_period * stamp;
    laser_cloud_scans[scan_id].push_back(pt);
  }

  cloud_size = count;
  std::vector<int> scan_start_idx(num_scans, 0);
  std::vector<int> scan_end_idx(num_scans, 0);

  /*
  using sidx = scan_start_idx
  using eidx = scan_end_idx
      sidx      eidx
  1-----|--****--|------, ...... ,16-----|*****|------
  */
  PointCloudPtr laser_cloud(new pcl::PointCloud<PointT>());
  PointCloudPtr corner_cloud_sharp_(new pcl::PointCloud<PointT>());
  PointCloudPtr corner_cloud_lesssharp_(new pcl::PointCloud<PointT>());
  PointCloudPtr surf_cloud_flat_(new pcl::PointCloud<PointT>());
  PointCloudPtr surf_cloud_lessflat_(new pcl::PointCloud<PointT>());

  for (std::size_t i = 0; i < num_scans; ++i) {
    scan_start_idx[i] = laser_cloud->size() + 5;
    *laser_cloud += laser_cloud_scans[i];
    scan_end_idx[i] = laser_cloud->size() - 6;
  }

  auto pts = laser_cloud->points;
  for (std::size_t i = 0; i < cloud_size - 5; ++i) {
    float diff_x = pts[i - 5].x + pts[i - 4].x + pts[i - 3].x + pts[i - 2].x +
                   pts[i - 1].x - 10 * pts[i].x + pts[i + 1].x + pts[i + 2].x +
                   pts[i + 3].x + pts[i + 4].x + pts[i + 5].x;
    float diff_y = pts[i - 5].y + pts[i - 4].y + pts[i - 3].y + pts[i - 2].y +
                   pts[i - 1].y - 10 * pts[i].y + pts[i + 1].y + pts[i + 2].y +
                   pts[i + 3].y + pts[i + 4].y + pts[i + 5].y;
    float diff_z = pts[i - 5].z + pts[i - 4].z + pts[i - 3].z + pts[i - 2].z +
                   pts[i - 1].z - 10 * pts[i].z + pts[i + 1].z + pts[i + 2].z +
                   pts[i + 3].z + pts[i + 4].z + pts[i + 5].z;
    float diff_range = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

    cloud_curvature_.push_back(diff_range);
    cloud_sort_idx_.push_back(i);
    cloud_neighbor_picked_.push_back(0);
    cloud_label_.push_back(0);
  }

  for (std::size_t i = 0; i < num_scans; ++i) {
    int cur_diff_idx = scan_end_idx[i] - scan_start_idx[i];
    if (cur_diff_idx < 6) continue;
    PointCloudPtr surf_cloud_lessflat_scan(new pcl::PointCloud<PointT>());
    for (std::size_t j = 0; j < 6; ++j) {
      int sp = scan_start_idx[i] + cur_diff_idx * j / 6;
      int ep = scan_start_idx[i] + cur_diff_idx * (j + 1) / 6 - 1;
      std::sort(cloud_sort_idx_.begin() + sp, cloud_sort_idx_.begin() + ep + 1,
                [this](const int& lv, const int& rv) {
                  return cloud_curvature_[lv] < cloud_curvature_[rv];
                });

      // extract corner features from every segment of scanline
      int largest_picked_num = 0;
      for (std::size_t k = ep; k >= sp; k--) {
        int id = cloud_sort_idx_[k];
        if (cloud_neighbor_picked_[id] == 0 && cloud_curvature_[id] > 0.1) {
          largest_picked_num++;
          if (largest_picked_num <= 2) {
            cloud_label_[id] = 2;
            corner_cloud_sharp_->push_back(pts[id]);
            corner_cloud_lesssharp_->push_back(pts[id]);
          } else if (largest_picked_num <= 20) {
            cloud_label_[id] = 1;
            corner_cloud_lesssharp_->push_back(pts[id]);
          } else {
            break;
          }

          cloud_neighbor_picked_[id] = 1;
          for (std::size_t l = 1; l <= 5; l++) {
            float diff_x = pts[id + l].x - pts[id + l - 1].x;
            float diff_y = pts[id + l].y - pts[id + l - 1].y;
            float diff_z = pts[id + l].z - pts[id + l - 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05) {
              break;
            }
            cloud_neighbor_picked_[id + l] = 1;
          }
          for (std::size_t l = -1; l >= -5; l--) {
            float diff_x = pts[id + l].x - pts[id + l + 1].x;
            float diff_y = pts[id + l].y - pts[id + l + 1].y;
            float diff_z = pts[id + l].z - pts[id + l + 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05) {
              break;
            }
            cloud_neighbor_picked_[id + l] = 1;
          }
        }
      }
      // extract flat surface features from every segment of scanline
      int smallest_picked_num = 0;
      for (std::size_t k = sp; k <= ep; ++k) {
        int id = cloud_sort_idx_[k];

        if (cloud_neighbor_picked_[id] == 0 && cloud_curvature_[id] < 0.1) {
          cloud_label_[id] = -1;
          surf_cloud_flat_->push_back(pts[id]);
          smallest_picked_num++;
          if (smallest_picked_num >= 4) {
            break;
          }

          cloud_neighbor_picked_[id] = 1;
          for (std::size_t l = 1; l <= 5; l++) {
            float diff_x = pts[id + l].x - pts[id + l - 1].x;
            float diff_y = pts[id + l].y - pts[id + l - 1].y;
            float diff_z = pts[id + l].z - pts[id + l - 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05) {
              break;
            }
            cloud_neighbor_picked_[id + l] = 1;
          }
          for (std::size_t l = -1; l >= -5; l--) {
            float diff_x = pts[id + l].x - pts[id + l + 1].x;
            float diff_y = pts[id + l].y - pts[id + l + 1].y;
            float diff_z = pts[id + l].z - pts[id + l + 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05) {
              break;
            }
            cloud_neighbor_picked_[id + l] = 1;
          }
        }
      }
      // extract less flat surface features
      for (std::size_t k = sp; k <= ep; k++) {
        if (cloud_label_[k] <= 0) {
          surf_cloud_lessflat_scan->push_back(pts[k]);
        }
      }
    }

    PointCloudPtr surf_cloud_lessflat_scan_ds(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> downsample_filter;
    downsample_filter.setInputCloud(surf_cloud_lessflat_scan);
    downsample_filter.setLeafSize(0.2, 0.2, 0.2);
    downsample_filter.filter(*surf_cloud_lessflat_scan_ds);
    *surf_cloud_lessflat_ += *surf_cloud_lessflat_scan_ds;
  }

  // inject msgs to datacenter
  auto& data_center = DataCenter::Instance();
  ScanSegMsgPtr segmsg(new ScanSegMsg(
      timestamp_, laser_cloud, corner_cloud_sharp_, corner_cloud_lesssharp_,
      surf_cloud_flat_, surf_cloud_lessflat_));
  data_center.SetScanSegMsg(segmsg);
}

template <typename T>
void CloudSegmentation::removeClosedPointCloud(
    const typename pcl::PointCloud<T>& cloud_in,
    typename pcl::PointCloud<T>& cloud_out, float threshold) {
  if (&cloud_in != &cloud_out) {
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());
  }

  std::size_t j = 0;
  for (std::size_t i = 0; i < cloud_in.points.size(); ++i) {
    PointT pt = cloud_in.points[i];
    auto range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (range < threshold) continue;
    cloud_out.points[i] = cloud_in.points[i];
    j++;
  }
  if (j != cloud_in.points.size()) {
    cloud_out.points.resize(j);
  }

  cloud_out.height = 1;
  cloud_out.width = static_cast<std::uint32_t>(j);
  cloud_out.is_dense = true;
}
template void CloudSegmentation::removeClosedPointCloud<pcl::PointXYZI>(
    const typename pcl::PointCloud<pcl::PointXYZI>& cloud_in,
    typename pcl::PointCloud<pcl::PointXYZI>& cloud_out, float threshold);

}  // namespace ceres_loam