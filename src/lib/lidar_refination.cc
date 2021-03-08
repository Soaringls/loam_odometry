#include "lidar_refination.h"
namespace ceres_loam {

LidarRefination::LidarRefination() {
  config_ = LoamConfig::GetInstance().config;
  laser_cloud_num_ =
      config_.cloud_width * config_.cloud_height * config_.cloud_depth;
  allocateMemory();
  double opti_params[7] = {0, 0, 0, 1, 0, 0, 0};
  memcpy(parameters, opti_params, sizeof(opti_params));

  correct_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  correct_t_ = Eigen::Vector3d(0, 0, 0);

  voxel_filter_corner_.setLeafSize(config_.corner_voxel_fiter_size,
                                   config_.corner_voxel_fiter_size,
                                   config_.corner_voxel_fiter_size);
  voxel_filter_surf_.setLeafSize(config_.surf_voxel_fiter_size,
                                 config_.surf_voxel_fiter_size,
                                 config_.surf_voxel_fiter_size);

  // test odom_pose, correct_pose, refined_pose;
  odom_pose = Eigen::Affine3d::Identity();
  correct_pose = Eigen::Affine3d::Identity();
  refined_pose = Eigen::Affine3d::Identity();
  LOG(INFO) << __FUNCTION__ << ":construct success!";
}

void LidarRefination::allocateMemory() {
  feature_scan_corner_.reset(new pcl::PointCloud<PointT>());
  feature_scan_corner_filtered_.reset(new pcl::PointCloud<PointT>());
  feature_scan_surf_.reset(new pcl::PointCloud<PointT>());
  feature_scan_surf_filtered_.reset(new pcl::PointCloud<PointT>());
  cloud_full_res.reset(new pcl::PointCloud<PointT>());

  corner_map_.reset(new pcl::PointCloud<PointT>());
  surf_map_.reset(new pcl::PointCloud<PointT>());
  surround_map_.reset(new pcl::PointCloud<PointT>());

  for (int i = 0; i < laser_cloud_num_; i++) {  // 4851
    PointCloudPtr corner_tmp(new pcl::PointCloud<PointT>());
    PointCloudPtr surf_tmp(new pcl::PointCloud<PointT>());
    corners_pool.push_back(corner_tmp);
    surfs_pool.push_back(surf_tmp);
  }

  // kd-tree
  kdtree_corner_map_.reset(new pcl::KdTreeFLANN<PointT>());
  kdtree_surf_map_.reset(new pcl::KdTreeFLANN<PointT>());
  globis_map.reset(new pcl::PointCloud<PointT>());
}

void LidarRefination::pointAssociateToMap(PointT const *const pi,
                                          PointT *const po) {
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = q_w_curr_ * point_curr + t_w_curr_;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
}

void LidarRefination::accessData() {
  auto msgs = DataCenter::Instance().GetScanSegMsg();
  feature_scan_corner_->clear();
  feature_scan_corner_ = msgs->cornerPointsLessSharp;

  feature_scan_surf_->clear();
  feature_scan_surf_ = msgs->surfPointsLessFlat;

  cloud_full_res->clear();
  cloud_full_res = msgs->laserCloud;
  odom_t_ = msgs->odom_to_init_t;
  odom_q_ = msgs->odom_to_init_q;
}

void LidarRefination::accessAvailableCubicNum(int &scans_valid_num,
                                              int &sub_scan_valid_num) {
  // LOG(INFO) << __FUNCTION__ << ":config_.cloud_cen_width-H-D1:" <<
  // config_.cloud_cen_width
  //           << ", " << config_.cloud_cen_height << ", "
  //           << config_.cloud_cen_depth;  // 10 10 5
  int centerCubeI =
      int((t_w_curr_.x() + config_.cube_size / 2) / config_.cube_size) +
      config_.cloud_cen_width;  // init 10
  int centerCubeJ =
      int((t_w_curr_.y() + config_.cube_size / 2) / config_.cube_size) +
      config_.cloud_cen_height;  // init 10
  int centerCubeK =
      int((t_w_curr_.z() + config_.cube_size / 2) / config_.cube_size) +
      config_.cloud_cen_depth;  // init 5
  // LOG(INFO) << __FUNCTION__ << ":centerCubeI-J-K-1:" << centerCubeI << ", "
  // << centerCubeJ << ", " << centerCubeK;
  if (t_w_curr_.x() + config_.cube_size / 2 < 0) centerCubeI--;
  if (t_w_curr_.y() + config_.cube_size / 2 < 0) centerCubeJ--;
  if (t_w_curr_.z() + config_.cube_size / 2 < 0) centerCubeK--;
  // LOG(INFO) << __FUNCTION__ << ":centerCubeI-J-K-2:" << centerCubeI << ", "
  //           << centerCubeJ << ", " << centerCubeK;
  while (centerCubeI < 3) {
    for (int j = 0; j < config_.cloud_height; j++) {
      for (int k = 0; k < config_.cloud_depth; k++) {
        int i = config_.cloud_width - 1;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; i >= 1; i--) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i - 1 + config_.cloud_width * j +
                           config_.cloud_width * config_.cloud_height * k];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i - 1 + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        laserCloudCubeCornerPointer->clear();

        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeSurfPointer->clear();
      }
    }
    centerCubeI++;
    config_.cloud_cen_width++;
  }
  // LOG(INFO) << __FUNCTION__ << ":centerCubeI-J-K-3:" << centerCubeI << ", "
  //           << centerCubeJ << ", " << centerCubeK;

  while (centerCubeI >= config_.cloud_width - 3) {
    for (int j = 0; j < config_.cloud_height; j++) {
      for (int k = 0; k < config_.cloud_depth; k++) {
        int i = 0;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; i < config_.cloud_width - 1; i++) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i + 1 + config_.cloud_width * j +
                           config_.cloud_width * config_.cloud_height * k];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i + 1 + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
      }
    }

    centerCubeI--;
    config_.cloud_cen_width--;
  }
  // LOG(INFO) << __FUNCTION__ << ":centerCubeI-J-K-4:" << centerCubeI << ", "
  //           << centerCubeJ << ", " << centerCubeK;
  while (centerCubeJ < 3) {
    for (int i = 0; i < config_.cloud_width; i++) {
      for (int k = 0; k < config_.cloud_depth; k++) {
        int j = config_.cloud_height - 1;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; j >= 1; j--) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i + config_.cloud_width * (j - 1) +
                           config_.cloud_width * config_.cloud_height * k];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i + config_.cloud_width * (j - 1) +
                         config_.cloud_width * config_.cloud_height * k];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
      }
    }

    centerCubeJ++;
    config_.cloud_cen_height++;
  }

  while (centerCubeJ >= config_.cloud_height - 3) {
    for (int i = 0; i < config_.cloud_width; i++) {
      for (int k = 0; k < config_.cloud_depth; k++) {
        int j = 0;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; j < config_.cloud_height - 1; j++) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i + config_.cloud_width * (j + 1) +
                           config_.cloud_width * config_.cloud_height * k];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i + config_.cloud_width * (j + 1) +
                         config_.cloud_width * config_.cloud_height * k];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
      }
    }

    centerCubeJ--;
    config_.cloud_cen_height--;
  }

  while (centerCubeK < 3) {
    for (int i = 0; i < config_.cloud_width; i++) {
      for (int j = 0; j < config_.cloud_height; j++) {
        int k = config_.cloud_depth - 1;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; k >= 1; k--) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i + config_.cloud_width * j +
                           config_.cloud_width * config_.cloud_height *
                               (k - 1)];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * (k - 1)];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
      }
    }

    centerCubeK++;
    config_.cloud_cen_depth++;
  }

  while (centerCubeK >= config_.cloud_depth - 3) {
    for (int i = 0; i < config_.cloud_width; i++) {
      for (int j = 0; j < config_.cloud_height; j++) {
        int k = 0;
        PointCloudPtr laserCloudCubeCornerPointer =
            corners_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * k];
        PointCloudPtr laserCloudCubeSurfPointer =
            surfs_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k];
        for (; k < config_.cloud_depth - 1; k++) {
          corners_pool[i + config_.cloud_width * j +
                       config_.cloud_width * config_.cloud_height * k] =
              corners_pool[i + config_.cloud_width * j +
                           config_.cloud_width * config_.cloud_height *
                               (k + 1)];
          surfs_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
              surfs_pool[i + config_.cloud_width * j +
                         config_.cloud_width * config_.cloud_height * (k + 1)];
        }
        corners_pool[i + config_.cloud_width * j +
                     config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeCornerPointer;
        surfs_pool[i + config_.cloud_width * j +
                   config_.cloud_width * config_.cloud_height * k] =
            laserCloudCubeSurfPointer;
        laserCloudCubeCornerPointer->clear();
        laserCloudCubeSurfPointer->clear();
      }
    }

    centerCubeK--;
    config_.cloud_cen_depth--;
  }

  // LOG(INFO) << __FUNCTION__
  //           << ":accessAvailableCubicNum:config_.cloud_cen_width-H-D2:"
  //           << config_.cloud_cen_width << ", " << config_.cloud_cen_height <<
  //           ", "
  //           << config_.cloud_cen_depth;
  // LOG(INFO) << __FUNCTION__ << ":centerCubeI-J-K-5:" << centerCubeI << ", "
  //           << centerCubeJ << ", " << centerCubeK;

  // for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
  //   for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++) {
  //     for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {
  for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++) {
    for (int j = centerCubeJ - 1; j <= centerCubeJ + 1; j++) {
      for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++) {
        // 21 21 11
        if (i >= 0 && i < config_.cloud_width && j >= 0 &&
            j < config_.cloud_height && k >= 0 && k < config_.cloud_depth) {
          scans_valid_indices[scans_valid_num] =
              i + config_.cloud_width * j +
              config_.cloud_width * config_.cloud_height * k;
          scans_valid_num++;
          subscans_valid_indices[sub_scan_valid_num] =
              i + config_.cloud_width * j +
              config_.cloud_width * config_.cloud_height * k;
          sub_scan_valid_num++;
        }
      }
    }
  }
}

void LidarRefination::Process() {
  Timer elapsed_time;
  accessData();
  // set initial guess
  setInitialGuess();
  // prepare data
  int scans_valid_num = 0;
  int subscans_valid_num = 0;
  accessAvailableCubicNum(scans_valid_num, subscans_valid_num);
  LOG(INFO) << __FUNCTION__ << ":scans valid num:" << scans_valid_num
            << ", subscans valid num:" << subscans_valid_num;
  prepareData(subscans_valid_num);
  // calculate refination result
  calculateTransformation();
  // update good pose
  updateOptimizedResult();

  // add feature scan to pool
  // todo: add feature pts by fix distance or angle-threshold
  static int cnt(0);
  if (cnt++ % 30 == 0) {
    addFeatureCloudtoPool();
    downSampleCornerSurfArray(scans_valid_num);
  }

  LOG(INFO) << __FUNCTION__ << std::fixed << std::setprecision(6)
            << ":laser-refination elapsed time:" << elapsed_time.end()
            << " [ms]";
}

void LidarRefination::prepareData(const int featuremap_scans) {
  // update map_corner and map_surf
  corner_map_->clear();
  surf_map_->clear();
  // generate regis-msp, use subscans_valid_num or scans_valid_num
  for (int i = 0; i < featuremap_scans; i++) {
    int ind = subscans_valid_indices[i];
    *corner_map_ += *(corners_pool[ind]);
    *surf_map_ += *(surfs_pool[ind]);
  }

  feature_scan_corner_filtered_->clear();
  voxel_filter_corner_.setInputCloud(feature_scan_corner_);
  voxel_filter_corner_.filter(*feature_scan_corner_filtered_);

  feature_scan_surf_filtered_->clear();
  voxel_filter_surf_.setInputCloud(feature_scan_surf_);
  voxel_filter_surf_.filter(*feature_scan_surf_filtered_);

  LOG(INFO) << __FUNCTION__ << ":regis to cornermap:" << corner_map_->size()
            << "->" << feature_scan_corner_filtered_->points.size();
  LOG(INFO) << __FUNCTION__ << ":regis to surfmap  :" << surf_map_->size()
            << "->" << feature_scan_surf_filtered_->points.size();
}

void LidarRefination::calculateTransformation() {
  int corner_map_num = corner_map_->points.size();
  int surf_map_num = surf_map_->points.size();

  if (corner_map_num > 10 && surf_map_num > 50) {
    Timer t_opt;
    Timer t_tree;
    kdtree_corner_map_->setInputCloud(corner_map_);
    kdtree_surf_map_->setInputCloud(surf_map_);
    LOG(INFO) << __FUNCTION__ << ":build kdtree time:" << t_tree.end()
              << " [ms]";

    for (int iterCount = 0; iterCount < 2; iterCount++) {
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);

      ceres::LocalParameterization *q_parameterization =
          new ceres::EigenQuaternionParameterization();
      problem.AddParameterBlock(parameters, 4, q_parameterization);
      problem.AddParameterBlock(parameters + 4, 3);

      //   Timer t_data;
      calculateTransformationCorner(problem, loss_function);
      calculateTransformationSurf(problem, loss_function);
      Timer t_solver;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.max_num_iterations = 4;
      options.minimizer_progress_to_stdout = false;
      options.check_gradients = false;
      options.gradient_check_relative_precision = 1e-4;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      LOG(INFO) << __FUNCTION__ << ":refination solver time:" << t_solver.end()
                << " [ms]";
    }

    LOG(INFO) << __FUNCTION__ << ":refination optimization time:" << t_opt.end()
              << " [ms]";
  } else {
    LOG(INFO) << __FUNCTION__ << ":time Map corner and surf num are not enough";
  }
}

void LidarRefination::setInitialGuess() {
  // initial guess for scan2map optimization
  q_w_curr_ = correct_q_ * odom_q_;
  t_w_curr_ = correct_q_ * odom_t_ + correct_t_;

  // test correct my version
  odom_pose = Eigen::Translation3d(odom_t_.x(), odom_t_.y(), odom_t_.z()) *
              odom_q_.normalized();
  refined_pose = odom_pose * correct_pose;

  // LOG(INFO) << std::fixed << std::setprecision(6)
  //           << "t_w_curr_:" << t_w_curr_.x() << ", " << t_w_curr_.y() << ", "
  //           << t_w_curr_.z();
  // LogOutputAffine3dPose("odom_pose", odom_pose);
  // LogOutputAffine3dPose("correct_pose", correct_pose);
  // LogOutputAffine3dPose("refined_pose", refined_pose);
}

void LidarRefination::updateOptimizedResult() {
  // refination global pose, good result
  t_w_curr_ = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
  q_w_curr_ = Eigen::Quaterniond(parameters[3], parameters[0], parameters[1],
                                 parameters[2]);
  // correct transform to optimized the global pose of lidar-odom
  correct_q_ = q_w_curr_ * odom_q_.inverse();
  correct_t_ = t_w_curr_ - correct_q_ * odom_t_;

  // test correct my version
  refined_pose =
      Eigen::Translation3d(t_w_curr_.x(), t_w_curr_.y(), t_w_curr_.z()) *
      q_w_curr_.normalized();
  correct_pose = odom_pose.inverse() * refined_pose;
  // test end

  auto &data_center = DataCenter::Instance();
  data_center.SetOdomCorrectPose(correct_t_, correct_q_);
}

void LidarRefination::calculateTransformationCorner(
    ceres::Problem &problem, ceres::LossFunction *loss_function) {
  int scan_corner_num = feature_scan_corner_filtered_->points.size();
  PointT point_laser, point_globis;
  for (int i = 0; i < scan_corner_num; i++) {
    point_laser = feature_scan_corner_filtered_->points[i];
    pointAssociateToMap(&point_laser, &point_globis);
    kdtree_corner_map_->nearestKSearch(point_globis, 5, neighbor_pts_indices,
                                       neighbor_pts_dist);

    if (neighbor_pts_dist[4] < 1.0) {
      // PCA分析
      std::vector<Eigen::Vector3d> nearCorners;
      Eigen::Vector3d center(0, 0, 0);
      for (int j = 0; j < 5; j++) {
        Eigen::Vector3d tmp(corner_map_->points[neighbor_pts_indices[j]].x,
                            corner_map_->points[neighbor_pts_indices[j]].y,
                            corner_map_->points[neighbor_pts_indices[j]].z);
        center = center + tmp;
        nearCorners.push_back(tmp);
      }
      center = center / 5.0;

      Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
      for (int j = 0; j < 5; j++) {
        Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
        covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
      }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

      // if is indeed line feature
      // note Eigen library sort eigenvalues in increasing order
      Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
      Eigen::Vector3d curr_point(point_laser.x, point_laser.y, point_laser.z);
      if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
        Eigen::Vector3d point_on_line = center;
        Eigen::Vector3d point_a, point_b;
        point_a = 0.1 * unit_direction + point_on_line;
        point_b = -0.1 * unit_direction + point_on_line;

        ceres::CostFunction *cost_function =
            LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
        problem.AddResidualBlock(cost_function, loss_function, parameters,
                                 parameters + 4);
      }
    }
  }
}

void LidarRefination::calculateTransformationSurf(
    ceres::Problem &problem, ceres::LossFunction *loss_function) {
  int scan_surf_num = feature_scan_surf_filtered_->points.size();
  PointT point_laser, point_globis;
  for (int i = 0; i < scan_surf_num; i++) {
    point_laser = feature_scan_surf_filtered_->points[i];

    pointAssociateToMap(&point_laser, &point_globis);
    kdtree_surf_map_->nearestKSearch(point_globis, 5, neighbor_pts_indices,
                                     neighbor_pts_dist);

    Eigen::Matrix<double, 5, 3> matA0;
    Eigen::Matrix<double, 5, 1> matB0 =
        -1 * Eigen::Matrix<double, 5, 1>::Ones();
    if (neighbor_pts_dist[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) = surf_map_->points[neighbor_pts_indices[j]].x;
        matA0(j, 1) = surf_map_->points[neighbor_pts_indices[j]].y;
        matA0(j, 2) = surf_map_->points[neighbor_pts_indices[j]].z;
      }
      // find the norm of plane
      Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
      double negative_OA_dot_norm = 1 / norm.norm();
      norm.normalize();

      // Here n(pa, pb, pc) is unit norm of plane
      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        // if OX * n > 0.2, then plane is not fit well
        if (fabs(norm(0) * surf_map_->points[neighbor_pts_indices[j]].x +
                 norm(1) * surf_map_->points[neighbor_pts_indices[j]].y +
                 norm(2) * surf_map_->points[neighbor_pts_indices[j]].z +
                 negative_OA_dot_norm) > 0.2) {
          planeValid = false;
          break;
        }
      }
      Eigen::Vector3d curr_point(point_laser.x, point_laser.y, point_laser.z);
      if (planeValid) {
        ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(
            curr_point, norm, negative_OA_dot_norm);
        problem.AddResidualBlock(cost_function, loss_function, parameters,
                                 parameters + 4);
      }
    }
  }
}

void LidarRefination::addFeatureCloudtoPool() {
  LOG(INFO) << __FUNCTION__
            << ":addFeatureCloudtoPool:config_.cloud_cen_width-H-D:"
            << config_.cloud_cen_width << ", " << config_.cloud_cen_height
            << ", " << config_.cloud_cen_depth;  // 10 10 5

  // add new points for corners_pool
  PointT point_globis;
  int num_pts = feature_scan_corner_filtered_->points.size();
  for (int i = 0; i < num_pts; i++) {
    // map from lidar-frame to global frame
    pointAssociateToMap(&feature_scan_corner_filtered_->points[i],
                        &point_globis);

    int cubeI =
        int((point_globis.x + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_width;  // 10
    int cubeJ =
        int((point_globis.y + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_height;  // 10
    int cubeK =
        int((point_globis.z + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_depth;  // 5

    if (point_globis.x + config_.cube_size / 2 < 0) cubeI--;
    if (point_globis.y + config_.cube_size / 2 < 0) cubeJ--;
    if (point_globis.z + config_.cube_size / 2 < 0) cubeK--;

    // 21 21 11
    if (cubeI >= 0 && cubeI < config_.cloud_width && cubeJ >= 0 &&
        cubeJ < config_.cloud_height && cubeK >= 0 &&
        cubeK < config_.cloud_depth) {
      int cubeInd = cubeI + config_.cloud_width * cubeJ +
                    config_.cloud_width * config_.cloud_height * cubeK;
      corners_pool[cubeInd]->push_back(point_globis);
    } else {
      LOG(INFO) << __FUNCTION__ << "\033[1;32m---->cubeI-J-K:" << cubeI << ", "
                << cubeJ << ", " << cubeK << " .\033[0m";
    }
  }
  // add new points for surfs_pool
  num_pts = feature_scan_surf_filtered_->points.size();
  for (int i = 0; i < num_pts; i++) {
    pointAssociateToMap(&feature_scan_surf_filtered_->points[i], &point_globis);

    int cubeI =
        int((point_globis.x + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_width;
    int cubeJ =
        int((point_globis.y + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_height;
    int cubeK =
        int((point_globis.z + config_.cube_size / 2) / config_.cube_size) +
        config_.cloud_cen_depth;

    if (point_globis.x + config_.cube_size / 2 < 0) cubeI--;
    if (point_globis.y + config_.cube_size / 2 < 0) cubeJ--;
    if (point_globis.z + config_.cube_size / 2 < 0) cubeK--;

    if (cubeI >= 0 && cubeI < config_.cloud_width && cubeJ >= 0 &&
        cubeJ < config_.cloud_height && cubeK >= 0 &&
        cubeK < config_.cloud_depth) {
      int cubeInd = cubeI + config_.cloud_width * cubeJ +
                    config_.cloud_width * config_.cloud_height * cubeK;
      surfs_pool[cubeInd]->push_back(point_globis);
    }
  }
}

void LidarRefination::downSampleCornerSurfArray(const int scans_valid_num) {
  for (int i = 0; i < scans_valid_num; i++) {
    int ind = scans_valid_indices[i];

    PointCloudPtr tmp_corner(new pcl::PointCloud<PointT>());
    voxel_filter_corner_.setInputCloud(corners_pool[ind]);
    voxel_filter_corner_.filter(*tmp_corner);
    corners_pool[ind] = tmp_corner;

    PointCloudPtr tmp_surf(new pcl::PointCloud<PointT>());
    voxel_filter_surf_.setInputCloud(surfs_pool[ind]);
    voxel_filter_surf_.filter(*tmp_surf);
    surfs_pool[ind] = tmp_surf;
  }
}

const PointCloudPtr LidarRefination::GenerateWholeMap() {
  static int frame_cnt(0);
  if (frame_cnt++ % 20 == 0) {
    globis_map->clear();
    for (int i = 0; i < laser_cloud_num_; i++) {
      *globis_map += *corners_pool[i];
      *globis_map += *surfs_pool[i];
    }
  }
  LOG(INFO) << __FUNCTION__ << ":global map's size:" << globis_map->size();
  return globis_map;
}

const PointCloudPtr LidarRefination::GenerateSurroundMap() {
  surround_map_->clear();
  *surround_map_ += *corner_map_;
  *surround_map_ += *surf_map_;
  return surround_map_;
}

void LogOutputAffine3dPose(const std::string &msgs,
                           const Eigen::Affine3d pose) {
  Eigen::Translation3d t(pose.translation());
  Eigen::Quaterniond q(pose.linear());
  LOG(INFO) << __func__ << ": " << std::fixed << std::setprecision(6) << msgs
            << "'t :" << t.x() << ", " << t.y() << ", " << t.z()
            << ", and qwxyz:" << q.w() << ", " << q.x() << ", " << q.y() << ", "
            << q.z();
}
}  // namespace ceres_loam
