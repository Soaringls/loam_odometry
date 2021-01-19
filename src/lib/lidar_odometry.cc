#include "lidar_odometry.h"

namespace ceres_loam {
LidarOdometry::LidarOdometry() {
  config_ = LoamConfig::GetInstance().config;
  init_ = false;
  allocateMemory();
  double q_temp[4] = {0.0, 0.0, 0.0, 1.0};  // x y z w
  memcpy(para_q, q_temp, sizeof(q_temp));
  double t_temp[3] = {0.0, 0.0, 0.0};  // x y z w
  memcpy(para_t, t_temp, sizeof(t_temp));
  tf_q_ = Eigen::Map<Eigen::Quaterniond>(para_q);
  tf_t_ = Eigen::Map<Eigen::Vector3d>(para_t);

  // transformation from current frame to world frame
  odom_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  odom_t_ = Eigen::Vector3d(0, 0, 0);

  LOG(INFO) << __func__ << ":construct success!";
}
void LidarOdometry::allocateMemory() {
  kdtreeCornerLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
  kdtreeSurfLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

  laserCloudFullRes.reset(new pcl::PointCloud<PointT>());
  cornerPointsSharp.reset(new pcl::PointCloud<PointT>());
  cornerPointsLessSharp.reset(new pcl::PointCloud<PointT>());
  surfPointsFlat.reset(new pcl::PointCloud<PointT>());
  surfPointsLessFlat.reset(new pcl::PointCloud<PointT>());

  laserCloudCornerLast.reset(new pcl::PointCloud<PointT>());
  laserCloudSurfLast.reset(new pcl::PointCloud<PointT>());
}
void LidarOdometry::accessData() {
  auto seg_msgs = DataCenter::Instance().GetScanSegMsg();

  cornerPointsSharp->clear();  // cur edge
  cornerPointsSharp = seg_msgs->cornerPointsSharp;

  cornerPointsLessSharp->clear();  // last edge
  cornerPointsLessSharp = seg_msgs->cornerPointsLessSharp;

  surfPointsFlat->clear();  // cur planar
  surfPointsFlat = seg_msgs->surfPointsFlat;

  surfPointsLessFlat->clear();  // last planar
  surfPointsLessFlat = seg_msgs->surfPointsLessFlat;

  laserCloudFullRes->clear();
  laserCloudFullRes = seg_msgs->laserCloud;
}

// undistort lidar point
void LidarOdometry::transformToStart(PointT const *const pi, PointT *const po) {
  // interpolation ratio
  double s;
  if (DISTORTION)
    s = (pi->intensity - int(pi->intensity)) / config_.scan_period;
  else
    s = 1.0;
  // s = 1;
  Eigen::Quaterniond q_point_last =
      Eigen::Quaterniond::Identity().slerp(s, tf_q_);
  Eigen::Vector3d t_point_last = s * tf_t_;
  Eigen::Vector3d point(pi->x, pi->y, pi->z);
  Eigen::Vector3d un_point = q_point_last * point + t_point_last;

  po->x = un_point.x();
  po->y = un_point.y();
  po->z = un_point.z();
  po->intensity = pi->intensity;
}

void LidarOdometry::Process() {
  Timer elapsed_time;

  accessData();
  if (!init_) {
    init_ = true;
    LOG(INFO) << __func__ << ":init success!";
  } else {
    calculateTransformation();
  }
  laserCloudCornerLast->clear();
  *laserCloudCornerLast = *cornerPointsLessSharp;
  laserCloudSurfLast->clear();
  *laserCloudSurfLast = *surfPointsLessFlat;
  kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
  kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

  auto seg_msgs = DataCenter::Instance().GetScanSegMsg();
  seg_msgs->setPose(odom_t_, odom_q_);

  LOG(INFO) << __func__ << std::fixed << std::setprecision(6)
            << ":laser-odometry elapsed time:" << elapsed_time.end() << " [ms]";
}

void LidarOdometry::calculateTransformation() {
  Timer opti_time;
  for (std::size_t opti_counter = 0; opti_counter < 2; ++opti_counter) {
    Timer elapse_time;
    int corner_correspondence = 0;
    int plane_correspondence = 0;
    // ceres
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *q_parameterization =
        new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;

    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);

    // find correspondence for edge and planar features
    pcl::PointXYZI pointSel;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    // find correspondence for edge or corner features
    int cornerPointsSharpNum = cornerPointsSharp->points.size();
    for (int i = 0; i < cornerPointsSharpNum;
         ++i) {  // test cornerPointsSharpNum->10
      transformToStart(&(cornerPointsSharp->points[i]), &pointSel);

      // kdtree save corner or surf's less
      kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                       pointSearchSqDis);
      // find the two closest point at laserCloudCornerLast for pointSel
      int closestPointInd = -1, minPointInd2 = -1;
      if (pointSearchSqDis[0] < config_.dist_sq_threshold) {  // < 25
        closestPointInd = pointSearchInd[0];
        int closestPointScanID =
            int(laserCloudCornerLast->points[closestPointInd].intensity);

        double minPointSqDis2 = config_.dist_sq_threshold;
        // search in the direction of increasing scan line
        for (int j = closestPointInd + 1;
             j < (int)laserCloudCornerLast->points.size(); ++j) {
          // if in the same scan line, continue
          if (int(laserCloudCornerLast->points[j].intensity) <=
              closestPointScanID)
            continue;

          // if not in nearby scans, end the loop
          if (int(laserCloudCornerLast->points[j].intensity) >
              (closestPointScanID + config_.nearby_scan))
            break;

          double pointSqDis =
              (laserCloudCornerLast->points[j].x - pointSel.x) *
                  (laserCloudCornerLast->points[j].x - pointSel.x) +
              (laserCloudCornerLast->points[j].y - pointSel.y) *
                  (laserCloudCornerLast->points[j].y - pointSel.y) +
              (laserCloudCornerLast->points[j].z - pointSel.z) *
                  (laserCloudCornerLast->points[j].z - pointSel.z);

          if (pointSqDis < minPointSqDis2) {
            // find nearer point
            minPointSqDis2 = pointSqDis;
            minPointInd2 = j;
          }
        }
        // search in the direction of decreasing scan line
        for (int j = closestPointInd - 1; j >= 0; --j) {
          // if in the same scan line, continue
          if (int(laserCloudCornerLast->points[j].intensity) >=
              closestPointScanID)
            continue;

          // if not in nearby scans, end the loop
          if (int(laserCloudCornerLast->points[j].intensity) <
              (closestPointScanID - config_.nearby_scan))
            break;

          double pointSqDis =
              (laserCloudCornerLast->points[j].x - pointSel.x) *
                  (laserCloudCornerLast->points[j].x - pointSel.x) +
              (laserCloudCornerLast->points[j].y - pointSel.y) *
                  (laserCloudCornerLast->points[j].y - pointSel.y) +
              (laserCloudCornerLast->points[j].z - pointSel.z) *
                  (laserCloudCornerLast->points[j].z - pointSel.z);

          if (pointSqDis < minPointSqDis2) {
            // find nearer point
            minPointSqDis2 = pointSqDis;
            minPointInd2 = j;
          }
        }
      }

      // both closestPointInd and minPointInd2 is valid
      if (minPointInd2 >= 0) {
        Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                   cornerPointsSharp->points[i].y,
                                   cornerPointsSharp->points[i].z);
        Eigen::Vector3d last_point_a(
            laserCloudCornerLast->points[closestPointInd].x,
            laserCloudCornerLast->points[closestPointInd].y,
            laserCloudCornerLast->points[closestPointInd].z);
        Eigen::Vector3d last_point_b(
            laserCloudCornerLast->points[minPointInd2].x,
            laserCloudCornerLast->points[minPointInd2].y,
            laserCloudCornerLast->points[minPointInd2].z);

        double s;
        if (DISTORTION)
          s = (cornerPointsSharp->points[i].intensity -
               int(cornerPointsSharp->points[i].intensity)) /
              config_.scan_period;
        else
          s = 1.0;
        ceres::CostFunction *cost_function =
            LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
        corner_correspondence++;
      }
    }

    // find correspondence for planar or surf flat features
    int surfPointsFlatNum = surfPointsFlat->points.size();
    for (int i = 0; i < surfPointsFlatNum; ++i) {
      transformToStart(&(surfPointsFlat->points[i]), &pointSel);
      kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd,
                                     pointSearchSqDis);

      int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
      if (pointSearchSqDis[0] < config_.dist_sq_threshold) {
        closestPointInd = pointSearchInd[0];

        // get closest point's scan ID
        int closestPointScanID =
            int(laserCloudSurfLast->points[closestPointInd].intensity);
        double minPointSqDis2 = config_.dist_sq_threshold,
               minPointSqDis3 = config_.dist_sq_threshold;

        // search in the direction of increasing scan line
        for (int j = closestPointInd + 1;
             j < (int)laserCloudSurfLast->points.size(); ++j) {
          // if not in nearby scans, end the loop
          if (int(laserCloudSurfLast->points[j].intensity) >
              (closestPointScanID + config_.nearby_scan))
            break;

          double pointSqDis =
              (laserCloudSurfLast->points[j].x - pointSel.x) *
                  (laserCloudSurfLast->points[j].x - pointSel.x) +
              (laserCloudSurfLast->points[j].y - pointSel.y) *
                  (laserCloudSurfLast->points[j].y - pointSel.y) +
              (laserCloudSurfLast->points[j].z - pointSel.z) *
                  (laserCloudSurfLast->points[j].z - pointSel.z);

          // if in the same or lower scan line
          if (int(laserCloudSurfLast->points[j].intensity) <=
                  closestPointScanID &&
              pointSqDis < minPointSqDis2) {
            minPointSqDis2 = pointSqDis;
            minPointInd2 = j;
          }
          // if in the higher scan line
          else if (int(laserCloudSurfLast->points[j].intensity) >
                       closestPointScanID &&
                   pointSqDis < minPointSqDis3) {
            minPointSqDis3 = pointSqDis;
            minPointInd3 = j;
          }
        }

        // search in the direction of decreasing scan line
        for (int j = closestPointInd - 1; j >= 0; --j) {
          // if not in nearby scans, end the loop
          if (int(laserCloudSurfLast->points[j].intensity) <
              (closestPointScanID - config_.nearby_scan))
            break;

          double pointSqDis =
              (laserCloudSurfLast->points[j].x - pointSel.x) *
                  (laserCloudSurfLast->points[j].x - pointSel.x) +
              (laserCloudSurfLast->points[j].y - pointSel.y) *
                  (laserCloudSurfLast->points[j].y - pointSel.y) +
              (laserCloudSurfLast->points[j].z - pointSel.z) *
                  (laserCloudSurfLast->points[j].z - pointSel.z);

          // if in the same or higher scan line
          if (int(laserCloudSurfLast->points[j].intensity) >=
                  closestPointScanID &&
              pointSqDis < minPointSqDis2) {
            minPointSqDis2 = pointSqDis;
            minPointInd2 = j;
          } else if (int(laserCloudSurfLast->points[j].intensity) <
                         closestPointScanID &&
                     pointSqDis < minPointSqDis3) {
            // find nearer point
            minPointSqDis3 = pointSqDis;
            minPointInd3 = j;
          }
        }

        if (minPointInd2 >= 0 && minPointInd3 >= 0) {
          Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                     surfPointsFlat->points[i].y,
                                     surfPointsFlat->points[i].z);
          Eigen::Vector3d last_point_a(
              laserCloudSurfLast->points[closestPointInd].x,
              laserCloudSurfLast->points[closestPointInd].y,
              laserCloudSurfLast->points[closestPointInd].z);
          Eigen::Vector3d last_point_b(
              laserCloudSurfLast->points[minPointInd2].x,
              laserCloudSurfLast->points[minPointInd2].y,
              laserCloudSurfLast->points[minPointInd2].z);
          Eigen::Vector3d last_point_c(
              laserCloudSurfLast->points[minPointInd3].x,
              laserCloudSurfLast->points[minPointInd3].y,
              laserCloudSurfLast->points[minPointInd3].z);

          double s;
          if (DISTORTION)
            s = (surfPointsFlat->points[i].intensity -
                 int(surfPointsFlat->points[i].intensity)) /
                config_.scan_period;
          else
            s = 1.0;
          ceres::CostFunction *cost_function = LidarPlaneFactor::Create(
              curr_point, last_point_a, last_point_b, last_point_c, s);
          problem.AddResidualBlock(cost_function, loss_function, para_q,
                                   para_t);
          plane_correspondence++;
        }
      }
    }
    if ((corner_correspondence + plane_correspondence) < 10) {
      LOG(WARNING) << "less correspondence!!!!";
    }
    LOG(INFO) << __func__ << ":find correspondence for features elapsed:"
              << elapse_time.end(true) << " [ms]";
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 4;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << __func__ << ":solver elapsed:" << elapse_time.end() << " [ms]";
  }
  LOG(INFO) << __func__ << ":optimization  elapsed:" << opti_time.end()
            << " [ms]";
  tf_t_ = Eigen::Vector3d(para_t[0], para_t[1], para_t[2]);
  tf_q_ = Eigen::Quaterniond(para_q[3], para_q[0], para_q[1], para_q[2]);

  // odom_t_  odom_q_计算顺序不影响
  // odom_q_ = odom_q_ * tf_q_;//
  odom_t_ = odom_t_ + odom_q_ * tf_t_;
  odom_q_ = odom_q_ * tf_q_;
}
}  // namespace ceres_loam