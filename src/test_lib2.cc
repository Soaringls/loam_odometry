#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include "lib/cloud_segmentation.h"
#include "lib/lidar_odometry.h"
#include "lib/lidar_refination.h"
// #include "lib/loam_odometry.h"
using namespace ceres_loam;

void publishCloud(ros::Publisher* thisPub,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr thisCloud,
                  ros::Time thisStamp, std::string thisFrame = "/camera_init") {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
}

void publishPath(ros::Publisher pub, nav_msgs::Path& path,
                 Eigen::Vector3d t_w_curr, Eigen::Quaterniond q_w_curr) {
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";
  laserOdometry.header.stamp = ros::Time().now();
  laserOdometry.pose.pose.orientation.x = q_w_curr.x();
  laserOdometry.pose.pose.orientation.y = q_w_curr.y();
  laserOdometry.pose.pose.orientation.z = q_w_curr.z();
  laserOdometry.pose.pose.orientation.w = q_w_curr.w();
  laserOdometry.pose.pose.position.x = t_w_curr.x();
  laserOdometry.pose.pose.position.y = t_w_curr.y();
  laserOdometry.pose.pose.position.z = t_w_curr.z();

  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  path.header.stamp = laserOdometry.header.stamp;
  path.poses.push_back(laserPose);
  path.header.frame_id = "/camera_init";
  pub.publish(path);
}

void publishPath(ros::Publisher pub, nav_msgs::Path& path,
                 Eigen::Affine3d pose) {
  Eigen::Translation3d t(pose.translation());
  Eigen::Quaterniond q(pose.linear());
  nav_msgs::Odometry laserOdometry;
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";
  laserOdometry.header.stamp = ros::Time().now();
  laserOdometry.pose.pose.orientation.x = q.x();
  laserOdometry.pose.pose.orientation.y = q.y();
  laserOdometry.pose.pose.orientation.z = q.z();
  laserOdometry.pose.pose.orientation.w = q.w();
  laserOdometry.pose.pose.position.x = t.x();
  laserOdometry.pose.pose.position.y = t.y();
  laserOdometry.pose.pose.position.z = t.z();

  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  path.header.stamp = laserOdometry.header.stamp;
  path.poses.push_back(laserPose);
  path.header.frame_id = "/camera_init";
  pub.publish(path);
}

class CloudHandle {
 public:
  CloudHandle() {
    sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
        "/sensor/velodyne16/back/PointCloud2", 10000, &CloudHandle::Callback,
        this);
    // pub path
    pub_odom_ = nh_.advertise<nav_msgs::Path>("/laser_odometry_path", 100);
    pub_odom_coarse_ =
        nh_.advertise<nav_msgs::Path>("/laser_odometry_coarsepath", 100);
    pub_mapping_ = nh_.advertise<nav_msgs::Path>("/laser_mapping_path", 100);
    // pub cloud
    pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/full_cloud", 1);
    pub_edge_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_seg/edge", 1);
    pub_edge_lesssharp_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/cloud_seg/edge_lesssharp", 1);
    pub_planar_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/cloud_seg/planar", 1);
    pub_planar_lessflat_ = nh_.advertise<sensor_msgs::PointCloud2>(
        "/cloud_seg/planar_lessflat", 1);
    // pub map
    pub_map_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/cloud_seg/globalMap", 1);
    pub_submap_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/cloud_seg/SubMap", 1);
    std::cout << "CloudHandle construct over!\n";
  }

  void Callback(const sensor_msgs::PointCloud2ConstPtr& input) {
    static int cnt(0);
    Timer t_frame, t_sub;
    cnt++;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromROSMsg(*input, *cloud);
    double time = input->header.stamp.toSec();

    cloud_segmentation_.CloudHandler(cloud, time);
    auto dataset = DataCenter::Instance().GetScanSegMsg();

    publishCloud(&pub_cloud_, dataset->laserCloud, ros::Time(time));
    publishCloud(&pub_edge_, dataset->cornerPointsSharp, ros::Time(time));
    publishCloud(&pub_edge_lesssharp_, dataset->cornerPointsLessSharp,
                 ros::Time(time));
    publishCloud(&pub_planar_, dataset->surfPointsFlat, ros::Time(time));
    publishCloud(&pub_planar_lessflat_, dataset->surfPointsLessFlat,
                 ros::Time(time));
    lidar_odom_.Process();
    lidar_mapping_.Process();

    LOG(INFO) << "final processTotal:" << t_frame.end() << " [ms]";
    // publish odometry
    auto mapping_pose = lidar_mapping_.GetPose();
    static nav_msgs::Path mappingPath;
    publishPath(pub_mapping_, mappingPath, mapping_pose);

    publishCloud(&pub_map_, lidar_mapping_.GenerateWholeMap(), ros::Time(time));
    publishCloud(&pub_submap_, lidar_mapping_.GenerateSurroundMap(),
                 ros::Time(time));
  }

 private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_odom_coarse_, pub_odom_, pub_mapping_;
  ros::Publisher pub_cloud_;
  ros::Publisher pub_edge_;
  ros::Publisher pub_edge_lesssharp_;
  ros::Publisher pub_planar_;
  ros::Publisher pub_planar_lessflat_;
  ros::Publisher pub_map_, pub_submap_;
  CloudSegmentation cloud_segmentation_;
  LidarOdometry lidar_odom_;
  LidarRefination lidar_mapping_;
  //   LoamOdometry loam_odometry_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "testlib2");
  CloudHandle test;
  ros::spin();
  return 0;
}
