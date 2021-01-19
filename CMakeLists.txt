cmake_minimum_required(VERSION 2.8.3)
project(loam_odometry)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)

link_directories(${Boost_LIBRARY_DIRS})

include_directories(
 # include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${YAMLCPP_INCLUDE_DIR}
)

set(THIRD_LIBS 
    yaml-cpp glog
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Boost_LIBRARIES}
)
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  #INCLUDE_DIRS include
) 

add_library(ceres_loam 
            src/lib/cloud_segmentation.cc
            src/lib/config.cc
            src/lib/lidar_odometry.cc
            src/lib/lidar_refination.cc
)
target_link_libraries(ceres_loam  ${THIRD_LIBS})
add_executable(testlib2 src/test_lib2.cc)
target_link_libraries(testlib2
       ${catkin_LIBRARIES} ceres_loam
)
