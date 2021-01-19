#include "config.h"
namespace ceres_loam {

// Config LoamConfig::config_ =
bool LoadConfig(const std::string& config_file) {
  //   if (!boost::filesystem::exists(config_file)) {
  //     LOG(ERROR) << "The config fiel:" << config_file << " not exist!!";
  //     return false;
  //   }
  //   YAML::Node yaml_config = YAML::LoadFile(config_file);
  Config config;  // LoamConfig::GetConfig();
                  //   config.use_cloud_ring =
  //   yaml_config["lidar"]["use_cloud_ring"].as<bool>(); config.num_scans =
  //   yaml_config["lidar"]["num_scans"].as<int>(); config.num_horizontal_scans
  //   =
  //       yaml_config["lidar"]["num_horizontal_scans"].as<int>();
  //   config.ground_scan_index =
  //       yaml_config["lidar"]["ground_scan_index"].as<int>();
  //   config.vertical_angle_bottom =
  //       yaml_config["lidar"]["vertical_angle_bottom"].as<float>();
  //   config.vertical_angle_top =
  //       yaml_config["lidar"]["vertical_angle_top"].as<float>();
  //   config.sensor_mount_angle =
  //       yaml_config["lidar"]["sensor_mount_angle"].as<float>();
  //   config.scan_period = yaml_config["lidar"]["yaml_config"].as<float>();
  // todo
  LoamConfig::GetInstance(config);
  return true;
}
}  // namespace ceres_loam