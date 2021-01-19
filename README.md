# loam_odometry

loam_odometry is an advanced implementation of **LOAM** and **ALOAM**, it also mainly refer the preprocessor of **Lego-Loam**,such as the segmentation and outlier removement of the each scan.

the original author(zhangji、qintong and Tixiaoshan etc) deserves all credits and respects, I just improve the software engineering practices to make the code more readable and efficient. 
the purpose of loam_odometry is as below:
1. to improve the quality of the code.
2. to convert a multi-process application into a single-process / multi-threading one; this makes the algorithm more deterministic and slightly faster.
3. to remove hard-coded values and use `yaml` configuration files to set the parameters.
4. to improve performance, in terms of amount of CPU used to implement the same result.
## dependency
1. pcl
2. ceres
3. eigen

## usage
```sh
roslaunch loam_odometry test_loamlib2.launch
```
## example
the result of dataset in Beijing
   ![asdf](./picture/my_dataset.gif)