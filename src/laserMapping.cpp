// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
// #include<pcl/io/pcd_io.h>         //保存pcd文件的头文件
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

//!程序中存在雷达坐标系，地图坐标系，里程计坐标系（laseOdometry节点粗估计得到的odom坐标系），下面是坐标系转换函数。
// set initial guess,里程计位姿转化为地图位姿，作为后端初始估计
void transformAssociateToMap()
{
	//T_wmap_curr = T_wmap_wodom * T_wodom_curr
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

// 更新odom到map之间的位姿变换
void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

//雷达坐标系点转化为地图点
void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}
//地图点转化到雷达坐标系点
void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

// !回调函数中将消息都是送入各自队列
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//receive odomtry
//接收前端发送过来的里程计话题，并将位姿转换到世界坐标系下后发布
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	// 获取里程计位姿
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

    // 里程计坐标系位姿转化为地图坐标系位姿       T_w_curr = T_wmap_wodom * T_wodom_curr 
	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

    // 发布更精细的位姿
	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";        //父坐标系：/camera_init
	odomAftMapped.child_frame_id = "/aft_mapped";             //子坐标系：/aft_mapped
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

/*
下面是Scan to Map当前帧位姿精估计，即周围10帧组成的子地图submap与其他所有帧组成的全部地图进行匹配，
这种位姿估计方法联系了所有帧的信息，位姿估计更准确。
*/

//! 主处理线程
void process()
{
	while(1)
	{
		// 四个队列分别存放边线点、平面点、全部点、和里程计位姿，要确保需要的buffer里都有值

		// laserOdometry模块对本节点的执行频率进行了控制，laserOdometry模块publish的位姿是10Hz，点云的publish频率没这么高，限制是2hz
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			//todo step1.确保4个队列(角点，面点，全部点云，里程计)时间戳同步+点云转成PCL格式，odom转成eigen数据格式
			mBuf.lock();   //线程加锁，避免线程冲突
			// 以cornerLastBuf为基准，把时间戳小于其的全部pop出去，保证其他容器的最新消息与cornerLastBuf.front()最新消息时间戳同步
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();  //如果没有数据了，则线程解锁
				break;
			}

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

            //获取时间戳
			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
			// 原则上取出来的时间戳都是一样的，如果不一定说明有问题了
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			// 点云全部转成pcl的数据格式
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			// lidar odom的结果转成eigen数据格式
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();


			/*考虑到实时性，就把队列里其他的都pop出去，不然可能出现处理延时的情况
			 (考虑到实时性，Mapping线程耗时>100ms导致的队列里缓存的其他边线点都pop出去，不然可能出现处理延时的情况)
			*/
			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}

			mBuf.unlock();
            //todo step2.根据里程计提供的位姿，计算初始估计值(T_wmap_curr)，然后计算当前位姿在珊格地图中的索引，并确保其不位于地图的边界
			TicToc t_whole; //计算这个线程的全部时间
			// *根据前端结果，将里程计位姿转化为地图位姿,得到后端优化的一个初始估计值
			transformAssociateToMap();   //里程计位姿转化为地图位姿函数

			TicToc t_shift; //计算位姿转换的时间
             
			// 后端地图本质上是一个以当前点为中心的一个珊格地图，根据初始估计值计算寻找当前位姿在地图中的索引，一个格子的边长是50m
			//?当前点刚开始位于珊格的中心，而一个珊格的边上为50m，一半正好是25m，根据+25m来判断是否走出当前珊格
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;          //laserCloudCenWidth = 10                  //加这些参数的目的：使其位于总的珊格地图中心       
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;      //laserCloudCenHeight = 10
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;        //laserCloudCenDepth = 5

			//向后走时珊格中心的判断,对于负数，int去尾操作会少减1，所以要单独在减1
			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;
			
			//!当珊格地图靠近边届时，进行珊格地图的调整
			// 如果当前珊格索引小于3,就说明当前点快接近地图边界了，需要进行调整，相当于地图整体往x正方向移动
			while (centerCubeI < 3)
			{                                                                                                      
				for (int j = 0; j < laserCloudHeight; j++)                     //laserCloudHeight = 21
				{
					for (int k = 0; k < laserCloudDepth; k++)                 //laserCloudDepth =11
					{ 
						int i = laserCloudWidth - 1;                                     //laserCloudWidth = 21
						// 从x最大值开始
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =                                                    
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];    //laserCloudCornerArray[]珊格索引的指针数组，里面每个元素为珊格索引的指针（取值0～4850，元素大小总共有4581)
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =                                                                   
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];       //laserCloudSurfArray[]雷达云平面点数组
						// 整体右移
						for (; i >= 1; i--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						// 此时i = 0,也就是最右边的格子赋值给了之前最左边的格子
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						// 该点云清零，由于是指针操作，相当于最左边的格子清空了
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}
				// 索引右移一位
				centerCubeI++;
				laserCloudCenWidth++;
			}
			// 同理x如果抵达右边界，就整体左移
			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						// 整体左移
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						// 此时i = 20,也就是最左边的格子赋值给了之前最右边的格子
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						// 该点云清零，由于是指针操作，相当于最右边的格子清空了
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}
			// y和z的操作同理
			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}

			//todo step3.取出当前位姿索引，一定范围内的珊格索引(5*5*3个珊格)，构建角点和面点的局部地图
			// 以上操作相当于维护了一个局部地图，保证当前帧不在这个局部地图的边缘，这样才可以从地图中获取足够的约束
			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;
			// 从当前格子为中心，选出地图中一定范围的点云
			// 即向IJ坐标轴的正负方向各拓展2个栅格，K坐标轴的正负方向各拓展1个栅格     （5*5*3个珊格）（250m*250m*150m的一个局部地图）
			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						// 如果坐标合理
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							// 把各自的索引记录下来(把submap子地图的有效栅格索引记录下来)
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();            //pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());     
			laserCloudSurfFromMap->clear();      
			// 开始构建用来这一帧优化的小的局部地图
			//将有效栅格的点云叠加到一起组成submap子地图的特征点云，构建用来这一帧优化的局部地图
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();     //局部地图中的角点数量大小
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();            //局部地图中的面点数量大小

			// 为了减少运算量，对点云进行下采样
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);                                     //laserCloudCornerStack为当前帧降采样后的角点指针(订阅的上一帧点云角点)
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();   //(订阅上一帧点云角点)当前帧降采样后角点数量大小

			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);                                                //laserCloudSurfStack为降采样后的面点指针(订阅的上一帧点云面点)
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();            //(订阅上一帧点云面点)当前帧降采样后的面点数量大小

			printf("map prepare time %f ms\n", t_shift.toc());  //打印位姿转换的时间
			printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			// 最终的有效点云数目进行判断(最终的地图有效点云数目进行判断)
			//todo step4.特征匹配, 构建点(线)面约束，求解优化变量T_wmap_curr
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)                    //地图中的角点数量大于10，面点数量要大于50                        
			{
				TicToc t_opt;  //计算优化时间
				TicToc t_tree;  //计算构建KD-tree时间
				// 送入kdtree便于最近邻搜索
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);         //角点局部地图构建kdtree
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);                 //面点局部地图构建kdtree
				printf("build tree time %f ms \n", t_tree.toc());  //打印KD-tree构建时间

				//*建立对应关系的优化迭代次数不超2次
				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					//ceres::LossFunction *loss_function = NULL;
					// 建立ceres问题，添加参数块
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);             //损失函数
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;      //*用于控制问题的选项结构

					ceres::Problem problem(problem_options);  //实例化求解最优化问题
					problem.AddParameterBlock(parameters, 4, q_parameterization);      //添加参数块（待优化的变量是当前帧到地图坐标系，即T_wmap_curr)
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;   //计算建图数据点关联的时间
					int corner_num = 0;
					//! 1.构建角点(边线点)相关的约束
					for (int i = 0; i < laserCloudCornerStackNum; i++)             //(订阅的上一帧点云角点)当前帧降采样后角点数量大小
					{
						pointOri = laserCloudCornerStack->points[i];                         
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;

						// submap子地图中的点云都是在world坐标系下，而接收到的当前帧点云都是Lidar坐标系下，所以要把当前点根据初值投到地图坐标系下去
						//即用预测的Mapping位姿T_w_curr，将Lidar坐标系下的特征点变换到world坐标系下
						pointAssociateToMap(&pointOri, &pointSel);       //pointAssociateToMap(雷达点，地图点)  雷达坐标系点转化为局部地图点函数
						// 局部地图中寻找和该特征点最近的5个点
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);       //kdtree搜索点的平方距离默认从小往大排列
						// 判断最远的点距离不能超过1m，否则就是无效约束
						if (pointSearchSqDis[4] < 1.0)
						{ 
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							// 计算这五个点的均值
							center = center / 5.0;

							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							// 构建5个最近邻点的协方差矩阵
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();   
							}
							// 进行特征值分解
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);  // 协方差矩阵进行特征值分解(可以得到特征值和特征向量)

							// if is indeed line feature
							//* note Eigen library sort eigenvalues in increasing order
                            // PCA的原理：计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布
                           // 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向向量
							
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);          //取出特征向量矩阵第3列（最大特征值对应的特征向量)
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							// 最大特征值大于次大特征值的3倍认为是线特征
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])        //(note:求出的特征值从小到大排列)      
							{ 
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
								// 根据拟合出来的线特征方向，以平均点为中心构建两个虚拟点，代替一条直线
								point_a = 0.1 * unit_direction + point_on_line;
								point_b = -0.1 * unit_direction + point_on_line;
								// 构建约束，和lidar odom约束一致
								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;	
							}							
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					int surf_num = 0;
					//!  2.构建面点相关的约束 
					for (int i = 0; i < laserCloudSurfStackNum; i++)         //(订阅上一帧点云面点)当前帧降采样后的面点数量大小
					{
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);   //地图中寻找和该特征点最近的5个点

						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();             //*Ax+By+Cz = -1;
						// 构建平面方程Ax + By +Cz + 1 = 0
						// 通过构建一个超定方程来求解这个平面方程
						if (pointSearchSqDis[4] < 1.0)
						{
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								//printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
							}
							// find the norm of plane
							// 调用eigen接口求解该方程，解就是这个平面的法向量
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
							double negative_OA_dot_norm = 1 / norm.norm();  //法向量长度的倒数  
							// 法向量归一化
							norm.normalize();

							// Here n(pa, pb, pc) is unit norm of plane
							bool planeValid = true;
							// 根据求出来的平面方程进行校验，看看是不是符合平面约束
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								// 这里相当于求解点到平面的距离
								// 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离公式 = fabs(Ax0 + By0 + Cz0 + D) / sqrt(A^2 + B^2 + C^2)
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;	// 点如果距离平面太远，就认为这是一个拟合的不好的平面
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							// 如果平面有效就构建平面约束
							if (planeValid)
							{
								// 利用平面方程构建约束，和前端构建形式稍有不同
								ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								surf_num++;
							}
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					printf("mapping data assosiation time %f ms \n", t_data.toc());  //打印建图数据关联时间

					// 调用ceres求解
					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;         //不输出过程信息
					options.check_gradients = false;                            //将通过正常方式计算的导数和有限差分计算的导数进行比较（自己理解的)
					options.gradient_check_relative_precision = 1e-4;   //梯度检查器中检查的精度
					ceres::Solver::Summary summary;                            //优化报告存入summary
					ceres::Solve(options, &problem, &summary);
					printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				printf("mapping optimization time %f \n", t_opt.toc());   //打印建图优化的时间
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}
            // 完成特征匹配、优化后，用最后匹配计算出的优化变量	T_w_curr，更新增量T_wmap_wodom，让下一次的Mapping初值更准确              
			transformUpdate();      // 更新odom到map之间的位姿变换(T_map_odom)

			//todo step5.将优化后的当前帧角(面)点加入到珊格地图中
			TicToc t_add; //计算增加特征点的时间
			// 将优化后的当前帧角点加到局部地图中去(将优化后的当前帧边线点（角点）加到对应的边线点局部地图中去)
			for (int i = 0; i < laserCloudCornerStackNum; i++)     //遍历(订阅上一帧点云角点)当前帧降采样后的角点数量大小
			{
				// 该点根据位姿投到地图坐标系
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
				// 算出这个点所在的格子的索引
				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;
				// 同样四舍五入一下（点是负值要单独减1）
				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;
				// 如果超过边界的话就算了
				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					// 根据xyz的索引计算在一纬数组中的索引
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}
			// 面点也做同样的处理
			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			printf("add points time %f ms\n", t_add.toc());  //打印增加特征点的时间

			
			TicToc t_filter;  //计算降采样的时间
			//todo step6.把当前帧涉及到的局部地图的珊格做一个下采样（对局部地图进行降采样)
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());  //打印降采样的时间
			

			//todo step7.发布地图+当前位姿和轨迹+发布tf变换('/aft_mapped'相对于'/camera_init')
			TicToc t_pub; //计算发布地图话题数据的时间
			//publish surround map for every 5 frame
			//*每隔5帧对外发布一下局部地图
			if (frameCount % 5 == 0)   //frameCount = 0
			{
				laserCloudSurround->clear();
				// 把该当前帧相关的局部地图发布出去
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}
			//*每隔20帧发布全局地图
			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				// 21 × 21 × 11 = 4851
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			int laserCloudFullResNum = laserCloudFullRes->points.size();
			//*把当前帧点云转到地图坐标系下发布出去
			for (int i = 0; i < laserCloudFullResNum; i++)  
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}
            //！将当前帧(在地图坐标系下的)点云写入PCD文件
			// pcl::io::savePCDFileASCII ("test_pcd.pcd", *laserCloudFullRes);
			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());  // 打印发布地图话题数据的时间


			printf("whole mapping time %f ms +++++\n", t_whole.toc());  //打印整个地图的时间
			// 发布当前位姿
			nav_msgs::Odometry odomAftMapped;                                               //消息格式文档链接：http://docs.ros.org/en/kinetic/api/nav_msgs/html/msg/Odometry.html
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);
			// 发布当前轨迹
			geometry_msgs::PoseStamped laserAfterMappedPose;                    //geometry_msgs::PoseStamped消息格式链接：http://docs.ros.org/en/api/geometry_msgs/html/msg/PoseStamped.html
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;              //nav_msgs::Path消息格式链接http://docs.ros.org/en/api/geometry_msgs/html/msg/PoseStamped.html
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);
			// 发布tf
			static tf::TransformBroadcaster br;      //创建TF广播器                                            编写tf广播器的链接 http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20broadcaster%20%28C%2B%2B%29
			tf::Transform transform;       //创建tf对象
			tf::Quaternion q;
			//设置tf的平移和旋转
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			//发布tf变换
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped")); //tf::StampedTransform(参数1：tf对象， 参数2: tf对象的时间戳， 参数3：父坐标系， 参数4：子坐标系)

			frameCount++;
		}
		std::chrono::milliseconds dura(2);  //延时2ms
        std::this_thread::sleep_for(dura);
	}
}

/*
laserMapping节点订阅了来自laserOdometry的四个话题：当前帧全部点云、上一帧的边线点集合，上一帧的平面点集合，以及当前帧的位姿粗估计。
发布了四个话题：附近帧组成的点云子地图（submap），所有帧组成的点云地图，当前帧位姿精估计。
*/

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");  
	ros::NodeHandle nh;

	float lineRes = 0;    // 次极大边线点云体素滤波分辨率
	float planeRes = 0;    // 次极小平面点云体素滤波分辨率
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);     //16线,，线分辨率为0.2
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);  //16线，面分辨率为0.4
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);

	//设置体素滤波网格降采样的叶子大小
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes); // 进行体素滤波实现降采样 
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
	// !订阅了点云以及起始位姿
	// 从laserOdometry节点接收边线点
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);
    // 从laserOdometry节点接收平面点
	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);
    // 从laserOdometry节点接收到的最新帧的位姿T_w_curr
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);
    // 从laserOdometry节点接收到的当前帧原始点云（只经过一次降采样）
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

    //!发布节点话题
	// submap（子地图）所在cube（栅格）中的点云，发布周围5帧的点云（降采样以后的）
	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);
    //map地图
	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);
    // 当前帧原始点云
	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);
    //*经过Scan to Map精估计优化后的当前帧位姿   （T_map_current）
	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);
    //*将里程计坐标系位姿转化到世界坐标系位姿（地图坐标系），相当于位姿优化初值
	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);
    // 经过Scan to Map精估计优化后的当前帧平移(轨迹)
	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);
    //重置这两个数组，这两数组用于存储所有边线点栅格和平面点栅格
	for (int i = 0; i < laserCloudNum; i++)         //laserCloudNum = 4851珊格地图的大小
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());        
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	std::thread mapping_process{process}; //创建另一个线程，主要的执行程序 

	ros::spin(); //不断执行回调函数

	return 0;
}