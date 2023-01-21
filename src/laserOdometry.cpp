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

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0              //!! 如果需要去畸变，DISTORTION =1 


int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;  //队列先进先出
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

/*
常用的做法是补偿到起始时刻，如果有IMU，我们通过IMU得到的雷达高频位姿，可以求出每个点相对起始点的位姿，就可以补偿回去。
如果没有IMU，我们可以使用匀速模型假设，使⽤上⼀个帧间⾥程记的结果作为当前两帧之间的运动，假设当前帧也是匀速运动，可以估计出每个点相对起始时刻的位姿。
最后，当前点云中的点相对第一个点去除因运动产生的畸变，效果相当于静止扫描得到的点云。
*/
// undistort lidar point
//*将一帧中的点转换到一帧数据的起始时刻
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    // 由于kitti数据集上的lidar已经做过了运动补偿，因此这里就不做具体补偿了
    if (DISTORTION)                       //DISTORTION =0               //!! 如果需要去畸变，DISTORTION =1                           
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;     // SCAN_PERIOD=0.1                 //点的强度值 = 线号+相对起始的时间
    else
        s = 1.0;    // s = 1s说明全部补偿到点云结束的时刻
    // 所有点的操作方式都是一致的，相当于从结束时刻补偿到起始时刻
    // 这里相当于是一个匀速模型的假设

    //*利用线形插值计算变换矩阵（该点到起始点的变换矩阵)
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);  //四元数的球面线形插值（参考文章链接）：https://zhuanlan.zhihu.com/p/538653027
    Eigen::Vector3d t_point_last = s * t_last_curr;

    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;         //将该点变换到一帧数据的起始时刻

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame //! 下面是去除运动畸变的函数

//?  流程：1.首先将一帧数据中的点变换到起始时刻  2.然后将起始时刻变换到结束时刻  (为什么不直接变换到结束时刻：因为需要先线形插值获取该点到起始时刻的变换矩阵,然后才能变换到结束时刻)
//! 个人认为可以：1.先线形插值获取该点到起始时刻的变换矩阵 2. 计算当前到结束的变换矩阵T_curr_p = T_last_curr.inverse() * T_last_p'   3.将该点变换到结束时刻  （Note:只是实现方式不一样，本质是一样的，都是基于速度模型去畸变）
void TransformToEnd(PointType const *const pi, PointType *const po)          //const使用：https://www.jianshu.com/p/f5e14d751dc0
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);       //*将起始点(某点变换之后的)变换到一帧数据的结束时刻   P_end = T_last_curr.inverse() * P'

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);              //此时强度 = 线号值
}

//!回调函数
// 操作都是送去各自的队列中，加了线程锁以避免线程数据冲突
//极大边线点处理
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}
//次极大边线点处理
void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}
//极小平面点处理
void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}
//次极小平面点处理
void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

/*
     laserOdometry这个节点订阅了5个话题：有序点云、极大边线点、次极大边线点、极小平面点、次极小平面点。
    发布了4个话题：有序点云、上一帧的平面点、上一帧的边线点、当前帧位姿粗估计。主要功能是前端的激光里程计和位姿粗估计
*/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);//设定里程计的帧率，16线帧率为1
    
    //if 1, do mapping 10 Hz, if 2, do mapping 5 Hz.
    printf("Mapping %d Hz \n", 10 / skipFrameNum); 
    // 订阅提取出来的点云
    //!订阅节点
    // 从scanRegistration节点订阅的消息话题，分别为极大边线点  次极大边线点   极小平面点  次极小平面点 全部点云点(有序点云)
    //订阅话题：极大边线点集合
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    // 订阅话题：次极大边线点集合
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    // 订阅话题：极小平面点集合
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    // 订阅话题：次极小平面点集合
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    // 订阅话题：有序点云
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);
     
    //!发布节点
    // 注册发布上一帧的边线点话题
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    // 注册发布上一帧的平面点话题
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    // 注册发布全部有序点云话题，就是从scanRegistration订阅来的点云，未经其他处理
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    // 注册发布帧间的位姿变换话题
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    // 注册发布帧间的平移运动话题 
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    //循环频率0.01s
    ros::Rate rate(100);

    while (ros::ok())
    {
        // 只触发一次回调，所以每次都要调用一次；等待回调函数执行完毕，执行一次后续代码，
        ros::spinOnce();    // 触发一次回调，参考https://www.cnblogs.com/liu-fa/p/5925381.html

        // 首先确保订阅的五个消息都有，有一个队列为空都不行
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&                 //empty()：如果 queue 中没有元素的话，返回 true,有元素就是false
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {   //todo step1.取出队列中的一帧点云数据，记录时间戳+转成PCL点云格式
            // 分别求出队列第一个时间,用来分配时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
            // 因为同一帧的时间戳都是相同的，因此这里比较是否是同一帧
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK(); //用于中断程序并输出本句所在文件/行数
            }
            // 分别将五个点云消息取出来,同时转成pcl的点云格式
            
            //*队列特性：只能访问容器的第一个和最后一个元素，只能在容器的末尾添加新元素，只能从头部移除元素。
            mBuf.lock();  //数据多个线程使用，这里先进行加锁，避免线程冲突
            cornerPointsSharp->clear();       //cornerPointsSharp是pcl格式， pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();   //pop()：删除 queue 中的第一个元素（队列中存放的为点云指针)

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);       //fromROSMsg(msg, cloud)
            cornerLessSharpBuf.pop();
            
            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();  // 数据取出来后进行解锁 (std::mutex mBuf 线程互斥锁   lock()和unlock成对存在，加锁时）

            TicToc t_whole;  //计算整个激光雷达里程计的时间
            // initializing一个什么也不干的初始化，没有延迟时间，主要用来跳过第一帧数据，直接作为第二帧的上一帧
            if (!systemInited)           //bool systemInited = false,默认值为false
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else   //todo step2.特征点匹配、位姿估计(从第2帧数据开始)
            {
                // 取出比较突出的特征点数量，极大边线点和极小平面点
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;   //计算优化的时间
                // 进行两次迭代( 点到线以及点到面的非线性优化，迭代2次（选择当前优化位姿的特征点匹配，并优化位姿（4次迭代），然后重新选择特征点匹配并优化）)
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)  
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    // 定义一下ceres的核函数，使用Huber核函数来减少外点的影响，即去除outliers
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    //*由于旋转不满足一般意义的加法，因此这里使用ceres自带的local param（正常的加减法不需要localParameterization)        参考文章链接：https://zhuanlan.zhihu.com/p/488016175
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;              //*用于控制问题的选项结构

                    ceres::Problem problem(problem_options);//实例化求解最优化问题
                    // 待优化的变量是帧间位姿，平移和旋转，这里旋转使用四元数来表示
                    problem.AddParameterBlock(para_q, 4, q_parameterization);            //添加参数块(旋转)                官网参考链接：http://www.ceres-solver.org/nnls_modeling.html?highlight=ceres%20localparameterization#problem
                    problem.AddParameterBlock(para_t, 3);                                                        //添加参数块(平移)

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;           //存放搜索近邻点的索引（laserCloudCornerLast->points[j] = laserCloudCornerLast->points[pointSearchInd[i]] )
                    std::vector<float> pointSearchSqDis;   //存放近邻点与目标点之间距离的平方（默认从小到大)

                    TicToc t_data;   //计算寻找关联点的时间

                    /* 
                            基于最近邻原理建立corner特征点（边线点）之间的关联，每一个极大边线点去上一帧的次极大边线点中找匹配；采用边线点匹配方法:假如在第k+1帧中发现了边线点i，
                            通过KD-tree查询在第k帧中的最近邻点j，查询j的附近扫描线上的最近邻点l，j与l相连形成一条直线l-j，让点i与这条直线的距离最短。
                    */
                    // find correspondence for corner features
                    // 寻找角点的约束
                    //!构建一个非线性优化问题：以点i与直线lj的距离为代价函数，以位姿变换T(四元数表示旋转+位移t)为优化变量。下面是优化的程序
                    for (int i = 0; i < cornerPointsSharpNum; ++i)//先进行线点的匹配  
                    {
                        // 将该帧中点变换到一帧数据的起始时刻（即上一帧数据的结束时刻)
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        // 在上一帧所有角点（次极大边线点）构成的kdtree中寻找距离当前帧最近的一个点
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);                               

                        int closestPointInd = -1, minPointInd2 = -1;
                        // 只有小于给定门限才认为是有效约束（如果最近邻的corner特征点（次极大边线点）之间距离平方小于阈值，则最近邻点有效）
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)   //DISTANCE_SQ_THRESHOLD = 25;
                        {
                            closestPointInd = pointSearchInd[0];    // 目标点对应的最近距离点的索引取出来
                            // 找到其所在线束id，线束信息是intensity的整数部分
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);                    //!第一个点通过kdtree找，为laserCloudCornerLast->points[pointSearchInd[0]]

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;

                            //!总结：就是在第一个近邻点i,索引前后和上下个两条线束上找第二个近邻点j
                            // search in the direction of increasing scan line            //!在第一个近邻点i，索引之后并且线束之上的两条线束里找第二个近邻点j(j与poinSel之间的距离平方要小于25)
                            // 寻找角点，在刚刚角点（次极大边线点）id上下分别继续寻找最近邻点，目的是找到最近的角点，由于其按照线束进行排序，所以就是向上找
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                // 不找小于等于该线束的
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                // 要求找到的线束距离当前线束不能太远(if not in nearby scans, end the loop，即要求找到的线束距离当前线束不能太远)
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN)) //NEARBY_SCAN = 2.5;          即在当前线束之上的两条线束里找
                                    break;
                                // 计算pointSel和当前找到的角点(次极大边线点)之间的距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);              //距离=(x-m1)^2+(y-m2)^2+(z-m3)^2
                                // 寻找距离最小的角点(次极大边线点)及其索引
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    // 记录其距离平方和索引
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line    //!在第一个近邻点i，索引之前并且线束之下的两条线束里找第二个近邻点j(j与poinSel之间的距离平方要小于minPointSqDis2)（Note:如果在线束之上找到j了，此处则是在线束之下找比j更近的点,然后令其为j)
                            // 同样另一个方向寻找对应角点(次极大边线点)
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue，不找大于等于该线束的
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)    
                                    continue;

                                // if not in nearby scans, end the loop,即要求找到的线束距离当前线束不能太远
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))//NEARBY_SCAN = 2.5;  当前线束之下的两条线束里找
                                    break;
                                // 计算pointSel和当前找到的角点(次极大边线点)之间的距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point,寻找距离最小的角点（次极大边线点）及其索引， 记录其索引
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }

                        //如果当前点的两个最近邻点i和j都有效，构建非线性优化问题
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            // 取出当前点和上一帧的两个角点
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;  //去运动畸变，这里没有做，kitii数据已经做了
                            if (DISTORTION)  //#define DISTORTION 0                       //!! 如果需要进行畸变去除可以令DISTORTION = 1；
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;  //SCAN_PERIOD = 0.1       //* s是比率
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);   //LidarEdgeFactor优化边线点的模板类程序
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t); //构建优化问题并求解
                            corner_correspondence++;
                        }
                    }
                    /*
                       下面采用平面点匹配方法：假如在第k+1帧中发现了平面点i，通过KD-tree查询在第k帧（上一帧）中的最近邻点j，
                       查询j的附近扫描线上的最近邻点l和同一条扫描线的最近邻点m，这三点确定一个平面，让点i与这个平面的距离最短；
                    */
                    //!构建一个非线性优化问题：以点i与平面lmj的距离为代价函数，以位姿变换T(四元数表示旋转+t)为优化变量。
                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)                                                                                                              
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);  //*将该点变换到一帧数据的起始时刻（上一帧数据的结束时刻)
                        // 先寻找上一帧距离这个面点最近的面点
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);                         //!! 第1个点通过kdtree找

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        // 距离必须小于给定阈值
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)   //DISTANCE_SQ_THRESHOLD = 25;
                        {
                            // 取出找到的上一帧面点的索引
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            // 取出最近的面点在上一帧的第几根scan上面
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                            // 额外在寻找两个点，要求，一个点和最近点同一个scan，另一个是不同scan
                            // search in the direction of increasing scan line
                            //*按照增量方向寻找其他面点(先升序遍历搜索点寻找这些点)
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                // 不能和当前找到的上一帧面点线束距离太远
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))  //NEARBY_SCAN = 2.5 //!不能超过当前线束之上的两条线
                                    break;
                                // 计算pointSel和当前帧该点距离
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);     //距离=(x-m1)^2+(y-m2)^2+(z-m3)^2

                                // if in the same or lower scan line
                                //* 如果线束小于等于当前线束(此处即等于该线束，因为在scanRegistration.cpp中点都是按线束累加的)且距离小于minPointSqDis2          
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)             //!! 第2个点在当前线束上找距离pointSel最近的点
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                // 如果是其他线束点
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)        //*如果线束在当前线束之上的距离小于minPointSqDis3  
                                {                                                                                                                                                                                                                                       //!! 第3个点在当前线束之上的两条线束里找距离pointSel最近的点
                                    minPointSqDis3=pointSqDis;                                                                                                                         
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            //同样的方式，去按照降序方向寻找这两个点
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop,线束不能离太远
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))               //!不能超过当前线束之下的两条线束
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                //* 如果线束大于等于当前线束(此处即等于该线束，因为在scanRegistration.cpp中点都是按线束累加的)且距离小于minPointSqDis2
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)     //!! 第2个点在当前线束上找距离pointSel最近的点
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)      //!! 第3个点在当前线束之下的两条线束里找距离pointSel最近的点
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                            // 如果另外找到的两个点是有效点，就取出他们的3d坐标
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)  //去运动畸变，这里没有做，kitii数据已经做了        //!如果需要去畸变，DISTORTION = 1；
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;  //SCAN_PERIOD = 0.1
                                else
                                    s = 1.0;
                                // 构建点到面的约束，构建cere的非线性优化问题
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);  //LidarPlaneFactor优化面点的模板类程序
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t); //构建优化问题并求解
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());// 输出寻找关联点消耗的时间  
                    
                    // 如果总的线约束和面约束太少(小于10)，就打印一下
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                    // 调用ceres求解器求解，设定求解器类型，最大迭代次数，不输出过程信息，优化报告存入summary
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;  //QR分解类型
                    options.max_num_iterations = 4;        //迭代4次
                    options.minimizer_progress_to_stdout = false;  //不输出过程信息
                    ceres::Solver::Summary summary;      //优化报告存入summary
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());      //求解时间
                }
                // 经过两次LM优化消耗的时间
                printf("optimization twice time %f \n", t_opt.toc());   //输出优化两次的时间
                // 这里的w_curr 实际上是 w_last，即上一帧
                //更新帧间匹配的结果，得到lidar odom位姿
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;     //!本质：T_w_curr = T_w_last * T_last_curr 即将当前帧点云变换到世界坐标系下 （式子右边 t_w_curr和 q_w_curr本质就是 T_w_last , 此处不过是迭代计算，省略T_w_last = T_w_curr程序步骤) 
                q_w_curr = q_w_curr * q_last_curr;
            }
            //todo step3.发布雷达里程计+去畸变+发布点云
            TicToc t_pub;  //计算发布运行时间    
            // 发布lidar里程记结果
            // publish odometry
            // 创建nav_msgs::Odometry消息类型，把信息导入，并发布
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";  //父坐标系：相机坐标系
            laserOdometry.child_frame_id = "/laser_odom";  //子坐标系：odom
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);  
            // 以四元数和平移向量发出去
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);
             
            // geometry_msgs::PoseStamped消息是laserOdometry的部分内容
            geometry_msgs::PoseStamped laserPose;                     //  geometry_msgs::PoseStamped类型的官方文档链接：http://docs.ros.org/en/api/geometry_msgs/html/msg/PoseStamped.html
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;           //      nav_msgs::Path类型的官方文档链接：http://docs.ros.org/en/api/nav_msgs/html/msg/Path.html
            laserPath.poses.push_back(laserPose);               //发布路径
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
           //!去畸变，没有调用,如果调用设为1
            if (0)                   
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();  //次极大边线点
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();                 //次极小平面点
                for (int i = 0; i < surfPointsLessFlatNum; i++)                  
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();                 //有序点云
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
            
            //位姿估计完毕之后，当前次极大边线点和次极小平面点就变成了上一帧的边线点和平面点(用于构建kdtree，从而方便构建点线和点面优化问题)，把索引和数量都转移过去           
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            // kdtree设置当前帧，用来下一帧lidar odom使用
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
            // 一定降频后给后端发送（  控制后端节点的执行频率，降频后给后端发送，只有整除时才发布结果）
            if (frameCount % skipFrameNum == 0)     //frameCount=0, skipFrameNum=1
            {
                frameCount = 0;
                // 发布次极大边线点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);               //toROSMsg(cloud,msg)
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";                               //坐标系为：/camera而不是 /camera_init
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
                // 发布次极小平面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                // 原封不动的转发当前帧有序点云
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());          //输出发布时间
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());       //输出整个雷达里程计的时间
            if(t_whole.toc() > 100)                       //里程计超过100ms则有问题
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}