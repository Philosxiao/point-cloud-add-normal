#include <iostream>
#include <ctime>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <mutex>

#include <pcl-1.9/pcl/point_types.h>
#include <pcl-1.9/pcl/filters/passthrough.h>
#include <pcl-1.9/pcl/io/pcd_io.h>
#include <pcl-1.9/pcl/io/ply_io.h>
#include <pcl-1.9/pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

#include <boost/filesystem.hpp>
#include <fstream>

// OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

typedef pcl::PointXYZRGB RGB_Cloud;
typedef pcl::PointCloud<RGB_Cloud> point_cloud;

IplImage *loadDepth(std::string a_name)
{
    std::ifstream l_file(a_name.c_str(), std::ofstream::in | std::ofstream::binary);

    if (l_file.fail() == true)
    {
        printf("cv_load_depth: could not open file for writing!\n");
        return NULL;
    }
    int l_row;
    int l_col;

    l_file.read((char *)&l_row, sizeof(l_row));
    l_file.read((char *)&l_col, sizeof(l_col));

    IplImage *lp_image = cvCreateImage(cvSize(l_col, l_row), IPL_DEPTH_16U, 1);

    for (int l_r = 0; l_r < l_row; ++l_r)
    {
        for (int l_c = 0; l_c < l_col; ++l_c)
        {
            l_file.read((char *)&CV_IMAGE_ELEM(lp_image, unsigned short, l_r, l_c), sizeof(unsigned short));
        }
    }
    l_file.close();

    return lp_image;
}

point_cloud::Ptr tran2pointscloud(cv::Mat rgb, cv::Mat depth)
{

    // 相机内参
    double camera_factor = 1;
    double camera_cx;
    double camera_cy;
    double camera_fx;
    double camera_fy;

    camera_fx = 572.41140;
    camera_fy = 573.57043;

    camera_cx = 325.26110;
    camera_cy = 242.04899;

    // 点云变量
    // 使用智能指针，创建一个空点云。这种指针用完会自动释放。
    point_cloud::Ptr cloud(new point_cloud);
    // 遍历深度图
    for (int m = 0; m < depth.rows; m++)     //m是y
        for (int n = 0; n < depth.cols; n++) //n是x
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
            {
                //<<"no _data"<<endl;
                continue;
            }
            // d 存在值，则向点云增加一个点
            RGB_Cloud p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = (int)rgb.ptr<uchar>(m)[n * 3];
            p.g = (int)rgb.ptr<uchar>(m)[n * 3 + 1];
            p.r = (int)rgb.ptr<uchar>(m)[n * 3 + 2];
            // 把p加入到点云中
            cloud->points.push_back(p);
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout << "point cloud size = " << cloud->points.size() << endl;
    cloud->is_dense = false;
    return cloud;
}

std::vector<std::string> readPicsList(std::string path, std::string ext)
{
    std::vector<std::string> proj_map_list;
    boost::filesystem::path directory(path);
    boost::filesystem::directory_iterator itr(directory), end_itr;
    for (; itr != end_itr; ++itr)
    {
        if (boost::filesystem::is_regular_file(itr->path()))
        {
            std::string current_file = itr->path().string();
            if (current_file.substr(current_file.find_last_of(".") + 1) == ext)
            {
                proj_map_list.push_back(current_file);
            }
        }
    }
    return proj_map_list;
}

vector<vector<float>> matrix_multiply(vector<vector<float>> arrA, vector<vector<float>> arrB)
{
    //矩阵arrA的行数
    float rowA = arrA.size();
    //矩阵arrA的列数
    float colA = arrA[0].size();
    //矩阵arrB的行数
    float rowB = arrB.size();
    //矩阵arrB的列数
    float colB = arrB[0].size();
    //相乘后的结果矩阵
    vector<vector<float>> res;
    if (colA != rowB) //如果矩阵arrA的列数不等于矩阵arrB的行数。则返回空
    {
        return res;
    }
    else
    {
        //设置结果矩阵的大小，初始化为为0
        res.resize(rowA);
        for (int i = 0; i < rowA; ++i)
        {
            res[i].resize(colB);
        }

        //矩阵相乘
        for (int i = 0; i < rowA; ++i)
        {
            for (int j = 0; j < colB; ++j)
            {
                for (int k = 0; k < colA; ++k)
                {
                    res[i][j] += arrA[i][k] * arrB[k][j];
                }
            }
        }
    }
    return res;
}

int main(int argc, char *argv[])
{
    string num("0");
    int a=0;
    Mat color_image = imread("/home/philos/Desktop/dataset/duck/data/color" + num + ".jpg", -1);
    IplImage *iplimg = loadDepth("/home/philos/Desktop/dataset/duck/data/depth" + num + ".dpt");
    Mat depth_image = cv::cvarrToMat(iplimg);
    point_cloud::Ptr source_point_cloud = tran2pointscloud(color_image, depth_image);

    //计算法线
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    //建立kdtree来进行近邻点集搜索
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    //为kdtree添加点运数据
    tree->setInputCloud(source_point_cloud);
    n.setInputCloud(source_point_cloud);
    n.setSearchMethod(tree);
    //点云法向计算时，需要所搜的近邻点大小
    n.setKSearch(20);
    //开始进行法向计算
    n.compute(*normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    pcl::concatenateFields(*source_point_cloud, *normals, *cloud_with_normals);

    //法向量纠正
    for (int s = 0; s < cloud_with_normals->points.size(); s++)
    {
        if (cloud_with_normals->points[s].normal_z > 0)
        {
            //cout<<"纠正法向量"<<endl;
            cloud_with_normals->points[s].normal_x = -cloud_with_normals->points[s].normal_x;
            cloud_with_normals->points[s].normal_y = -cloud_with_normals->points[s].normal_y;
            cloud_with_normals->points[s].normal_z = -cloud_with_normals->points[s].normal_z;
        }
    }
    pcl::io::savePLYFile("duck(" + num + ").ply", *cloud_with_normals); //将点云保存到Ply文件
}