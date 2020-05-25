#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/camera_subscriber.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector> 
#include <string>
#include "tf/transform_listener.h"
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <tf2_ros/static_transform_broadcaster.h>
#include <urdf/model.h>
#include <opencv2/features2d.hpp>
#include <camera_info_manager/camera_info_manager.h>
#include <dynamic_reconfigure/server.h>
#include <tiago_blob_detection/tiago_blob_detection_paramsConfig.h>

using namespace std;
using namespace sensor_msgs;

class BlobDetection
{
    public:
	    BlobDetection(ros::NodeHandle nh_);
	    ~BlobDetection();
       
    protected:
        void imageCB(const sensor_msgs::ImageConstPtr& msg);
        // void imageCB(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info);
        void transformCB(const geometry_msgs::TransformStamped &tfMsg);
        void blobDetect(cv::Mat     image, 
                        cv::Mat     &imgMask, 
                        std::vector<cv::KeyPoint> &keypoints,           
                        vector<int> fHsvMin,
                        vector<int> fHsvMax,
                        bool        bBlur = false,
                        bool        bImshow = false);
        void publishHoleTF(std::vector<cv::KeyPoint> &keypoints);
        void configCallback(tiago_blob_detection::tiago_blob_detection_paramsConfig &config, uint32_t level);

        ros::NodeHandle nh_;    
        
        image_transport::ImageTransport    _imageTransport;
        // image_transport::CameraSubscriber  image_sub;
         image_transport::Subscriber  image_sub;
        image_transport::Publisher         image_pub;
        image_transport::Publisher         mask_pub;
        geometry_msgs::TransformStamped    transformStamped;
        image_geometry::PinholeCameraModel cam_model;
       
        tf::TransformListener tf_;
        ros::Subscriber transform_sub;
        string target_frame_;

        //Ros topics names
        string strImage_sub_topic;
        string strImage_pub_topic;
        string strMask_pub_topic;
        string strAgimusFrame_sub_topic;
        string strCameraInfo_sub_topic;
        string strTransform_sub_topic;
       
    
        //intrinsic matrix
        std::vector<cv::KeyPoint> keypoints;
        cv::Matx33d cameraMatrix;
        

        //Parameters for HSV mask filter
        int iHSV_min_H;
        int iHSV_min_S;
        int iHSV_min_V;
        int iHSV_max_H;
        int iHSV_max_S;
        int iHSV_max_V;

        //Parameters for blob properties
        float  fMinThreshold;
        float  fMaxThreshold;
        bool   bFilterByArea;
        float  fMinArea;
        float  fMaxArea;
        bool   bFilterByCircularity;
        float  fMinCircularity;
        float  fMaxCircularity;
        bool   bFilterByConvexity;
        float  fMinConvexity;
        float  fMaxConvexity;
        bool   bFilterByInertia;
        float  fMinInertiaRatio;
        float  fMaxInertiaRatio;

        //Parameters for camera info
        boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
        // const sensor_msgs::CameraInfo cam_info;
        string strCameraModel;
        bool   bSimulation;

        dynamic_reconfigure::Server<tiago_blob_detection::tiago_blob_detection_paramsConfig> param_server;  
        dynamic_reconfigure::Server<tiago_blob_detection::tiago_blob_detection_paramsConfig>::CallbackType call_type;
};

BlobDetection::BlobDetection(ros::NodeHandle nh_): _imageTransport(nh_), 
                                                  cinfo_(new camera_info_manager::CameraInfoManager(nh_))

{       
    //Parameters for topics 
    nh_.param<std::string>("strImage_sub_topic", strImage_sub_topic, "/rgb/image");
    nh_.param<std::string>("strImage_pub_topic", strImage_pub_topic, "/blob_detection/image_blob");
    nh_.param<std::string>("strMask_pub_topic" , strMask_pub_topic , "/blob_detection/image_mask");
    nh_.param<std::string>("strCameraInfo_sub_topic", strCameraInfo_sub_topic, "/rgb/camera_info");
    nh_.param<std::string>("strTransform_sub_topic", strTransform_sub_topic, "/agimus/vision/tags");
  

    //Parameters for HSV mask filter
    nh_.param<int>("iHSV_min_H", iHSV_min_H, 0);
    nh_.param<int>("iHSV_min_S", iHSV_min_S, 0);
    nh_.param<int>("iHSV_min_V", iHSV_min_V, 0);
    nh_.param<int>("iHSV_max_H", iHSV_max_H, 0);
    nh_.param<int>("iHSV_max_S", iHSV_max_S, 0);
    nh_.param<int>("iHSV_max_V", iHSV_max_V, 0);
    ROS_INFO("HSV Min:%d %d %d",iHSV_min_H,iHSV_min_S,iHSV_min_V);
    ROS_INFO("HSV Max:%d %d %d",iHSV_max_H,iHSV_max_S,iHSV_max_V);
    

    // Parameters for blob properties
    nh_.param<float>("fMinThreshold",        fMinThreshold,       0.0);
    nh_.param<float>("fMaxThreshold",        fMaxThreshold,       100.0);
    nh_.param<bool> ("bFilterByArea",        bFilterByArea,       true);
    nh_.param<float>("fMinArea",             fMinArea,            2.0);
    nh_.param<float>("fMaxArea",             fMaxArea,            2000.0);
    nh_.param<bool> ("bFilterByCircularity", bFilterByCircularity,true);
    nh_.param<float>("fMinCircularity",      fMinCircularity,     0.8);
    nh_.param<float>("fMaxCircularity",      fMaxCircularity,     1.0);
    nh_.param<bool> ("bFilterByConvexity",   bFilterByConvexity,  true);
    nh_.param<float>("fMinConvexity",        fMinConvexity,       0.2);
    nh_.param<float>("fMaxConvexity",        fMaxConvexity,       0.2);
    nh_.param<bool> ("bFilterByInertia",     bFilterByInertia,    true);
    nh_.param<float>("fMinInertiaRatio",    fMinInertiaRatio,   0.3);
    nh_.param<float>("fMaxInertiaRatio",    fMaxInertiaRatio,   1.0);


     //Parameters for camera info'
    nh_.param<bool> ("bSimulation",         bSimulation,   false);
    nh_.param<std::string>("strCameraModel", strCameraModel, "");
    ROS_INFO("CamInfo Link: %s",strCameraModel.c_str());

    if (cinfo_->validateURL(strCameraModel))
    {
        cinfo_->loadCameraInfo(strCameraModel);
        ROS_INFO("Got camera info & loaded!");
        cout <<  cinfo_->getCameraInfo();
    }
    else
        ROS_INFO("Recheck URL, stupid!!!!!!!");
   


    //publisher & subcriber 
    image_transport::TransportHints th("compressed");
    image_sub = _imageTransport.subscribe(strImage_sub_topic, 1, &BlobDetection::imageCB, this,image_transport::TransportHints("compressed")); 
    // image_sub = _imageTransport.subscribeCamera(strImage_sub_topic, 10, &BlobDetection::imageCB, this, th);   
    ROS_INFO("Subcribed to the topic: %s", strImage_sub_topic.c_str());

    image_pub = _imageTransport.advertise(strImage_pub_topic, 10);
    ROS_INFO("Published to the topic: %s", strImage_pub_topic.c_str());

    mask_pub  = _imageTransport.advertise(strMask_pub_topic, 10);
    ROS_INFO("Published to the topic: %s",strMask_pub_topic.c_str());

    // transform_sub = nh_.subscribe(strTransform_sub_topic, 10, &BlobDetection::transformCB, this);
    // ROS_INFO("Subcribed to the topic: %s", strTransform_sub_topic.c_str());

    // call_type = boost::bind(&configCallback, _1, _2);
    // param_server.setCallback(call_type);

    param_server.setCallback(boost::bind(&BlobDetection::configCallback, this, _1, _2));
}

BlobDetection::~BlobDetection()
{
	cv::destroyAllWindows();
}

void BlobDetection::configCallback(tiago_blob_detection::tiago_blob_detection_paramsConfig &config, uint32_t level) {
   
    iHSV_min_H = config.iHSV_min_H;
    iHSV_min_S = config.iHSV_min_S;
    iHSV_min_V = config.iHSV_min_V;
    iHSV_max_H = config.iHSV_max_H;
    iHSV_max_S = config.iHSV_max_S;
    iHSV_max_V = config.iHSV_max_V;

    fMinCircularity = config.fMinCircularity;
    fMinConvexity   = config.fMinConvexity;
    fMinThreshold   = config.fMinThreshold;
    fMinArea        = config.fMinArea;
    fMinInertiaRatio= config.fMinInertiaRatio;


    ROS_INFO("Reconfigure Request HSV: %d %d %d %d %d %d" , 
            iHSV_min_H, iHSV_min_S, iHSV_min_V,
            iHSV_max_H, iHSV_max_S, iHSV_max_V);
    ROS_INFO("Reconfigure Circularity: %6.4f, Convexity: %6.4f, Threshold: %6.4f, Area: %6.4f, Inertia: %6.4f", 
            fMinCircularity, fMinConvexity, fMinThreshold, fMinArea, fMinInertiaRatio);
    
}

void BlobDetection::transformCB(const geometry_msgs::TransformStamped &transformStamped)
{   
    ROS_DEBUG("Enter transformCB");

    ROS_DEBUG("parent_name: %s",(transformStamped.header.frame_id).c_str());
    ROS_DEBUG("translation.x:%6.4f", transformStamped.transform.translation.x);
    ROS_DEBUG("translation.y:%6.4f", transformStamped.transform.translation.y);
    ROS_DEBUG("translation.z:%6.4f", transformStamped.transform.translation.z);
    std::string urdf_file = "/home/tlha/catkin_ws/install/share/agimus_demos/urdf/aircraft_skin_with_marker.urdf";

    static tf2_ros::TransformBroadcaster tf_broadcaster;
    for (int i = 0; i < 2; i++)
    {

        geometry_msgs::TransformStamped transformStamped_hole;
        cv::Matx31d world_cord(keypoints[i].pt.x, keypoints[i].pt.y, 1);

        world_cord = cameraMatrix.inv() * world_cord;
        world_cord *= transformStamped.transform.translation.z;

        transformStamped_hole.header.stamp = ros::Time::now();
        transformStamped_hole.child_frame_id = "skin/hole_" + to_string(i) + "_link";
        transformStamped_hole.header.frame_id = "xtion_optical_frame";

        transformStamped_hole.transform.translation.x = world_cord(0, 0);
        transformStamped_hole.transform.translation.y = world_cord(1, 0);
        transformStamped_hole.transform.translation.z = world_cord(2, 0);

        transformStamped_hole.transform.rotation.x = transformStamped.transform.rotation.x;
        transformStamped_hole.transform.rotation.y = transformStamped.transform.rotation.y;
        transformStamped_hole.transform.rotation.z = transformStamped.transform.rotation.z;
        transformStamped_hole.transform.rotation.w = transformStamped.transform.rotation.w;

        tf_broadcaster.sendTransform(transformStamped_hole);
    }
    ROS_DEBUG("Exit transformCB");

}
// void BlobDetection::imageCB(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
void BlobDetection::imageCB(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat img, img_gray, imgMask, im_with_keypoints;
	cv_bridge::CvImagePtr cvPtr;
 	
    try
	{
        //convert msg to cvPtr
		cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e) 
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
        
    cam_model.fromCameraInfo(cinfo_->getCameraInfo());
    cameraMatrix = cam_model.intrinsicMatrix();
   
    //copy value to cvMat
    cvPtr->image.copyTo(img);
    // ROS_INFO("img cols: %d", img.cols);

    vector<int> hsv_min = {iHSV_min_H,  iHSV_min_S, iHSV_min_V};
    vector<int> hsv_max = {iHSV_max_H,  iHSV_max_S, iHSV_max_V};
    
	if ( img.cols > 60 && img.rows > 60)
    {
        blobDetect(img, imgMask, keypoints, hsv_min, hsv_max, false, false);
        drawKeypoints(img, keypoints, im_with_keypoints, CV_RGB(255, 0, 0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    }   

    //publish the topic to ROS
    try
    {   
        //convert cv image to ROS message
        cv_bridge::CvImage image_msg;
        image_msg.header   = msg->header; // Same timestamp and tf frame as input image
        image_msg.encoding = sensor_msgs::image_encodings::BGR8; 
        image_msg.image    = im_with_keypoints; 

        cv_bridge::CvImage mask_msg;
        mask_msg.header   = msg->header; // Same timestamp and tf frame as input image
        mask_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC1; 
        mask_msg.image    = imgMask;

        image_pub.publish(image_msg.toImageMsg());
        mask_pub.publish(mask_msg.toImageMsg());
    }
    catch  (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

   
}
    
//outout :: imgMask, keypoints
void BlobDetection::blobDetect(cv::Mat image, 
                        cv::Mat &imgMask,
                        std::vector<cv::KeyPoint> &keypoints,
                        vector<int> hsv_min,
                        vector<int> hsv_max,
                        bool  bBlur,
                        bool  bImshow)
{
    cv::SimpleBlobDetector::Params  params;

    params.minThreshold        = fMinThreshold;
    params.maxThreshold        = fMaxThreshold;
    params.filterByArea        = bFilterByArea;
    params.minArea             = fMinArea;
    params.maxArea             = fMaxArea;
    params.filterByCircularity = bFilterByCircularity;
    params.minCircularity      = fMinCircularity;
    params.maxCircularity      = fMaxCircularity;
    params.filterByConvexity   = bFilterByConvexity;
    params.minConvexity        = fMinConvexity;
    params.maxConvexity        = fMaxConvexity;
    params.filterByInertia     = bFilterByInertia;
    params.minInertiaRatio     = fMinInertiaRatio;
    params.maxInertiaRatio     = fMaxInertiaRatio;

    cv::blur(image,image,cv::Size(5,5));
    if (bImshow)
    {
        cv::imshow("Blur Image",imgMask);
        cv::waitKey(0);
    }

    //convert bgr image to hsv
    cv::Mat hsvImage;
    cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    //Hsv threshold
    cv::inRange(hsvImage, hsv_min, hsv_max, imgMask);

    //Dilate & Eroreros::init(argc, argv, "video_stream");
    cv::dilate(imgMask, imgMask, cv::Mat(), cv::Point(-1, -1), 2);
    if (bImshow)
    {
        cv::imshow("Dilate Mask",imgMask);
        cv::waitKey(0);
    }

    cv::erode(imgMask, imgMask, cv::Mat(), cv::Point(-1, -1), 2);
    if (bImshow)
    {
        cv::imshow("Erpde Mask",imgMask);
        cv::waitKey(0);
    }

    // cv::Mat im_with_keypoints;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);   

    cv::Mat imgMaskReserved;
    imgMaskReserved = 255 - imgMask;

    // Show the mask
    if (bImshow)
        cv::imshow("HSV Mask",imgMaskReserved);
    
    detector->detect(imgMaskReserved, keypoints);    
    imgMask = imgMaskReserved;

    vector<cv::KeyPoint>::const_iterator it = keypoints.begin(),
                                     end = keypoints.end();


    //Check coordinates of keypoints
    ROS_DEBUG("Keypoints:");
    int iIt = 0;                        
    for( ; it != end; ++it )
    {
        ROS_DEBUG("Keypoint %d: %4.2f,%4.2f", iIt, (*it).pt.x, (*it).pt.y);
        iIt++;
    } 
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "blob_detection");
  ros::NodeHandle nh;
  BlobDetection bd(nh);
  ros::spin();
}