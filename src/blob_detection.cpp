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
#include <visp3/core/vpXmlParser.h>
#include <visp3/core/vpTracker.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/mbt/vpMbGenericTracker.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <visp3/core/vpTranslationVector.h>
#include <visp3/core/vpQuaternionVector.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector> 
#include <string>



using namespace std;
using namespace sensor_msgs;

class BlobDetection
{
    public:
	    BlobDetection(ros::NodeHandle nh_);
	    ~BlobDetection();
    
    protected:
        
        void imageCB(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info);
        void blobDetect(cv::Mat     image, 
                        cv::Mat     &imgMask, 
                        std::vector<cv::KeyPoint> &keypoints,           
                        vector<int> fHsvMin,
                        vector<int> fHsvMax,
                        bool        bBlur = false,
                        bool        bImshow = false);

        ros::NodeHandle nh_;    
        image_transport::ImageTransport    _imageTransport;
        image_transport::CameraSubscriber  image_sub;
        image_transport::Publisher         image_pub;
        image_transport::Publisher         mask_pub;
        geometry_msgs::TransformStamped    transformStamped;
        image_geometry::PinholeCameraModel cam_model;

        //Ros topics names
        string strImage_sub_topic;
        string strImage_pub_topic;
        string strMask_pub_topic;
        string strAgimusFrame_sub_topic;
        string strCameraInfo_sub_topic;


        //intrinsic matrix
        cv::Matx33d cameraMatrix;

        int iHSV_min_H;
        int iHSV_min_S;
        int iHSV_min_V;
        int iHSV_max_H;
        int iHSV_max_S;
        int iHSV_max_V;
        
};

BlobDetection::BlobDetection(ros::NodeHandle nh_): _imageTransport(nh_)
{
       
    //get from params
    nh_.param<std::string>("strImage_sub_topic", strImage_sub_topic, "/rgb/image");
    nh_.param<std::string>("strImage_pub_topic", strImage_pub_topic, "/blob_detection/image_blob");
    nh_.param<std::string>("strMask_pub_topic" , strMask_pub_topic , "/blob_detection/image_mask");
    nh_.param<std::string>("strCameraInfo_sub_topic", strCameraInfo_sub_topic, "/rgb/camera_info");
    
    //Subcribe the tags topic from agimus-vision
    nh_.param<std::string>("strAgimusFrame_sub_topic" , strAgimusFrame_sub_topic , "/agimus/vision/tags");

    //get HSV RGB params
    // nh_.getParam("iHSV_min_H", iHSV_min_H, 0);

    nh_.param<int>("iHSV_min_H", iHSV_min_H, 0);
    nh_.param<int>("iHSV_min_S", iHSV_min_S, 0);
    nh_.param<int>("iHSV_min_V", iHSV_min_V, 0);
    nh_.param<int>("iHSV_max_H", iHSV_max_H, 0);
    nh_.param<int>("iHSV_max_S", iHSV_max_S, 0);
    nh_.param<int>("iHSV_max_V", iHSV_max_V, 0);
    

    image_sub = _imageTransport.subscribeCamera(strImage_sub_topic, 1, &BlobDetection::imageCB, this);   
    // image_sub.subscribe(nh_, strImage_sub_topic, 1);
    ROS_INFO("Subcribed to the topic: %s", strImage_sub_topic.c_str());

    image_pub = _imageTransport.advertise(strImage_pub_topic, 1);
    ROS_INFO("Published to the topic: %s", strImage_pub_topic.c_str());

    mask_pub  = _imageTransport.advertise(strMask_pub_topic, 1);
    ROS_INFO("Published to the topic: %s",strMask_pub_topic.c_str());

    
    ROS_INFO("HSV Min:%d %d %d",iHSV_min_H,iHSV_min_S,iHSV_min_V);
    ROS_INFO("HSV Max:%d %d %d",iHSV_max_H,iHSV_max_S,iHSV_max_V);

 		
}

BlobDetection::~BlobDetection()
{
	cv::destroyAllWindows();
}

void BlobDetection::imageCB(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
{
    
    cv::Mat img, img_gray, imgMask, im_with_keypoints;
	cv_bridge::CvImagePtr cvPtr;
    std::vector<cv::KeyPoint> keypoints;
 	
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
    
    cam_model.fromCameraInfo(cam_info);
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
    params.minThreshold        =    0;
    params.maxThreshold        =  100;
    params.filterByArea        = true;
    params.minArea             =    2;
    params.maxArea             =20000;
    params.filterByCircularity = true;
    params.minCircularity      = 0.1;
    params.filterByConvexity   = true;
    params.minConvexity        = 0.2;
    params.filterByInertia     = true;
    params.minInertiaRatio     = 0.3;

    // cout << "intrinsic matrix:" << endl;
    // cout << cameraMatrix << endl;

     
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

    cv::Mat im_with_keypoints;
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

    // //look up the transform
    ros::Rate rate(10.0);
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

        try
        {
            transformStamped = tfBuffer.lookupTransform("xtion_optical_frame","skin/tag36_11_00020_tf",ros::Time(0));
            ROS_DEBUG("parent_name: %s",(transformStamped.header.frame_id).c_str());
            ROS_DEBUG("transformStamped.transform.translation.x:%6.4f", transformStamped.transform.translation.x);
            ROS_DEBUG("transformStamped.transform.translation.x:%6.4f", transformStamped.transform.translation.y);
            ROS_DEBUG("transformStamped.transform.translation.x:%6.4f", transformStamped.transform.translation.z);
           
            static tf2_ros::TransformBroadcaster tf_broadcaster;
            
            for (int i = 0; i < keypoints.size(); i++)
            {
 
                geometry_msgs::TransformStamped transformStamped_hole;
                cv::Matx31d world_cord(keypoints[i].pt.x,keypoints[i].pt.y,1); 
                
                world_cord =  cameraMatrix.inv() * world_cord;
                world_cord *= transformStamped.transform.translation.z;

                transformStamped_hole.header.stamp = ros::Time::now();
                transformStamped_hole.child_frame_id = "hole_" + to_string(i) + "_tf";
                // transformStamped_hole.child_frame_id = "skin/hole_link_measured";
                transformStamped_hole.header.frame_id = "xtion_optical_frame";

                transformStamped_hole.transform.translation.x = world_cord(0,0);
                transformStamped_hole.transform.translation.y = world_cord(1,0);
                transformStamped_hole.transform.translation.z = world_cord(2,0);

                transformStamped_hole.transform.rotation.x = transformStamped.transform.rotation.x;
                transformStamped_hole.transform.rotation.y = transformStamped.transform.rotation.y;
                transformStamped_hole.transform.rotation.z = transformStamped.transform.rotation.z;
                transformStamped_hole.transform.rotation.w = transformStamped.transform.rotation.w;

                tf_broadcaster.sendTransform(transformStamped_hole);
            }
        }
        catch (tf2::TransformException &ex) {
            ROS_WARN("%s",ex.what());
            ros::Duration(1.0).sleep();
            //  continue;
        }

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "blob_detection");
  
  ros::NodeHandle nh;
  BlobDetection bd(nh);
  ros::spin();
}