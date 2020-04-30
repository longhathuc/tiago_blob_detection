#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <visp3/core/vpXmlParser.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/mbt/vpMbGenericTracker.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector> 

using namespace std;


class BlobDetection
{
    public:
	    BlobDetection(ros::NodeHandle nh_);
	    ~BlobDetection();
    
    protected:
        void imageCB(const sensor_msgs::ImageConstPtr& msg);
        void blobDetect(cv::Mat     image, 
                        cv::Mat     &imgMask, 
                        std::vector<cv::KeyPoint> &keypoints,           
                        vector<int> fHsvMin,
                        vector<int> fHsvMax,
                        bool        bBlur = false,
                        bool        bImshow = false);
    
        image_transport::ImageTransport _imageTransport;
        image_transport::Subscriber     image_sub;
        image_transport::Publisher      image_pub;
        image_transport::Publisher      mask_pub;
        // image_transport::Publisher      point_pub;

        //Ros topics names
        string strImage_sub_topic;
        string strImage_pub_topic;
        string strMask_pub_topic;
        string strAgimusFrame_sub_topic;

        
};

BlobDetection::BlobDetection(ros::NodeHandle nh_): _imageTransport(nh_)
{
       
    //get from params
    nh_.param<std::string>("strImage_sub_topic", strImage_sub_topic, "/rgb/image");
    nh_.param<std::string>("strImage_pub_topic", strImage_pub_topic, "/blob_detection/image_blob");
    nh_.param<std::string>("strMask_pub_topic" , strMask_pub_topic , "/blob_detection/image_mask");
    
    //Subcribe the tags topic from agimus-vision
    nh_.param<std::string>("strAgimusFrame_sub_topic" , strMask_pub_topic , "/agimus/vision/tags");

    image_sub = _imageTransport.subscribe(strImage_sub_topic, 1, &BlobDetection::imageCB, this,image_transport::TransportHints("compressed"));   
    ROS_INFO("Subcribed to the topic: %s", strImage_sub_topic.c_str());

    image_pub = _imageTransport.advertise(strImage_pub_topic, 1);
    ROS_INFO("Published to the topic: %s", strImage_pub_topic.c_str());

    mask_pub  = _imageTransport.advertise(strMask_pub_topic, 1);
    ROS_INFO("Published to the topic: %s",strMask_pub_topic.c_str());
 		
}

BlobDetection::~BlobDetection()
{
	cv::destroyAllWindows();
}

void BlobDetection::imageCB(const sensor_msgs::ImageConstPtr& msg)
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
    
    //copy value to cvMat
    cvPtr->image.copyTo(img);
    // ROS_INFO("img cols: %d", img.cols);

    vector<int> hsv_min = { 0,  0, 0};
    vector<int> hsv_max = {255, 255, 59};

    
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
    

    
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "blob_detection");
  
  ros::NodeHandle nh;
  BlobDetection bd(nh);
  ros::spin();
}