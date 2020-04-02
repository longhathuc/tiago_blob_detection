#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

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
              
        
};

BlobDetection::BlobDetection(ros::NodeHandle nh_): _imageTransport(nh_)
{
	    
    image_sub = _imageTransport.subscribe("/videofile/image_raw", 1, &BlobDetection::imageCB, this,image_transport::TransportHints("compressed"));   
    ROS_INFO("Subcribed to the topic: /videofile/image_raw \n");

    image_pub = _imageTransport.advertise("/blob_detection/image_blob", 1);
    ROS_INFO("Published to the topic: /blob_detection/image_blob \n");

    mask_pub  = _imageTransport.advertise("/blob_detection/image_mask", 1);
    ROS_INFO("Published to the topic: /blob_detection/image_mask \n");
 	// cv::namedWindow("Blob Tracking Window", CV_WINDOW_FREERATIO);	

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

    vector<int> hsv_min = {0,  0, 166};
    vector<int> hsv_max = {26, 66, 255};

    
	if ( img.cols > 60 && cvPtr->image.rows > 60)
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
    params.maxArea             =  200000;
    params.filterByCircularity = true;
    params.minCircularity      = 0.1;
    params.filterByConvexity   = true;
    params.minConvexity        = 0.2;
    params.filterByInertia     = true;
    params.minInertiaRatio     = 0.3;

  
    
    //convert bgr image to hsv
    cv::Mat hsvImage;
    cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    //Hsv threshold
    cv::inRange(hsvImage, hsv_min, hsv_max, imgMask);

    //todo: erode & dilate

    
    
    cv::Mat im_with_keypoints;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);   

    cv::Mat imgMaskReserved;
    imgMaskReserved = 255 - imgMask;

    // Show the mask
    if (bImshow)
        cv::imshow("HSV Mask",imgMaskReserved);
    
    detector->detect(imgMaskReserved, keypoints);    
    imgMask = imgMaskReserved;

    
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "blob_detection", ros::init_options::AnonymousName);

  ros::NodeHandle nh;

  BlobDetection bd(nh);

  ros::spin();
}