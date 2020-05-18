cmake_minimum_required(VERSION 2.8.3)
project(tiago_blob_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
  geometry_msgs
  image_geometry
  visp_bridge
  tf2_ros
  tf2_geometry_msgs
  tf
)

find_package(OpenCV COMPONENTS opencv_highgui opencv_features2d opencv_xfeatures2d opencv_calib3d)
if(NOT OPENCV_FOUND)
	find_package(OpenCV REQUIRED COMPONENTS opencv_highgui opencv_features2d opencv_calib3d)
	add_definitions(-DNO_CV_XFEATURES2D)
endif()



#list(APPEND OpenCV_LIBRARIES opencv_nonfree)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

catkin_package(
  LIBRARIES tiago_blob_detection
  CATKIN_DEPENDS roscpp image_transport cv_bridge sensor_msgs  std_msgs geometry_msgs  
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(blob_detection src/blob_detection.cpp)
target_link_libraries(blob_detection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

# install
install (TARGETS blob_detection
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS)

