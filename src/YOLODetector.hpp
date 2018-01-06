/**
@brief YOLODetector.hpp
YOLO detector GPU cuda version
@author Shane Yuan
@date Nov 4, 2017
*/

#ifndef  __YOLO_DETECTOR_HPP__ 
#define  __YOLO_DETECTOR_HPP__

#ifdef YOLO_DETECTOR_EXPORTS
#if defined(_MSC_VER)
#define YOLO_DETECTOR_API __declspec(dllexport) 
#else
#define YOLO_DETECTOR_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLO_DETECTOR_API __declspec(dllimport) 
#else
#define YOLODLL_API
#endif
#endif

#include <iostream>
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>

// opencv
#include <opencv2/opencv.hpp>

// cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


struct bbox_t {
	double x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;					// confidence - probability that the object was found correctly
	unsigned int obj_id;		// class of object - from range [0, classes-1]
	unsigned int track_id;		// tracking id for video (0 - untracked, 1 - inf - tracked object)
};

class YOLODetector {
private:
	std::shared_ptr<void> detector_ptr;
	bool isInit;
public:
	float *input;
	float nms;
	float thresh;
private:

public:
	YOLO_DETECTOR_API YOLODetector();
	YOLO_DETECTOR_API ~YOLODetector();

	/**
	@brief init function
	@param std::string cfgfile: cfg filename 
	@param std::string weightfile: nn weight filename
	@param std::string namefile: class labelname filename
	@param int deviceId: input GPU index used for detector
	@return int
	*/
	YOLO_DETECTOR_API int init(std::string cfgfile, std::string weightfile,
		int deviceId = 0);

	/**
	@brief release function
	@return int
	*/
	YOLO_DETECTOR_API int release();

	/**
	@brief apply detector
	@param float* img_d: input image whose size equals the neural work
	input tensor size (float type on GPU)
	@param int w: input image width
	@param int h: input image height
	@param std::vector<bbox_t> & bbox_vec: output bounding box infos
	@return int
	*/
	YOLO_DETECTOR_API int detect(float* img_d, int w, int h, std::vector<bbox_t> & bbox_vec);

	/**
	@brief get input width of network
	@return int: input width of network
	*/
	YOLO_DETECTOR_API int getNetworkWidth();

	/**
	@brief get input height of network
	@return int: input height of network
	*/
	YOLO_DETECTOR_API int getNetworkHeight();

	/**
	@brief get input channels of network
	@return int: input channels of network
	*/
	YOLO_DETECTOR_API int getNetworkChannels();


	/**********************************************************************************/
	/*                                static functions                                */
	/**********************************************************************************/
	/**
	@brief read object names from file
	@param std::string object name file
	@return std::vector<std::string>: return object name list
	*/
	static YOLO_DETECTOR_API std::vector<std::string> objects_names_from_file(
		std::string filename);

	/**
	@brief draw bounding boxes for detection result
	@param std::string object name file
	@return std::vector<std::string>: return object name list
	*/
	static YOLO_DETECTOR_API cv::Mat draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,
		std::vector<std::string> obj_names);
};


#endif