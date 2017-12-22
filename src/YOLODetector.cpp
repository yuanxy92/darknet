/**
@brief YOLODetector.cpp
YOLO detector GPU cuda version
@author Shane Yuan
@date Nov 4, 2017
*/

#include "YOLODetector.hpp"
#include "network.h"
extern "C" {
	#include "detection_layer.h"
	#include "region_layer.h"
	#include "cost_layer.h"
	#include "utils.h"
	#include "parser.h"
	#include "box.h"
	#include "image.h"
	#include "demo.h"
	#include "option_list.h"
	#include "stb_image.h"
}

#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

struct YOLO_detector_ptr{
	network net;
	float **probs;
	box *boxes;
	float *avg;
	float *predictions;
	unsigned int *track_id;
};


/**********************************************************************************/
/*                                  YOLODetector class                            */
/**********************************************************************************/

YOLO_DETECTOR_API YOLODetector::YOLODetector(): nms(0.4), thresh(0.2), isInit(false) {}
YOLO_DETECTOR_API YOLODetector::~YOLODetector() {
	if (isInit) {
		std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
			(detector_ptr);
		layer l = detector->net.layers[detector->net.n - 1];
		free(detector->track_id);
		free(detector->avg);
		free(detector->predictions);
		for (int j = 0; j < l.w*l.h*l.n; ++j)
			free(detector->probs[j]);
		free(detector->boxes);
		free(detector->probs);
		cudaFree(this->input);

		int old_gpu_index;
		cudaGetDevice(&old_gpu_index);
		cudaSetDevice(detector->net.gpu_index);
		free_network(detector->net);
		cudaSetDevice(old_gpu_index);
	}
}

/**
@brief read object names from file
@param std::string object name file
@return std::vector<std::string>: return object name list
*/
YOLO_DETECTOR_API std::vector<std::string> YOLODetector::objects_names_from_file(
	std::string filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

/**
@brief draw bounding boxes for detection result
@param std::string object name file
@return std::vector<std::string>: return object name list
*/
YOLO_DETECTOR_API cv::Mat YOLODetector::draw_boxes(cv::Mat mat_img,
	std::vector<bbox_t> result_vec, std::vector<std::string> obj_names) {
	cv::Mat visualImg = mat_img.clone();
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	for (auto &i : result_vec) {
		if (i.prob < 0.25)
			break;
		int const offset = i.obj_id * 123457 % 6;
		int const color_scale = 150 + (i.obj_id * 123457) % 100;
		cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
		color *= color_scale;
		cv::rectangle(visualImg, cv::Rect(i.x, i.y, i.w, i.h), color, 5);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			cv::rectangle(visualImg, cv::Point2f(std::max<int>((int)i.x - 3, 0), 
				std::max<int>((int)i.y - 30, 0)),
				cv::Point2f(std::min<int>((int)i.x + max_width, mat_img.cols - 1),
					std::min<int>((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(visualImg, obj_name, cv::Point2f(i.x, i.y - 10),
				cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
		}
	}
	return visualImg;
}

/**
@brief get input width of network
@return int: input width of network
*/
YOLO_DETECTOR_API int YOLODetector::getNetworkWidth() {
	std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
		(detector_ptr);
	return detector->net.w;
}

/**
@brief get input height of network
@return int: input height of network
*/
YOLO_DETECTOR_API int YOLODetector::getNetworkHeight() {
	std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
		(detector_ptr);
	return detector->net.h;
}

/**
@brief get input channels of network
@return int: input channels of network
*/
YOLO_DETECTOR_API int YOLODetector::getNetworkChannels() {
	std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
		(detector_ptr);
	return detector->net.c;
}

/**
@brief init function
@param std::string cfgfile: cfg filename
@param std::string weightfile: nn weight filename
@param std::string namefile: class labelname filename
@param int deviceId: input GPU index used for detector
@return int
*/
YOLO_DETECTOR_API int YOLODetector::init(std::string cfgfile, std::string weightfile,
	int deviceId) {
	// init and get ptr
	detector_ptr = std::make_shared<YOLO_detector_ptr>();
	std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
		(detector_ptr);
	cudaSetDevice(deviceId);
	// init network
	// load network config file
	detector->net = parse_network_cfg_custom(const_cast<char*>(cfgfile.c_str()), 1);
	// load network weight file
	load_weights(&detector->net, const_cast<char*>(weightfile.c_str()));
	set_batch_network(&detector->net, 1);
	detector->net.gpu_index = deviceId;
	// init detector ptr
	layer l = detector->net.layers[detector->net.n - 1];
	detector->avg = (float *)calloc(l.outputs, sizeof(float));
	detector->predictions = (float *)calloc(l.outputs, sizeof(float));
	detector->boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	detector->probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (size_t j = 0; j < l.w*l.h*l.n; ++j) 
		detector->probs[j] = (float *)calloc(l.classes, sizeof(float));
	detector->track_id = (unsigned int *)calloc(l.classes, sizeof(unsigned int));
	for (size_t j = 0; j < l.classes; ++j) 
		detector->track_id[j] = 1;
	// malloc input gpu memory
	cudaError_t cudaStatus = cudaMalloc(&this->input, detector->net.w * detector->net.h
		* detector->net.c * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::cout << "YOLO detector init failed! Can not malloc gpu memory!" << std::endl;
		return -1;
	}
	isInit = true;
	// load obj name
	return 0;
}

/**
@brief apply detector
@param float* img_d: input image whose size equals the neural work
input tensor size (float type on GPU)
@return std::vector<bbox_t> return bounding box infos
*/
YOLO_DETECTOR_API int YOLODetector::detect(float* img_d, int w, int h,
	std::vector<bbox_t> & bbox_vec) {
	// init and get ptr
	std::shared_ptr<YOLO_detector_ptr> detector = std::static_pointer_cast<YOLO_detector_ptr>
		(detector_ptr);
	// forward neural network
#ifdef CUDA_PROFILE
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif
	float *prediction = network_predict_gpu_cuda_pointer(detector->net, img_d);
#ifdef CUDA_PROFILE
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("YOLO detection: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);
#endif
	//float *prediction = network_predict(detector->net, img_d);
	// generate final bounding box infos
	layer l = detector->net.layers[detector->net.n - 1];
	get_region_boxes(l, 1, 1, thresh, detector->probs, detector->boxes, 0, 0);
	if (nms) 
		do_nms_sort(detector->boxes, detector->probs, l.w*l.h*l.n, l.classes, nms);
	for (size_t i = 0; i < (l.w*l.h*l.n); ++i) {
		box b = detector->boxes[i];
		int const obj_id = max_index(detector->probs[i], l.classes);
		float const prob = detector->probs[i][obj_id];
		if (prob > thresh) {
			bbox_t bbox;
			bbox.x = std::max<double>((double)0, (b.x - b.w / 2.) * w);
			bbox.y = std::max<double>((double)0, (b.y - b.h / 2.) * h);
			bbox.w = b.w * w;
			bbox.h = b.h * h;
			bbox.obj_id = obj_id;
			bbox.prob = prob;
			bbox.track_id = 0;
			bbox_vec.push_back(bbox);
		}
	}
	return 0;
}