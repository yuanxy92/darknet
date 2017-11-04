/**
@brief main.cpp
YOLO detector GPU cuda version
@author Shane Yuan
@date Nov 4, 2017
*/

#include <iostream>
#include <cstdlib>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "YOLODetector.hpp"

int main(int argc, char* argv[]) {
	std::cout << "Test YOLO detector ..." << std::endl;

	std::string cfgfile = "E:/data/YOLO/tiny-yolo.cfg";
	std::string weightfile = "E:/data/YOLO/tiny-yolo.weights";
	std::string objnamefile = "E:/data/YOLO/coco.names";
	YOLODetector detector;
	detector.init(cfgfile, weightfile, 0);
	
	cv::Mat img = cv::imread("E:/Project/darknet/src/build/Release/local_04.jpg");
	cv::Mat imgf;
	cv::cvtColor(img, imgf, cv::COLOR_BGR2RGB);
	imgf.convertTo(imgf, CV_32F, 1.0 / 255.0);

	float *img_h = new float[imgf.rows * imgf.cols * 3];
	size_t count = 0;
	for (size_t k = 0; k < 3; ++k) {
		for (size_t i = 0; i < imgf.rows; ++i) {
			for (size_t j = 0; j < imgf.cols; ++j) {
				img_h[count++] = imgf.at<cv::Vec3f>(i, j).val[k];
			}
		}
	}

	float* img_d;
	cudaMalloc(&img_d, sizeof(float) * 3 * img.rows * img.cols);
	cudaMemcpy(img_d, img_h, sizeof(float) * 3 * img.rows * img.cols,
		cudaMemcpyHostToDevice);

	std::vector<std::string> objnames = YOLODetector::objects_names_from_file(objnamefile);

	std::vector<bbox_t> result_vec;
	detector.detect(img_d, img.cols, img.rows, result_vec);

	cv::Mat visualImg = YOLODetector::draw_boxes(img, result_vec, objnames);

	cudaFree(img_d);

	return 0;
}