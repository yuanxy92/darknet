/**
@brief main.cpp
YOLO detector GPU cuda version
@author Shane Yuan
@date Nov 4, 2017
*/

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>

#include "YOLODetector.hpp"

int main(int argc, char* argv[]) {
	std::cout << "Test YOLO detector ..." << std::endl;

	std::string cfgfile = "E:/data/YOLO/yolo-face.cfg";
	//std::string cfgfile = "E:/data/YOLO/yolo.cfg";
	std::string weightfile = "E:/data/YOLO/yolo-face.weights";
	//std::string weightfile = "E:/data/YOLO/yolo.weights";
	std::string objnamefile = "E:/data/YOLO/coco.names";
	std::shared_ptr<YOLODetector> detector = std::make_shared<YOLODetector>();

	return 0;

	//detector->init(cfgfile, weightfile, 0);

	//std::string imgname = std::string(argv[1]);
	//std::size_t found = imgname.find_last_of(".");
	//std::string suffix = imgname.substr(found + 1);

	//cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> ptr;
	//
	//cv::Mat img;
	//int frameNum = atoi(argv[7]);
	//cv::VideoCapture reader;
	//cv::VideoWriter writer, writer_back;

	//for (size_t ind = 0; ind < frameNum; ind ++) {
	//	std::cout << "Process image " << ind << "...\n";
	//	if (suffix.compare("png") == 0 || suffix.compare("jpg") == 0) {
	//		img = cv::imread(imgname);
	//	}
	//	else {
	//		if (ind == 0)
	//			reader.open(imgname);
	//		reader >> img;
	//	}
	//	//img = img(cv::Rect(1200, 1230, 1000, 750));
	//	float ratio_width;
	//	float ratio_height;
	//	float final_width;
	//	float final_height;
	//	if (argc == 3) {
	//		final_width = img.cols;
	//		final_height = img.rows;
	//	}
	//	else {
	//		final_width = atoi(argv[3]);
	//		final_height = final_width * (static_cast<float>(img.rows) / static_cast<float>(img.cols));

	//	}

	//	if (argc > 5) {
	//		float gain_b = atof(argv[5]);
	//		float gain_r = atof(argv[6]);
	//		std::vector<cv::Mat> chs(3);
	//		cv::split(img, chs);
	//		chs[0].convertTo(chs[0], -1, gain_b);
	//		chs[2].convertTo(chs[2], -1, gain_r);
	//		cv::merge(chs, img);
	//	}

	//	ratio_width = static_cast<float>(final_width) / 416.0f;
	//	ratio_height = static_cast<float>(final_height) / 416.0f;
	//	cv::Mat img_orig = img.clone();
	//	cv::resize(img, img, cv::Size(416, 416));
	//	cv::resize(img_orig, img_orig, cv::Size(final_width, final_height));

	//	cv::Mat imgf;
	//	cv::cvtColor(img, imgf, cv::COLOR_BGR2RGB);
	//	imgf.convertTo(imgf, CV_32F, 1.0 / 255.0);

	//	float *img_h = new float[imgf.rows * imgf.cols * 3];
	//	size_t count = 0;
	//	for (size_t k = 0; k < 3; ++k) {
	//		for (size_t i = 0; i < imgf.rows; ++i) {
	//			for (size_t j = 0; j < imgf.cols; ++j) {
	//				img_h[count++] = imgf.at<cv::Vec3f>(i, j).val[k];
	//			}
	//		}
	//	}

	//	float* img_d;
	//	cudaMalloc(&img_d, sizeof(float) * 3 * img.rows * img.cols);
	//	cudaMemcpy(img_d, img_h, sizeof(float) * 3 * img.rows * img.cols,
	//		cudaMemcpyHostToDevice);

	//	std::vector<std::string> objnames = YOLODetector::objects_names_from_file(objnamefile);

	//	std::vector<bbox_t> result_vec;
	//	detector->detect(img_d, img.cols, img.rows, result_vec);

	//	// remap bounding boxes
	//	for (size_t i = 0; i < result_vec.size(); i++) {
	//		result_vec[i].x *= ratio_width;
	//		result_vec[i].y *= ratio_height;
	//		result_vec[i].w *= ratio_width;
	//		result_vec[i].h *= ratio_height;
	//	}

	//	cv::Mat visualImg = YOLODetector::draw_boxes(img_orig, result_vec, objnames);

	//	if (suffix.compare("png") == 0 || suffix.compare("jpg") == 0)
	//		cv::imwrite(argv[2], visualImg);
	//	else {
	//		if (ind == 0) {
	//			ptr = cv::cuda::createBackgroundSubtractorMOG2();
	//			writer.open(std::string(argv[2]), cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 
	//				12, cv::Size(final_width, final_height), true);
	//			writer_back.open(std::string(argv[2]) + "back.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
	//				12, cv::Size(final_width, final_height), true);
	//		}
	//		cv::cuda::GpuMat imgd;
	//		cv::cuda::GpuMat maskd;
	//		cv::Mat mask;
	//		imgd.upload(img_orig);
	//		ptr->apply(imgd, maskd);
	//		maskd.download(mask);
	//		cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
	//		writer << visualImg;
	//		writer_back << mask;
	//	}

	//	cudaFree(img_d);
	//}

	//return 0;
}