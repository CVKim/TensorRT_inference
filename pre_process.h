#pragma once
#include <opencv2/opencv.hpp>


class PreProcess 
{
public:
	bool CvNormalize(cv::Mat& image, std::string mode);
	bool CvConvertChannelOrder(cv::Mat& image, std::string to_channel_order);

};