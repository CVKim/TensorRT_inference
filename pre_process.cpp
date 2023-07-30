#include "pre_process.h"

bool PreProcess::CvNormalize(cv::Mat& image, std::string mode)
{
	if (mode == "max")
		image.convertTo(image, 1.f / 255);
	else if (mode == "")
		image.convertTo(image, 1.f);
	else {
		std::cout << "*** ERROR) There is no such nomalization mode: " << mode << std::endl;
		return -1;
	}

	return 1;
}

bool PreProcess::CvConvertChannelOrder(cv::Mat& image, std::string to_channel_order)
{
	if (to_channel_order == "rgb")
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // bgr to rgb

	return 1;
}