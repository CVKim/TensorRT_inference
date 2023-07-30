#pragma once
#include "agent.h"


Agent* m_agent;

std::wstring m_trt_file_path;
std::string m_input_image_path;
std::string m_dl_framework;
std::string m_channel_order;
bool m_flag_roi;
string m_norm;
int m_roi_x;
int m_roi_y;
int m_roi_width;
int m_roi_height;
int m_input_width;
int m_input_height;
int m_input_channel;
std::string m_output_path;

cv::Mat m_input_image;
std::vector<cv::Mat> m_input_images;
