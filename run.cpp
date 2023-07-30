#include "agent.h"
#include "run.h"
#include "pre_process.h"

#include <stdio.h>
#include <cstdlib>
#include <thread>
#include <string>
#include <Windows.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include <chrono>
#include <mutex>
#include <cuda_runtime_api.h>

using std::filesystem::recursive_directory_iterator;
using Arguments = std::unordered_multimap<std::string, std::string>;

void convertONNXtoTRT(const std::string& onnxFile, const std::string& trtFile, const std::string& workspace, int gpuIndex, bool fp16Mode) {

	std::mutex mtx;
	mtx.lock();
	//std::string cmd = "trtexec --onnx=" + onnxFile + " --saveEngine=" + trtFile + " --device=" + std::to_string(gpuIndex) + " --workspace=8192";
	std::string cmd = "trtexec --onnx=" + onnxFile + " --saveEngine=" + trtFile + " --device=" + std::to_string(gpuIndex) + " --workspace=" + workspace;
	if (fp16Mode)
		cmd += " --fp16 --refit";

	int ret = system(cmd.c_str());

	if (ret != 0) {
		std::cout << "Error executing trtexec for file " << onnxFile << std::endl;
	}
	mtx.unlock();
}

template <typename T>
T stringToValue(const std::string& option)
{
	return T{ option };
}

template <>
int32_t stringToValue<int32_t>(const std::string& option)
{
	return std::stoi(option);
}

template <>
float stringToValue<float>(const std::string& option)
{
	return std::stof(option);
}

template <>
bool stringToValue<bool>(const std::string& option)
{
	return true;
}

template <typename T>
bool getAndDelOption(Arguments& arguments, const std::string& option, T& value)
{
	const auto match = arguments.find(option);
	if (match != arguments.end())
	{
		value = stringToValue<T>(match->second);
		arguments.erase(match);
		return true;
	}

	return false;
}

Arguments argsToArgumentsMap(int32_t argc, char* argv[])
{
	Arguments arguments;
	for (int32_t i = 1; i < argc; ++i)
	{
		auto valuePtr = strchr(argv[i], '=');
		if (valuePtr)
		{
			std::string value{ valuePtr + 1 };
			arguments.emplace(std::string(argv[i], valuePtr - argv[i]), value);
		}
		else
		{
			arguments.emplace(argv[i], "");
		}
	}
	return arguments;
}

int main(int argc, char** argv)
{
	#pragma region temp
	//m_trt_file_path = L"D:\\trt\\pytorch.trt";
	//m_input_image_path = "D:\\trt\\122111520150620_7_EdgeDown_ML.bmp";
	//m_dl_framework = "pytorch-trt";
	//m_channel_order = "rgb";
	//m_flag_roi = true;
	//m_norm = "max";
	//m_roi_x = 2048;
	//m_roi_y = 0;
	//m_roi_width = 512;
	//m_roi_height = 512;
	//m_output_path = "D:\\outputs\\pytorch.png";
	#pragma endregion

	clock_t start, end;
	double result;

	start = clock();

	Arguments args = argsToArgumentsMap(argc, argv);
	const int kCnt = args.size();

	bool bCppMode = (kCnt > 0) ? true : false;
	
	if (bCppMode) {

		auto getAndDelNegOption = [&](Arguments& arguments, const std::string& option, bool& value)
		{
			bool dummy;
			if (getAndDelOption(arguments, option, dummy))
			{
				value = false;
				return true;
			}
			return false;
		};

		int nGPUCount = -1;
		{
			cudaGetDeviceCount(&nGPUCount);
			std::cout << std::to_string(nGPUCount) << "개의 GPU 발견 \n";
			std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
			Sleep(1);
			if (nGPUCount < 1) {
				std::wcout << L"Empty GPU \n";
				return -1;
			}
		}

		std::thread* threads = new std::thread[nGPUCount];

		std::vector<std::string> vecStrModelName, vecStrSaveModelName, vecStrWorkSpace, vecStrDevice;
		std::string str;
		bool fp16{ false };

		for (int i = 0; i < kCnt; i++)
		{
			if (getAndDelOption(args, "--onnx", str))			vecStrModelName.push_back(str);
			if (getAndDelOption(args, "--saveEngine", str))		vecStrSaveModelName.push_back(str);
			if (getAndDelOption(args, "--workspace", str))		vecStrWorkSpace.push_back(str);
			if (getAndDelOption(args, "--device", str))			vecStrDevice.push_back(str);
			if (getAndDelOption(args, "--fp16", fp16)); // 다중 onnx to trt 변환 시, 1개라도 fp16이 존재하면 fp16 mode로 변환하게끔... 모델마다 fp mode를 각각 사용하는 건 추후 수정
		}

		// minmum gpu cnt 1
		if (vecStrDevice.size() == 0 && nGPUCount > 0)
		{
			str = '0';
			vecStrDevice.push_back(str);
		}

		int tempGpuCnt = 0;
		for (int i = 0; i < vecStrModelName.size(); i++) {
			try
			{
				if (std::stoi(vecStrDevice[i]) >= nGPUCount)
					tempGpuCnt = (nGPUCount - 1);
				else
					tempGpuCnt = std::stoi(vecStrDevice[i]);

				threads[i] = std::thread(convertONNXtoTRT, vecStrModelName[i], vecStrSaveModelName[i], vecStrWorkSpace[i], tempGpuCnt, fp16);
				tempGpuCnt++;
				std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
				Sleep(500);
			}
			catch (const std::exception& e)
			{
				std::fprintf(stderr, "Error: %s\n", e.what());
				return 1;
			}
		}

		// join
		for (int j = 0; j < tempGpuCnt; j++) {
			threads[j].join();
		}
	}
	else {

		m_trt_file_path = L"D:\\test\\4090.trt";
		m_input_image_path = "D:\\test\\122100415212389_7_Outer5_1.bmp";
		m_dl_framework = "tensorflow-trt";
		m_channel_order = "bgr";
		m_flag_roi = true;
		m_norm = "";
		m_roi_x = 320;
		m_roi_y = 640;
		m_roi_width = 1024;
		m_roi_height = 1024;
		m_output_path = "D:\\outputs\\4090.png";

		m_agent = new Agent();
		PreProcess* pre_process;
		pre_process = new PreProcess();
		if (m_agent->LoadDLModel(m_trt_file_path, m_dl_framework)) {

			std::cout << "* Successfully Loaded TRT Model:  " << m_trt_file_path.c_str() << std::endl;

			m_input_image = cv::imread(m_input_image_path, -1);
			if (!m_input_image.data) {
				std::cout << "* ERROR) There is no such image file: " << m_input_image_path << std::endl;;
				return -1;
			}

			if (m_flag_roi) {
				//crop = image(cv::Rect(m_roi_x, m_roi_y, m_roi_width, m_roi_height)).clone();
				cv::Rect bounds(0, 0, m_input_image.cols, m_input_image.rows);
				cv::Rect roi(m_roi_x, m_roi_y, m_roi_width, m_roi_height); // partly outside
				m_input_image = m_input_image(roi & bounds).clone(); // cropped to fit image
			}

			pre_process->CvNormalize(m_input_image, m_norm);
			pre_process->CvConvertChannelOrder(m_input_image, m_norm);

			m_input_images.push_back(m_input_image);

			if (m_agent->DoInference(m_input_images, m_dl_framework, m_output_path)) {
				std::cout << "*** Successfully run inference !!!" << std::endl;
			}
			else
				std::cout << "* ERROR) There is error in inference process .................." << std::endl;
		}
		else {
			std::cout << "* ERROR) Cannot Load trt file: " << m_trt_file_path.c_str() << std::endl;
		}
	}

	end = clock(); // end
	result = (double)(end - start);

	// Convert Processing Time
	std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << "result : " << ((result) / CLOCKS_PER_SEC) << " seconds" << std::endl;
	std::cout << "------------------------------------------------------------------------------------------------------" << std::endl;

	return 0;
}

