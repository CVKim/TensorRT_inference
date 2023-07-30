#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include "argsParser.h"
#include "buffers.h"

#include "logging.h"
#include "parserOnnxConfig.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <chrono>

class Agent
{
public:
	bool LoadDLModel(std::wstring TRTFilePath, std::string dl_framework);
	bool UnloadDLModel();
	bool DoInference(const std::vector<cv::Mat>& inputBatchImages, std::string dl_framework, std::string output_mask);

	bool DoPytorchTrtInference(std::vector<cv::Mat> input_batch_images, std::string output_path);
	bool DoTensorflowTrtInference(std::vector<cv::Mat> input_batch_images, std::string output_path);
	std::shared_ptr<nvinfer1::ICudaEngine> GetEngine();

private:
	std::vector<char> ReadBuffer(wstring const& path);
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr; //!< The TensorRT engine used to run the network
	std::shared_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
	nvinfer1::Dims m_InputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims m_InputDims_Threshold;  //!< The dimensions of the input to the network.
	nvinfer1::Dims m_OutputDims; //!< The dimensions of the output to the network.

	std::vector<int> m_input_dimension; // (batch, width, height, channel)

	size_t m_prevBatchSize = 0;
	samplesCommon::BufferManager* m_pBuffers;
	cudaStream_t m_cudaStream = nullptr;

	std::string m_output_name;
	int m_output_batchsize;
	int m_output_width;
	int m_output_height;
	int m_output_channel;

	std::string m_input_name;
	int m_input_batchsize;
	int m_input_width;
	int m_input_height;
	int m_input_channel;

	float* m_host_input_buffer;
	float* m_host_output_buffer;

	std::chrono::steady_clock::time_point m_tic;
	std::chrono::steady_clock::time_point m_tac;
	std::chrono::microseconds m_duration;
};