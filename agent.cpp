#include "agent.h"
#include "atlconv.h"

bool Agent::UnloadDLModel()
{
	if (m_engine)
		m_engine->destroy();
	return true;
}

bool Agent::LoadDLModel(std::wstring TRTFilePath, std::string dl_framework)
{
	std::wstringstream wstreamLogMsg;

	IRuntime* runtime = createInferRuntime(sample::gLogger);
	assert(runtime != nullptr);

	std::vector<char> buffer = ReadBuffer(TRTFilePath);
	if (buffer.size())
		// try to deserialize engine        
		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr), samplesCommon::InferDeleter());

	m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
	if (!m_context) {
		return false;
	}


	auto nGetNbBindings = m_engine->getNbBindings();
	if (nGetNbBindings < 2) throw std::runtime_error("Trt model should have input and output, not ");

	for (int i = 0; i < nGetNbBindings; i++)
	{
		auto dimGetBindingDimensions = m_engine->getBindingDimensions(i);
		if (m_engine->bindingIsInput(i))
		{
			m_input_name = m_engine->getBindingName(i);

			if (i == 0) {
				m_InputDims = dimGetBindingDimensions;
				if (dl_framework == "pytorch-trt") {
					m_input_batchsize = m_InputDims.d[0];
					m_input_width = m_InputDims.d[3];
					m_input_height = m_InputDims.d[2];
					m_input_channel = m_InputDims.d[1];

				}
				else if (dl_framework == "tensorflow-trt") {
					m_input_batchsize = m_InputDims.d[0];
					m_input_width = m_InputDims.d[1];
					m_input_height = m_InputDims.d[2];
					m_input_channel = m_InputDims.d[3];
				}
			}
			else {
				m_InputDims_Threshold = dimGetBindingDimensions;

				std::cout << "* ERROR) NOT YET IMPLEMENTED for multiple inputs" << std::endl;
				return -1;
			}

		}
		else {
			m_output_name = m_engine->getBindingName(i);
			m_OutputDims = dimGetBindingDimensions;
		}
	}

	std::cout << "* From trt file) input Trt Name: " << m_input_name << std::endl;
	std::cout << "* From trt file) output Trt Name: " << m_output_name << std::endl;

	if (dl_framework == "pytorch-trt") { // bs, ch, w, h
		m_output_batchsize = m_engine->getBindingDimensions(1).d[0]; // output
		m_output_width = m_engine->getBindingDimensions(1).d[3]; // output
		m_output_height = m_engine->getBindingDimensions(1).d[2]; // output
		m_output_channel = m_engine->getBindingDimensions(1).d[1]; // output
	}
	else if (dl_framework == "tensorflow-trt") { // bs, w, h, ch
		m_output_batchsize = m_engine->getBindingDimensions(1).d[0]; // output
		m_output_width = m_engine->getBindingDimensions(1).d[1]; // output
		m_output_height = m_engine->getBindingDimensions(1).d[2]; // output
		m_output_channel = m_engine->getBindingDimensions(1).d[3]; // output
	}
	else {
		std::cout << "ERROR) There is no such dl-framework: " << dl_framework << std::endl;
		return -1;
	}

	std::cout << "** From trt file) batch-size: " << m_output_batchsize << std::endl;
	std::cout << "** From trt file) channel: " << m_output_channel << std::endl;
	std::cout << "** From trt file) width: " << m_output_width << std::endl;
	std::cout << "** From trt file) height: " << m_output_height << std::endl;

	return true;
}

std::vector<char> Agent::ReadBuffer(wstring const& path)
{
	std::vector<char> buffer;
	std::ifstream stream(path.c_str(), ios::binary);
	size_t size{ 0 };

	if (stream.good())
	{
		stream >> noskipws;
		stream.seekg(0, stream.end);
		size = stream.tellg();
		stream.seekg(0, stream.beg);
		buffer.resize(size);

		stream.read(buffer.data(), size);
		stream.close();
	}
	return buffer;
}

bool Agent::DoTensorflowTrtInference(std::vector<cv::Mat> input_batch_images, std::string output_path)
{
	m_tic = std::chrono::high_resolution_clock::now();
	vector<cv::Mat> temp;
	cv::split(input_batch_images[0], temp);
	for (int row = 0; row < m_input_height; row++)
	{
		for (int col = 0; col < m_input_width; col++)
		{
			for (int ch = 0; ch < m_input_channel; ch++)
			{
				m_host_input_buffer[(row * m_input_width + col) * 3 + ch] = temp[ch].at<uchar>(row, col);
			}
		}
	}
	m_tac = std::chrono::high_resolution_clock::now();
	m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_tac - m_tic);
	std::cout << "- cv-image to buffer: " << m_duration.count()/1000 << " ms" << std::endl;

	// Copy from CPU to GPU
	m_tic = std::chrono::high_resolution_clock::now();
	m_pBuffers->copyInputToDevice();
	m_tac = std::chrono::high_resolution_clock::now();
	m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_tac - m_tic);
	std::cout << "- copy input cpu to gpu: " << m_duration.count()/1000 << " ms" << std::endl;

	m_tic = std::chrono::high_resolution_clock::now();
	bool status = m_context->executeV2(m_pBuffers->getDeviceBindings().data());
	if (!status) {
		return false;
	}
	m_tac = std::chrono::high_resolution_clock::now();
	m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_tac - m_tic);
	std::cout << "- inference: " << m_duration.count() / 1000 << " ms" << std::endl;
	m_pBuffers->copyOutputToHost();

	std::vector<float> result;
	result.resize(m_output_channel * m_output_height * m_output_width);
	memcpy(result.data(), reinterpret_cast<float*>(m_host_output_buffer), m_output_channel * m_output_height * m_output_width * sizeof(float));

	std::vector <cv::Mat>stacks;
	m_tic = std::chrono::high_resolution_clock::now();
	for (int c = 1; c < m_output_channel; ++c) {
		cv::Mat outputImage(m_output_height, m_output_width, CV_8UC1);
		for (int h = 0; h < m_output_height; h++) {
			for (int w = 0; w < m_output_width; w++) {
				int maxClassIdx = 0;
				float maxClassProb = result[h * m_output_width + w];

				outputImage.at<uchar>(h, w) = (uchar)(m_host_output_buffer[((h * m_output_width) + w) * m_output_channel + c] * 255);
			}
		}
		stacks.push_back(outputImage);
	}

	cv::Mat _outputImage(m_output_height, m_output_width, CV_8UC1);

	for (int h = 0; h < m_output_height; h++) {
		for (int w = 0; w < m_output_width; w++) {
			std::vector <int>max_idx;
			for (int i = 0; i < stacks.size(); i++) {
				max_idx.push_back(stacks[i].at<uchar>(h, w));
			}

			_outputImage.at<uchar>(h, w) = *std::max_element(max_idx.begin(), max_idx.end());
		}
	}
	m_tac = std::chrono::high_resolution_clock::now();
	m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_tac - m_tic);
	std::cout << "- post: " << m_duration.count() / 1000 << " ms" << std::endl;
	//memcpy(outputImage.data, m_host_output_buffer + m_output_height * m_output_width, sizeof(float)* m_output_height* m_output_width);

	cv::imwrite(output_path, _outputImage);
	return true;
}

bool Agent::DoPytorchTrtInference(std::vector<cv::Mat> input_batch_images, std::string output_path)
{
	for (size_t batch = 0; batch < input_batch_images.size(); ++batch) {
		auto image = input_batch_images[batch];

		// Preprocess code
		// NHWC to NCHW conversion
		// NHWC: For each pixel, its 3 colors are stored together in RGB order.
		// For a 3 channel image, say RGB, pixels of the R channel are stored first, then the G channel and finally the B channel.
		// https://user-images.githubusercontent.com/20233731/85104458-3928a100-b23b-11ea-9e7e-95da726fef92.png
		int offset = m_input_channel * m_input_height * m_input_width * batch;
		int r = 0, g = 0, b = 0;
		auto imgPtr = image.ptr<cv::Vec3f>(0);

		const float subR = 0;
		const float divR = 1;
		const float subG = 0;
		const float divG = 1;
		const float subB = 0;
		const float divB = 1;

		for (int i = 0; i < m_input_height * m_input_width; ++i) {
			m_host_input_buffer[offset + r++] = (static_cast<float>(imgPtr[i][0]) - subR) / divR;
			m_host_input_buffer[offset + g++ + m_input_height * m_input_width] = (static_cast<float>(imgPtr[i][1]) - subG) / divG;
			m_host_input_buffer[offset + b++ + m_input_height * m_input_width * 2] = (static_cast<float>(imgPtr[i][2]) - subB) / divB;
		}
	}
	//buffers->copyInputToDevice();

	// Copy from CPU to GPU
	m_pBuffers->copyInputToDevice();
	bool status = m_context->executeV2(m_pBuffers->getDeviceBindings().data());
	if (!status) {
		return false;
	}

	m_pBuffers->copyOutputToHost();

	std::vector<float> result;
	result.resize(m_output_channel * m_output_height * m_output_width);
	memcpy(result.data(), reinterpret_cast<float*>(m_host_output_buffer), m_output_channel * m_output_height * m_output_width * sizeof(float));

	cv::Mat outputImage(m_output_height, m_output_width, CV_8UC1);

	for (int h = 0; h < m_output_height; h++) {
		for (int w = 0; w < m_output_width; w++) {
			int maxClassIdx = 0;
			float maxClassProb = result[h * m_output_width + w];

			for (int c = 1; c < m_output_channel; ++c)
			{
				float prob = result[m_output_height * m_output_width * c + h * m_output_width + w];
				if (prob > maxClassProb)
				{
					maxClassIdx = c;
					maxClassProb = prob;
				}
			}

			// Set the pixel value in the output image based on the max class index
			outputImage.at<uchar>(h, w) = static_cast<uchar>(maxClassIdx * 255 / (m_output_channel - 1));
		}
	}
	cv::imwrite(output_path, outputImage);
	return true;
}

bool Agent::DoInference(const std::vector<cv::Mat>& input_batch_images, std::string dl_framework, std::string output_path)
{
	m_pBuffers = new samplesCommon::BufferManager(m_engine);
	CHECK(cudaStreamCreate(&m_cudaStream));

	if (!m_context->allInputDimensionsSpecified()) {
		throw std::runtime_error("Error, not all input dimensions specified.");
	}

	const samplesCommon::BufferManager& buffers = *m_pBuffers;
	m_host_input_buffer = static_cast<float*>(buffers.getHostBuffer(m_input_name));
	m_host_output_buffer = static_cast<float*>(buffers.getHostBuffer(m_output_name));

	if (dl_framework == "pytorch-trt") {
		if (!DoPytorchTrtInference(input_batch_images, output_path))
			std::cout << "* ERROR) There is error in inference with pytorch-trt" << std::endl;
	}
	else if (dl_framework == "tensorflow-trt") {
		if (!DoTensorflowTrtInference(input_batch_images, output_path))
			std::cout << "* ERROR) There is error in inference with pytorch-trt" << std::endl;
	}
}

std::shared_ptr<nvinfer1::ICudaEngine> Agent::GetEngine()
{
	return m_engine;
}
