#include "renderer.h"
#include "renderer_cuda.h"

#include <cuMat/src/Context.h>
#include <iostream>
#include <filesystem>
#include <nvrtc.h>
#include <sstream>
#include <fstream>
#include <magic_enum.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>

#include "renderer_kernels.cuh"
#include "sha1.h"
#include "kernel_loader.h"

namespace fs = std::filesystem;

renderer::RendererCuda& renderer::RendererCuda::Instance()
{
	static renderer::RendererCuda INSTANCE;
	return INSTANCE;
}

bool renderer::RendererCuda::initCuda()
{
	return KernelLoader::Instance().initCuda();
}

void renderer::RendererCuda::setCudaCacheDir(const std::filesystem::path& path)
{
	KernelLoader::Instance().setCudaCacheDir(path);
}

void renderer::RendererCuda::disableCudaCache()
{
	KernelLoader::Instance().disableCudaCache();
}

void renderer::RendererCuda::reloadCudaKernels()
{
	KernelLoader::Instance().reloadCudaKernels();
}

void renderer::RendererCuda::cleanup()
{
	KernelLoader::Instance().cleanup();
}

void renderer::RendererCuda::setSyncMode(bool sync)
{
	syncMode = sync;
}
bool renderer::RendererCuda::getSyncMode() const
{
	return syncMode;
}

void renderer::RendererCuda::setDebugMode(bool debug)
{
	KernelLoader::Instance().setDebugMode(debug);
}

std::string renderer::RendererCuda::getForwardKernelName(kernel::VolumeFilterMode volumeFilterMode,
                                                         kernel::CameraMode cameraMode, kernel::TFMode tfMode, kernel::BlendMode blendMode)
{
	//build kernel name
	const auto volumeFilterModeName = magic_enum::enum_name(volumeFilterMode);
	const auto cameraModeName = magic_enum::enum_name(cameraMode);
	const auto tfModeName = magic_enum::enum_name(tfMode);
	const auto blendModeName = magic_enum::enum_name(blendMode);

	std::stringstream ss;
	ss << "kernel::DvrKernelForwardDevice<"
		<< "kernel::VolumeFilterMode::" << volumeFilterModeName
		<< ", kernel::CameraMode::" << cameraModeName
		<< ", kernel::TFMode::" << tfModeName
		<< ", kernel::BlendMode::" << blendModeName
		<< ">";
	std::string kernelName = ss.str();
	return kernelName;
}

bool renderer::RendererCuda::renderForward(const kernel::RendererInputs& inputs, kernel::RendererOutputs& outputs,
                                           int B, int W, int H, kernel::VolumeFilterMode volumeFilterMode, kernel::CameraMode cameraMode,
                                           kernel::TFMode tfMode, kernel::BlendMode blendMode)
{
	std::string kernelName = getForwardKernelName(volumeFilterMode, cameraMode, tfMode, blendMode);

	auto kernelFunctionOrOptional = KernelLoader::Instance().getKernelFunction(kernelName);
	if (!kernelFunctionOrOptional.has_value()) return false;
	KernelLoader::KernelFunction kernelFunction = kernelFunctionOrOptional.value();

	if (B == 0 && W == 0 && H == 0) return true; //Unit-test support to check if it can be compiled

	int minGridSize = std::min(
		int(CUMAT_DIV_UP(W*H*B, kernelFunction.bestBlockSize())),
		kernelFunction.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(W),
		static_cast<unsigned int>(H),
		static_cast<unsigned int>(B) };
	const void* args[] = { &virtual_size, &inputs, &outputs };
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	auto result = cuLaunchKernel(
		kernelFunction.fun(), minGridSize, 1, 1, kernelFunction.bestBlockSize(), 1, 1,
		0, stream, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS)
		return printError(result, kernelName);
	if (syncMode)
	{
		result = cuCtxSynchronize();
		if (result != CUDA_SUCCESS)
			return printError(result, kernelName);
	}
	return true;
}

std::string renderer::RendererCuda::getForwardGradientsKernelName(kernel::VolumeFilterMode volumeFilterMode,
	kernel::CameraMode cameraMode, kernel::TFMode tfMode, kernel::BlendMode blendMode, int numDerivatives,
	bool hasStepsizeDerivative, bool hasCameraDerivative, bool hasTFDerivative)
{
	//build kernel name
	const auto volumeFilterModeName = magic_enum::enum_name(volumeFilterMode);
	const auto cameraModeName = magic_enum::enum_name(cameraMode);
	const auto tfModeName = magic_enum::enum_name(tfMode);
	const auto blendModeName = magic_enum::enum_name(blendMode);

	std::stringstream ss;
	ss << "kernel::DvrKernelForwardGradientsDevice<"
		<< "kernel::VolumeFilterMode::" << volumeFilterModeName
		<< ", kernel::CameraMode::" << cameraModeName
		<< ", kernel::TFMode::" << tfModeName
		<< ", kernel::BlendMode::" << blendModeName
		<< ", " << numDerivatives
		<< ", " << hasStepsizeDerivative
		<< ", " << hasCameraDerivative
		<< ", " << hasTFDerivative
		<< ">";
	std::string kernelName = ss.str();
	return kernelName;
}

bool renderer::RendererCuda::renderForwardGradients(const kernel::RendererInputs& inputs,
                                                    const kernel::ForwardDifferencesSettings& settings, kernel::RendererOutputs& outputs,
                                                    kernel::ForwardDifferencesOutput& gradients, int B, int W, int H, kernel::VolumeFilterMode volumeFilterMode,
                                                    kernel::CameraMode cameraMode, kernel::TFMode tfMode, kernel::BlendMode blendMode, int numDerivatives,
                                                    bool hasStepsizeDerivative, bool hasCameraDerivative, bool hasTFDerivative)
{
	//build kernel name
	std::string kernelName = getForwardGradientsKernelName(
		volumeFilterMode, cameraMode, tfMode, blendMode,
		numDerivatives, hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative);

	auto kernelFunctionOrOptional = KernelLoader::Instance().getKernelFunction(kernelName);
	if (!kernelFunctionOrOptional.has_value()) return false;
	KernelLoader::KernelFunction kernelFunction = kernelFunctionOrOptional.value();

	if (B == 0 && W == 0 && H == 0) return true; //Unit-test support to check if it can be compiled

	int minGridSize = std::min(
		int(CUMAT_DIV_UP(W * H * B, kernelFunction.bestBlockSize())),
		kernelFunction.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(W),
		static_cast<unsigned int>(H),
		static_cast<unsigned int>(B) };
	const void* args[] = { &virtual_size, &inputs, &settings, &outputs, &gradients };
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	auto result = cuLaunchKernel(
		kernelFunction.fun(), minGridSize, 1, 1, kernelFunction.bestBlockSize(), 1, 1,
		0, stream, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS)
		return printError(result, kernelName);
	if (syncMode)
	{
		result = cuCtxSynchronize();
		if (result != CUDA_SUCCESS)
			return printError(result, kernelName);
	}
	return true;
}

bool renderer::RendererCuda::forwardVariablesToGradients(const kernel::ForwardDifferencesOutput& forwardVariables,
	const kernel::AdjointColor_t& gradientOutputColor, const kernel::ForwardDifferencesSettings& differencesSettings,
	kernel::AdjointOutputs& adj_outputs, int B, int W, int H)
{
	//build kernel name
	std::string kernelName = getForwardVariablesToGradientsName();

	auto kernelFunctionOrOptional = KernelLoader::Instance().getKernelFunction(kernelName);
	if (!kernelFunctionOrOptional.has_value()) return false;
	KernelLoader::KernelFunction kernelFunction = kernelFunctionOrOptional.value();

	if (B == 0 && W == 0 && H == 0) return true; //Unit-test support to check if it can be compiled

	int minGridSize = std::min(
		int(CUMAT_DIV_UP(W * H * B, kernelFunction.bestBlockSize())),
		kernelFunction.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(W),
		static_cast<unsigned int>(H),
		static_cast<unsigned int>(B) };
	const void* args[] = { &virtual_size, &forwardVariables, &gradientOutputColor, &differencesSettings, &adj_outputs};
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	auto result = cuLaunchKernel(
		kernelFunction.fun(), minGridSize, 1, 1, kernelFunction.bestBlockSize(), 1, 1,
		0, stream, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS)
		return printError(result, kernelName);
	if (syncMode)
	{
		result = cuCtxSynchronize();
		if (result != CUDA_SUCCESS)
			return printError(result, kernelName);
	}
	return true;
}

std::string renderer::RendererCuda::getForwardVariablesToGradientsName()
{
	return "kernel::ForwardVariablesToGradientsDevice";
}

std::string renderer::RendererCuda::getAdjointKernelName(kernel::VolumeFilterMode volumeFilterMode,
	kernel::CameraMode cameraMode, kernel::TFMode tfMode, kernel::BlendMode blendMode, bool hasStepsizeDerivative,
	bool hasCameraDerivative, bool hasTFDerivative, bool tfDelayedAccumulation, bool hasVolumeDerivative)
{
	const auto volumeFilterModeName = magic_enum::enum_name(volumeFilterMode);
	const auto cameraModeName = magic_enum::enum_name(cameraMode);
	const auto tfModeName = magic_enum::enum_name(tfMode);
	const auto blendModeName = magic_enum::enum_name(blendMode);

	std::stringstream ss;
	ss << "kernel::DvrKernelAdjointDevice<"
		<< "kernel::VolumeFilterMode::" << volumeFilterModeName
		<< ", kernel::CameraMode::" << cameraModeName
		<< ", kernel::TFMode::" << tfModeName
		<< ", kernel::BlendMode::" << blendModeName
		<< ", " << hasStepsizeDerivative
		<< ", " << hasCameraDerivative
		<< ", " << hasTFDerivative
		<< ", " << tfDelayedAccumulation
		<< ", " << hasVolumeDerivative
		<< ">";
	std::string kernelName = ss.str();
	return kernelName;
}

bool renderer::RendererCuda::renderAdjoint(
	const kernel::RendererInputs& inputs, 
	const kernel::RendererOutputsAsInput& outputsFromForward, 
	const kernel::AdjointColor_t& adj_color, 
	kernel::AdjointOutputs& adj_outputs, 
	int B, int W, int H, 
	kernel::VolumeFilterMode volumeFilterMode, kernel::CameraMode cameraMode, 
	kernel::TFMode tfMode, kernel::BlendMode blendMode, 
	bool hasStepsizeDerivative, bool hasCameraDerivative,
	bool hasTFDerivative, bool tfDelayedAccumulation, bool hasVolumeDerivative)
{
	//build kernel name
	std::string kernelName = getAdjointKernelName(
		volumeFilterMode, cameraMode, tfMode, blendMode,
		hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative, tfDelayedAccumulation, hasVolumeDerivative);

	auto kernelFunctionOrOptional = KernelLoader::Instance().getKernelFunction(kernelName);
	if (!kernelFunctionOrOptional.has_value()) return false;
	KernelLoader::KernelFunction kernelFunction = kernelFunctionOrOptional.value();

	if (B == 0 && W == 0 && H == 0) return true; //Unit-test support to check if it can be compiled

	int sharedMemoryPerThread = sizeof(real_t) * 
		adj_outputs.adj_tf.size(1) * adj_outputs.adj_tf.size(2);
	int bestBlockSize = kernelFunction.bestBlockSize();
	int sharedMemorySize = 0;
	int minGridSize = kernelFunction.minGridSize();
	if (tfDelayedAccumulation)
	{
		static std::mutex g_mutex;
		static int g_sharedMemoryPerThread;
		std::lock_guard<std::mutex> guard(g_mutex);
		//ugly workaround via global variable since I can't captuare
		//a local variable when calling cuOccupancyMaxPotentialBlockSize 
		g_sharedMemoryPerThread = sharedMemoryPerThread;
		static auto blockToSmem = [](int blockSize)
		{
			return static_cast<size_t>(g_sharedMemoryPerThread * blockSize);
		};
		CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(
			&minGridSize, &bestBlockSize, kernelFunction.fun(), blockToSmem, 0, 0));
		sharedMemorySize = sharedMemoryPerThread * bestBlockSize;
		if (bestBlockSize == 0)
			throw std::runtime_error("too much shared memory requested, can't even fulfill the demands for a single warp");
	}
	minGridSize = std::min(
		int(CUMAT_DIV_UP(W * H * B, bestBlockSize)),
		minGridSize);
	dim3 virtual_size{
		static_cast<unsigned int>(W),
		static_cast<unsigned int>(H),
		static_cast<unsigned int>(B) };
	const void* args[] = { &virtual_size, &inputs, &outputsFromForward, &adj_color, &adj_outputs };
	cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
	auto result = cuLaunchKernel(
		kernelFunction.fun(), minGridSize, 1, 1, bestBlockSize, 1, 1,
		sharedMemorySize, stream, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS)
		return printError(result, kernelName);
	if (syncMode)
	{
		result = cuCtxSynchronize();
		if (result != CUDA_SUCCESS)
			return printError(result, kernelName);
	}
	return true;
}

