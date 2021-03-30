#pragma once

#include <cuda.h>
#include <filesystem>
#include <map>
#include <fstream>
#include <functional>

#include "commons.h"
#include "renderer_settings.cuh"

BEGIN_RENDERER_NAMESPACE

struct RendererCuda : public NonAssignable
{
	static RendererCuda& Instance();

	/**
	 * Loads the dynamic cuda libraries of PyTorch.
	 * Call this in the main executable before any other calls
	 * to the renderer or PyTorch.
	 * \return true if cuda is available and was loaded successfully
	 */
	bool initCuda();
	void setCudaCacheDir(const std::filesystem::path& path);
	void disableCudaCache();
	void reloadCudaKernels();
	void cleanup();
	//in sync-mode, a device sync is issued after each kernel launch (for debugging)
	void setSyncMode(bool sync);
	bool getSyncMode() const;
	void setDebugMode(bool debug);
	
	bool renderForward(
		const kernel::RendererInputs& inputs,
		kernel::RendererOutputs& outputs,
		int B, int W, int H,
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode);
	
	std::string getForwardKernelName(
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode);

	bool renderForwardGradients(
		const kernel::RendererInputs& inputs,
		const kernel::ForwardDifferencesSettings& settings,
		kernel::RendererOutputs& outputs,
		kernel::ForwardDifferencesOutput& gradients,
		int B, int W, int H,
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode,
		int numDerivatives,
		bool hasStepsizeDerivative,
		bool hasCameraDerivative,
		bool hasTFDerivative);

	std::string getForwardGradientsKernelName(
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode,
		int numDerivatives,
		bool hasStepsizeDerivative,
		bool hasCameraDerivative,
		bool hasTFDerivative);

	bool forwardVariablesToGradients(
		const kernel::ForwardDifferencesOutput& forwardVariables,
		const kernel::AdjointColor_t& gradientOutputColor,
		const kernel::ForwardDifferencesSettings& differencesSettings,
		kernel::AdjointOutputs& adj_outputs,
		int B, int W, int H);

	std::string getForwardVariablesToGradientsName();

	void compareToImage_NoReduce(
		const kernel::Tensor4Read& colorInput,
		const kernel::Tensor5Read& gradientsInput,
		const kernel::Tensor4Read& colorReference,
		kernel::Tensor4RW& differenceOut,
		kernel::Tensor4RW& gradientsOut,
		int B, int W, int H, int D);

	void compareToImage_WithReduce(
		const kernel::Tensor4Read& colorInput,
		const kernel::Tensor5Read& gradientsInput,
		const kernel::Tensor4Read& colorReference,
		kernel::Tensor1RW& differenceOut,
		kernel::Tensor1RW& gradientsOut,
		int B, int W, int H, int D);

	bool renderAdjoint(
		const kernel::RendererInputs& inputs,
		const kernel::RendererOutputsAsInput& outputsFromForward,
		const kernel::AdjointColor_t& adj_color,
		kernel::AdjointOutputs& adj_outputs,
		int B, int W, int H,
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode,
		bool hasStepsizeDerivative,
		bool hasCameraDerivative,
		bool hasTFDerivative,
		bool tfDelayedAccumulation,
		bool hasVolumeDerivative);

	std::string getAdjointKernelName(kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode,
		bool hasStepsizeDerivative,
		bool hasCameraDerivative,
		bool hasTFDerivative,
		bool tfDelayedAccumulation,
		bool hasVolumeDerivative);

private:
	bool syncMode = false;

};

END_RENDERER_NAMESPACE
