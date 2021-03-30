#pragma once

#include "renderer.h"

BEGIN_RENDERER_NAMESPACE

struct RendererCpu
{
	static RendererCpu& Instance();
	
	void renderForward(
		const kernel::RendererInputs& inputs,
		kernel::RendererOutputs& outputs,
		int B, int W, int H,
		kernel::VolumeFilterMode volumeFilterMode,
		kernel::CameraMode cameraMode,
		kernel::TFMode tfMode,
		kernel::BlendMode blendMode);

	void renderForwardGradients(
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

	void forwardVariablesToGradients(
		const kernel::ForwardDifferencesOutput& forwardVariables,
		const kernel::AdjointColor_t& gradientOutputColor,
		const kernel::ForwardDifferencesSettings& differencesSettings,
		kernel::AdjointOutputs& adj_outputs,
		int B, int W, int H);

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

	void renderAdjoint(
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
		bool hasVolumeDerivative);
};

END_RENDERER_NAMESPACE
