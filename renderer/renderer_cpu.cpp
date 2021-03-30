#include "renderer_cpu.h"

#include <cuda_runtime.h>
#include "renderer_kernels.cuh"
#include "renderer_compareToImage.cuh"

#ifndef RENDERER_BUILD_CPU_KERNELS
//build cpu kernels by default
#define RENDERER_BUILD_CPU_KERNELS 1
#endif

renderer::RendererCpu& renderer::RendererCpu::Instance()
{
	static renderer::RendererCpu INSTANCE;
	return INSTANCE;
}

#define CALL_KERNEL(vfm_static, cm_static, tf_static, blend_static, ...)	\
	do {	\
	static constexpr kernel::VolumeFilterMode volumeFilterMode = vfm_static;	\
	static constexpr kernel::CameraMode cameraMode = cm_static;	\
	static constexpr kernel::TFMode tfMode = tf_static;	\
	static constexpr kernel::BlendMode blendMode = blend_static;	\
	__VA_ARGS__ ();	\
	} while(0)

	//kernel::DvrKernelForwardHost	\
	//	<vfm_static, cm_static, tf_static, blend_static> \
	//	(virtual_size, inputs, outputs)

#define SWITCH_VOLUME_FILTER_MODE(vfm, cm_static, tf_static, blend_static, ...)	\
	switch(vfm) {	\
		case kernel::VolumeFilterMode::FilterNearest: \
			CALL_KERNEL(kernel::VolumeFilterMode::FilterNearest, cm_static, tf_static, blend_static, __VA_ARGS__); \
			break;	\
		case kernel::VolumeFilterMode::FilterTrilinear: \
			CALL_KERNEL(kernel::VolumeFilterMode::FilterTrilinear, cm_static, tf_static, blend_static, __VA_ARGS__);	\
			break;	\
	}
//TODO: add tricubic someday

#define SWITCH_CAMERA_MODE(vfm, cm, tf_static, blend_static, ...) \
	switch (cm) {	\
		case kernel::CameraMode::CameraRayStartDir:	\
			SWITCH_VOLUME_FILTER_MODE(vfm, kernel::CameraMode::CameraRayStartDir, tf_static, blend_static, __VA_ARGS__);	\
			break;	\
		case kernel::CameraMode::CameraInverseViewMatrix:	\
			SWITCH_VOLUME_FILTER_MODE(vfm, kernel::CameraMode::CameraInverseViewMatrix, tf_static, blend_static, __VA_ARGS__);	\
			break;	\
		case kernel::CameraMode::CameraReferenceFrame:	\
			SWITCH_VOLUME_FILTER_MODE(vfm, kernel::CameraMode::CameraReferenceFrame, tf_static, blend_static, __VA_ARGS__);	\
			break;	\
	}

#define SWITCH_TF_MODE(vfm, cm, tf, blend_static, ...) \
	switch (tf) {	\
		case kernel::TFMode::TFIdentity: \
			SWITCH_CAMERA_MODE(vfm, cm, kernel::TFMode::TFIdentity, blend_static, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFTexture: \
			SWITCH_CAMERA_MODE(vfm, cm, kernel::TFMode::TFTexture, blend_static, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFLinear: \
			SWITCH_CAMERA_MODE(vfm, cm, kernel::TFMode::TFLinear, blend_static, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFGaussian: \
			SWITCH_CAMERA_MODE(vfm, cm, kernel::TFMode::TFGaussian, blend_static, __VA_ARGS__);	\
			break;	\
	}

#define SWITCH_BLEND_MODE(vfm, cm, tf, blend, ...)	\
	switch (blend) {	\
		case kernel::BlendMode::BlendBeerLambert:	\
			SWITCH_TF_MODE(vfm, cm, tf, kernel::BlendMode::BlendBeerLambert, __VA_ARGS__);	\
			break;	\
		case kernel::BlendMode::BlendAlpha:	\
			SWITCH_TF_MODE(vfm, cm, tf, kernel::BlendMode::BlendAlpha, __VA_ARGS__);	\
			break;	\
	}

void renderer::RendererCpu::renderForward(const kernel::RendererInputs& inputs, kernel::RendererOutputs& outputs,
	int B, int W, int H, kernel::VolumeFilterMode volumeFilterMode, kernel::CameraMode cameraMode,
	kernel::TFMode tfMode, kernel::BlendMode blendMode)
{
#if RENDERER_BUILD_CPU_KERNELS==1
	dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };
	SWITCH_BLEND_MODE(volumeFilterMode, cameraMode, tfMode, blendMode, 
		[virtual_size, inputs, outputs]()
	{
		kernel::DvrKernelForwardHost
			<volumeFilterMode, cameraMode, tfMode, blendMode>
			(virtual_size, inputs, outputs);
	});
#else
	throw std::runtime_error("CPU kernels are disabled, enable via RENDERER_BUILD_CPU_KERNELS option");
#endif
}

//these are the counts that are needed in the unit tests
#define SWITCH_N(N, VariableName, ...)	\
	switch (N) { \
		case 1: {static constexpr int VariableName = 1; __VA_ARGS__; } break;	\
		case 2: {static constexpr int VariableName = 2; __VA_ARGS__; } break;	\
		case 8: {static constexpr int VariableName = 8; __VA_ARGS__; } break;	\
		case 16: {static constexpr int VariableName = 16; __VA_ARGS__; } break;	\
		case 32: {static constexpr int VariableName = 32; __VA_ARGS__; } break;	\
		case 64: {static constexpr int VariableName = 64; __VA_ARGS__; } break;	\
		case 20: {static constexpr int VariableName = 20; __VA_ARGS__; } break;	\
		default: {std::cerr << "Only some number of variables are supported at the moment, not " << N << std::endl;} break;	\
	}

#define CASE_DERIVATIVES(num, sd, cd, tfd, ...)	\
	{	\
	static constexpr int hasStepsizeDerivative = sd;	\
	static constexpr int hasCameraDerivative = cd;		\
	static constexpr int hasTFDerivative = tfd;			\
	SWITCH_N(num, numDerivatives, __VA_ARGS__)	\
	}
#define SWITCH_DERIVATIVES(num, sd, cd, tfd, ...)	\
	do {	\
	if ((sd) && (cd) && (tfd)) CASE_DERIVATIVES(num, true, true, true, __VA_ARGS__)		\
	else if (!(sd) && (cd) && (tfd)) CASE_DERIVATIVES(num, false, true, true, __VA_ARGS__)	\
	else if ((sd) && !(cd) && (tfd)) CASE_DERIVATIVES(num, true, false, true, __VA_ARGS__)	\
	else if (!(sd) && !(cd) && (tfd)) CASE_DERIVATIVES(num, false, false, true, __VA_ARGS__)	\
	else if ((sd) && (cd) && !(tfd)) CASE_DERIVATIVES(num, true, true, false, __VA_ARGS__)		\
	else if (!(sd) && (cd) && !(tfd)) CASE_DERIVATIVES(num, false, true, false, __VA_ARGS__)	\
	else if ((sd) && !(cd) && !(tfd)) CASE_DERIVATIVES(num, true, false, false, __VA_ARGS__)	\
	} while(0)

void renderer::RendererCpu::renderForwardGradients(
	const kernel::RendererInputs& inputs, const kernel::ForwardDifferencesSettings& settings,
	kernel::RendererOutputs& outputs, kernel::ForwardDifferencesOutput& gradients, 
	int B, int W, int H, 
	kernel::VolumeFilterMode volumeFilterMode, kernel::CameraMode cameraMode, 
	kernel::TFMode tfMode, kernel::BlendMode blendMode, 
	int numDerivatives, bool hasStepsizeDerivative,
	bool hasCameraDerivative, bool hasTFDerivative)
{
#if RENDERER_BUILD_CPU_KERNELS==1
	dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };

	SWITCH_DERIVATIVES(numDerivatives, hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative,
		SWITCH_BLEND_MODE(volumeFilterMode, cameraMode, tfMode, blendMode,
		[virtual_size, inputs, settings, outputs, gradients]()
	{
		kernel::DvrKernelForwardGradientsHost
			<volumeFilterMode, cameraMode, tfMode, blendMode,
			numDerivatives, hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative>
			(virtual_size, inputs, settings, outputs, gradients);
	}));
#else
	throw std::runtime_error("CPU kernels are disabled, enable via RENDERER_BUILD_CPU_KERNELS option");
#endif
}

template<int D>
void CompareToImage_NoReduce_CPU_Kernel(
	const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput,
	const kernel::Tensor4Read& colorReference,
	kernel::Tensor4RW& differenceOut,
	kernel::Tensor4RW& gradientsOut,
	int B, int W, int H)
{
	using namespace kernel;
	
	dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int b = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * b);
		
		cudAD::fvar<real4, D> input = CompareToImage<D>::readInput(
			colorInput, gradientsInput, b, y, x);
		real4 reference = CompareToImage<D>::readReference(
			colorReference, b, y, x);
		cudAD::fvar<real_t, D> result = CompareToImage<D>::compare(input, reference);
		CompareToImage<D>::writeOutput(result, differenceOut, gradientsOut, b, y, x);

		//debug
		if (x == 500 && y == 400)
		{
			printf("[%d, %d]: input = (%.3f, %.3f, %.3f, %.3f) [%.3f, %.3f, %.3f, %.3f]\n",
				int(x), int(y), input.value().x, input.value().y, input.value().z, input.value().w,
				input.derivative(0).x, input.derivative(0).y, input.derivative(0).z, input.derivative(0).w);
			printf("[%d, %d]: reference = (%.3f, %.3f, %.3f, %.3f)\n",
				int(x), int(y), reference.x, reference.y, reference.z, reference.w);
			printf("[%d, %d]: result = %.4f [%.4f]\n",
				int(x), int(y), result.value(), result.derivative(0));
		}
	}
}

template<int D>
void CompareToImage_WithReduce_CPU_Kernel(
	const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput,
	const kernel::Tensor4Read& colorReference,
	kernel::Tensor1RW& differenceOut,
	kernel::Tensor1RW& gradientsOut,
	int B, int W, int H)
{
	using namespace kernel;

	dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };
	int count = virtual_size.x * virtual_size.y * virtual_size.z;

	cudAD::fvar<real_t, D> globalAcc;
	real_t normalization = real_t(1) / real_t(B * W * H);
	
#pragma omp parallel
	{
		cudAD::fvar<real_t, D> localAcc;
#pragma omp for
		for (int __i = 0; __i < count; ++__i)
		{
			int b = __i / (virtual_size.x * virtual_size.y);
			int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
			int x = __i - virtual_size.x * (y + virtual_size.y * b);

			cudAD::fvar<real4, D> input = CompareToImage<D>::readInput(
				colorInput, gradientsInput, b, y, x);
			real4 reference = CompareToImage<D>::readReference(
				colorReference, b, y, x);
			cudAD::fvar<real_t, D> result = CompareToImage<D>::compare(input, reference);
			localAcc += result * normalization;
		}
#pragma omp critical
		{
			globalAcc += localAcc;
		}
	}

	CompareToImage<D>::writeOutput(globalAcc, differenceOut, gradientsOut);
}

void renderer::RendererCpu::compareToImage_NoReduce(const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput, const kernel::Tensor4Read& colorReference,
	kernel::Tensor4RW& differenceOut, kernel::Tensor4RW& gradientsOut, int B, int W, int H, int D)
{
#if RENDERER_BUILD_CPU_KERNELS==1
	SWITCH_N(D, NumDerivatives,
		CompareToImage_NoReduce_CPU_Kernel<NumDerivatives>(
			colorInput, gradientsInput, colorReference, differenceOut, gradientsOut,
			B, W, H)
	);
#else
	throw std::runtime_error("CPU kernels are disabled, enable via RENDERER_BUILD_CPU_KERNELS option");
#endif
}

void renderer::RendererCpu::compareToImage_WithReduce(const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput, const kernel::Tensor4Read& colorReference,
	kernel::Tensor1RW& differenceOut, kernel::Tensor1RW& gradientsOut, int B, int W, int H, int D)
{
#if RENDERER_BUILD_CPU_KERNELS==1
	SWITCH_N(D, NumDerivatives,
		CompareToImage_WithReduce_CPU_Kernel<NumDerivatives>(
			colorInput, gradientsInput, colorReference, differenceOut, gradientsOut,
			B, W, H)
	);
#else
	throw std::runtime_error("CPU kernels are disabled, enable via RENDERER_BUILD_CPU_KERNELS option");
#endif
}

#define CASE_ADJOINT(sd, cd, tfd, vd, ...)	\
	{	\
	static constexpr int hasStepsizeDerivative = sd;	\
	static constexpr int hasCameraDerivative = cd;		\
	static constexpr int hasTFDerivative = tfd;			\
	static constexpr int hasVolumeDerivative = vd;		\
	__VA_ARGS__;	\
	}
#define SWITCH_ADJOINT(sd, cd, tfd, vd, ...)	\
	do {	\
	     if ( (sd) &&  (cd) &&  (tfd) &&  (vd)) CASE_ADJOINT(true , true , true , true , __VA_ARGS__)		\
	else if (!(sd) &&  (cd) &&  (tfd) &&  (vd)) CASE_ADJOINT(false, true , true , true , __VA_ARGS__)		\
	else if ( (sd) && !(cd) &&  (tfd) &&  (vd)) CASE_ADJOINT(true , false, true , true , __VA_ARGS__)		\
	else if (!(sd) && !(cd) &&  (tfd) &&  (vd)) CASE_ADJOINT(false, false, true , true , __VA_ARGS__)		\
	else if ( (sd) &&  (cd) && !(tfd) &&  (vd)) CASE_ADJOINT(true , true , false, true , __VA_ARGS__)		\
	else if (!(sd) &&  (cd) && !(tfd) &&  (vd)) CASE_ADJOINT(false, true , false, true , __VA_ARGS__)		\
	else if ( (sd) && !(cd) && !(tfd) &&  (vd)) CASE_ADJOINT(true , false, false, true , __VA_ARGS__)		\
	else if (!(sd) && !(cd) && !(tfd) &&  (vd)) CASE_ADJOINT(false, false, false, true , __VA_ARGS__)		\
	else if ( (sd) &&  (cd) &&  (tfd) && !(vd)) CASE_ADJOINT(true , true , true , false, __VA_ARGS__)		\
	else if (!(sd) &&  (cd) &&  (tfd) && !(vd)) CASE_ADJOINT(false, true , true , false, __VA_ARGS__)		\
	else if ( (sd) && !(cd) &&  (tfd) && !(vd)) CASE_ADJOINT(true , false, true , false, __VA_ARGS__)		\
	else if (!(sd) && !(cd) &&  (tfd) && !(vd)) CASE_ADJOINT(false, false, true , false, __VA_ARGS__)		\
	else if ( (sd) &&  (cd) && !(tfd) && !(vd)) CASE_ADJOINT(true , true , false, false, __VA_ARGS__)		\
	else if (!(sd) &&  (cd) && !(tfd) && !(vd)) CASE_ADJOINT(false, true , false, false, __VA_ARGS__)		\
	else if ( (sd) && !(cd) && !(tfd) && !(vd)) CASE_ADJOINT(true , false, false, false, __VA_ARGS__)		\
	else if (!(sd) && !(cd) && !(tfd) && !(vd)) {throw std::runtime_error("At least one adjoint variable required. This should be already caught in renderer.cpp");}		\
	} while(0)

void renderer::RendererCpu::renderAdjoint(const kernel::RendererInputs& inputs, const kernel::RendererOutputsAsInput& outputsFromForward, const kernel::AdjointColor_t& adj_color, kernel::AdjointOutputs& adj_outputs, int B, int W, int H, kernel::VolumeFilterMode volumeFilterMode, kernel::CameraMode cameraMode, kernel::TFMode tfMode, kernel::BlendMode blendMode, bool hasStepsizeDerivative, bool hasCameraDerivative, bool hasTFDerivative, bool hasVolumeDerivative)
{
#if RENDERER_BUILD_CPU_KERNELS==1
	dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };
	
	SWITCH_ADJOINT(hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative, hasVolumeDerivative,
		SWITCH_BLEND_MODE(volumeFilterMode, cameraMode, tfMode, blendMode,
		[virtual_size, inputs, outputsFromForward, adj_color, adj_outputs]()
	{
		kernel::DvrKernelAdjointHost
			<volumeFilterMode, cameraMode, tfMode, blendMode,
			hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative, hasVolumeDerivative>
			(virtual_size, inputs, outputsFromForward, adj_color, adj_outputs);
	}));
#else
	throw std::runtime_error("CPU kernels are disabled, enable via RENDERER_BUILD_CPU_KERNELS option");
#endif
}

void renderer::RendererCpu::forwardVariablesToGradients(const kernel::ForwardDifferencesOutput& forwardVariables,
	const kernel::AdjointColor_t& gradientOutputColor,
	const kernel::ForwardDifferencesSettings& differencesSettings, kernel::AdjointOutputs& adj_outputs, int B,
	int W, int H)
{
    dim3 virtual_size{ static_cast<unsigned int>(W), static_cast<unsigned int>(H), static_cast<unsigned int>(B) };
    kernel::ForwardVariablesToGradientsHost(virtual_size, forwardVariables, gradientOutputColor, differencesSettings, adj_outputs);
}
