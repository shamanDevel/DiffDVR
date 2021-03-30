#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_settings.cuh"
#include "renderer_interpolation.cuh"
#include "renderer_tf.cuh"
#include "renderer_blending.cuh"
#include "renderer_camera.cuh"

#ifndef FLT_MAX
#define FLT_MAX          3.402823466e+38F        // max value
#endif

namespace kernel {
	extern __shared__ real_t shared[];

//=======================================================
// forward simple (no gradients)
//=======================================================
	
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode>
__host__ __device__ void DvrKernelForwardImpl(
	int x, int y, int b,
	const RendererInputs& inputs, RendererOutputs& outputs)
{
	//world transformation
	const real3 boxMin = fetchReal3(inputs.boxMin, b);
	const real3 boxSize = fetchReal3(inputs.boxSize, b);
	const real3 voxelSize = boxSize / make_real3(inputs.volumeSize - make_int3(1));

	//camera
	real3 rayStart, rayDir;
	CameraEval<cameraMode>::computeRays(x, y, b, inputs, rayStart, rayDir);

	//stepsize
	const real_t stepsize = inputs.stepSize[b][y][x];
	assert(stepsize > 0);
	assert(stepsize < FLT_MAX);

	//entry, exit points
	real_t tmin, tmax;
	intersectionRayAABB(rayStart, rayDir, boxMin, boxSize, tmin, tmax);
	tmin = rmax(tmin, 0);

	//if (x == 0 && y == 0)
	//	printf("rayStart=(%.3f, %.3f, %.3f), rayDir=(%.3f, %.3f, %.3f)\n",
	//		float(rayStart.x), float(rayStart.y), float(rayStart.z),
	//		float(rayDir.x), float(rayDir.y), float(rayDir.z));
	
	//stepping
	real4 outputColor = make_real4(0);
	int terminationIndex;
	VolumeInterpolation<volumeFilterMode> volumeInterpolation;
	TransferFunctionEval<tfMode> transferFunctionEval;
	for (terminationIndex = 0; ; ++terminationIndex)
	{
		real_t tcurrent = tmin + terminationIndex * stepsize;
		if (tcurrent > tmax) break;
		if (outputColor.w > inputs.blendingEarlyOut) break;
		
		//fetch density
		real3 worldPos = rayStart + tcurrent * rayDir;
		real3 volumePos;
		if constexpr(volumeFilterMode == FilterNetwork)
			volumePos = (worldPos - boxMin); // networks expects it in [0,1], no dependency on the resolution
		else
			volumePos = (worldPos - boxMin) / voxelSize;
		auto density = volumeInterpolation.fetch(
			inputs.volume, inputs.volumeSize, b, volumePos);

		//evaluate transfer function
		real4 colorAbsorption = transferFunctionEval.eval(inputs.tf, b, density);

		//blend into accumulator
		outputColor = Blending<blendMode>::blend(outputColor, colorAbsorption, stepsize);
	}

	//output
	outputs.color[b][y][x][0] = outputColor.x;
	outputs.color[b][y][x][1] = outputColor.y;
	outputs.color[b][y][x][2] = outputColor.z;
	outputs.color[b][y][x][3] = outputColor.w;
	outputs.terminationIndex[b][y][x] = terminationIndex;
	//printf("%d %d, t=(%f, %f) -> %f\n", int(x), int(y), float(tmin), float(tmax), float(outputColor.w));
}
	
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode>
__global__ void DvrKernelForwardDevice(dim3 virtual_size,
	RendererInputs inputs, RendererOutputs outputs)
{
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		DvrKernelForwardImpl<volumeFilterMode, cameraMode, tfMode, blendMode>(
			x, y, b, inputs, outputs);
	}
	KERNEL_3D_LOOP_END
}

#ifndef CUDA_NO_HOST
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode>
__host__ void DvrKernelForwardHost(dim3 virtual_size,
	RendererInputs inputs, RendererOutputs outputs)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int b = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * b);
		DvrKernelForwardImpl<volumeFilterMode, cameraMode, tfMode, blendMode>(
			x, y, b, inputs, outputs);
	}
}
#endif


//=======================================================
// forward gradients
//=======================================================

template<int D, bool HasStepsizeDerivative>
struct FetchStepsize;
template<int D>
struct FetchStepsize<D, false>
{
	typedef real_t value_t;
	static __host__ __device__ __forceinline__ value_t fetch(int x, int y, int b, 
		const RendererInputs& inputs,
		const ForwardDifferencesSettings& settings)
	{
		return inputs.stepSize[b][y][x];
	}
};
template<int D>
struct FetchStepsize<D, true>
{
	typedef cudAD::fvar<real_t, D> value_t;
	static __host__ __device__ __forceinline__ value_t fetch(int x, int y, int b,
		const RendererInputs& inputs,
		const ForwardDifferencesSettings& settings)
	{
		real_t value = FetchStepsize<D, false>::fetch(x, y, b, inputs, settings);
		return cudAD::fvar<real_t, D>::input(value, settings.d_stepsize);
	}
};
	
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	int D, //number of derivatives
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative>
__host__ __device__ void DvrKernelForwardGradientsImpl(
	int x, int y, int b,
	const RendererInputs& inputs,
	const ForwardDifferencesSettings& settings,
	RendererOutputs& outputs,
	ForwardDifferencesOutput& gradOutput)
{
	static_assert(HasStepsizeDerivative || HasCameraDerivative || HasTFDerivative,
		"at least one derivative must be activated");
	
	using namespace cudAD;
	typedef fvar<real_t, D> dreal;
	typedef fvar<real3, D> dreal3;
	typedef fvar<real4, D> dreal4;
	using dstep1 = conditional_t<HasStepsizeDerivative, dreal, real_t>;
	using dcam3 = conditional_t<HasCameraDerivative, dreal3, real3>;
	
	//world transformation
	const real3 boxMin = fetchReal3(inputs.boxMin, b);
	const real3 boxSize = fetchReal3(inputs.boxSize, b);
	const real3 voxelSize = boxSize / make_real3(inputs.volumeSize - make_int3(1));

	//camera
	dcam3 rayStart, rayDir;
	CameraEvalForwardGradients<cameraMode, D, HasCameraDerivative>::computeRays(
		x, y, b, inputs, settings, rayStart, rayDir);

	//stepsize
	const dstep1 stepsize = FetchStepsize<D, HasStepsizeDerivative>::fetch(
		x, y, b, inputs, settings);

	//entry, exit points
	//Do I need to trace derivatives through here?
	//I don't think so as this is just an optimization for the tracing.
	//Therefore, cast to real3, this removes the forward variable tracing
	real_t tmin, tmax;
	intersectionRayAABB(static_cast<real3>(rayStart), static_cast<real3>(rayDir), 
		boxMin, boxSize, tmin, tmax);
	tmin = rmax(tmin, 0);

	//stepping
	dreal4 outputColor = dreal4::constant(make_real4(0));
	int terminationIndex;
	VolumeInterpolation<volumeFilterMode> volumeInterpolation;
	TransferFunctionEval<tfMode> transferFunctionEval;
	for (terminationIndex = 0; ; ++terminationIndex)
	{
		auto tcurrent = tmin + terminationIndex * stepsize;
		if (static_cast<real_t>(tcurrent) > tmax) break;
		if (static_cast<real4>(outputColor).w > inputs.blendingEarlyOut) break;

		//fetch density
		auto worldPos = rayStart + broadcast3(tcurrent) * rayDir;
		auto volumePos = (worldPos - boxMin) / voxelSize;
		auto density = volumeInterpolation.fetch(
			inputs.volume, inputs.volumeSize, b, volumePos);

		//evaluate transfer function
		auto colorAbsorption = transferFunctionEval.template evalForwardGradients<D>(
			inputs.tf, b, density, settings.d_tf, integral_constant<bool, HasTFDerivative>());

		//blend into accumulator
		outputColor = Blending<blendMode>::blend(outputColor, colorAbsorption, stepsize);
	}

	//output - value
	outputs.color[b][y][x][0] = outputColor.value().x;
	outputs.color[b][y][x][1] = outputColor.value().y;
	outputs.color[b][y][x][2] = outputColor.value().z;
	outputs.color[b][y][x][3] = outputColor.value().w;
	//gradients
	for (int d=0; d<D; ++d)
	{
		gradOutput.gradients[b][y][x][d][0] = outputColor.derivative(d).x;
		gradOutput.gradients[b][y][x][d][1] = outputColor.derivative(d).y;
		gradOutput.gradients[b][y][x][d][2] = outputColor.derivative(d).z;
		gradOutput.gradients[b][y][x][d][3] = outputColor.derivative(d).w;
		////debug
		//if (any(isnan(outputColor.derivative(d))))
		//	printf("NaN at x=%d, y=%d, b=%d, derivative=%d\n",
		//		x, y, b, d);
	}
	//termination index
	outputs.terminationIndex[b][y][x] = terminationIndex;
	//printf("%d %d, t=(%f, %f) -> %f\n", int(x), int(y), float(tmin), float(tmax), float(outputColor.w));
}


template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	int D, //number of derivatives
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative>
__global__ void DvrKernelForwardGradientsDevice(dim3 virtual_size,
	RendererInputs inputs, ForwardDifferencesSettings settings, 
	RendererOutputs outputs, ForwardDifferencesOutput gradOutput)
{
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		DvrKernelForwardGradientsImpl<
			volumeFilterMode, cameraMode, tfMode, blendMode,
			D, HasStepsizeDerivative, HasCameraDerivative, HasTFDerivative>(
			x, y, b, inputs, settings, outputs, gradOutput);
	}
	KERNEL_3D_LOOP_END
}

#ifndef CUDA_NO_HOST
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	int D, //number of derivatives
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative>
__host__ void DvrKernelForwardGradientsHost(dim3 virtual_size,
	RendererInputs inputs, ForwardDifferencesSettings settings, 
	RendererOutputs outputs, ForwardDifferencesOutput gradOutput)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int b = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * b);
		DvrKernelForwardGradientsImpl<
			volumeFilterMode, cameraMode, tfMode, blendMode,
			D, HasStepsizeDerivative, HasCameraDerivative, HasTFDerivative>(
				x, y, b, inputs, settings, outputs, gradOutput);
	}
}
#endif
	

//=======================================================
// adjoint code
//=======================================================

template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative, bool TfDelayedAccumulation,
	bool HasVolumeDerivative>
__host__ __device__ void DvrKernelAdjointImpl(
		int x, int y, int b,
		const RendererInputs& inputs, RendererOutputsAsInput& outputsFromForward,
		const AdjointColor_t& adj_color, AdjointOutputs& adj_outputs)
{	
	//world transformation
	const real3 boxMin = fetchReal3(inputs.boxMin, b);
	const real3 boxSize = fetchReal3(inputs.boxSize, b);
	const real3 voxelSize = boxSize / make_real3(inputs.volumeSize - make_int3(1));

	//camera
	real3 rayStart, rayDir;
	CameraEval<cameraMode>::computeRays(x, y, b, inputs, rayStart, rayDir);

	//stepsize
	const real_t stepsize = inputs.stepSize[b][y][x];
	assert(stepsize > 0);
	assert(stepsize < FLT_MAX);
	
	//entry, exit points
	real_t tmin, tmax;
	intersectionRayAABB(rayStart, rayDir, boxMin, boxSize, tmin, tmax);
	tmin = rmax(tmin, 0);

	//fetch result from the forward pass
	real4 outputColor = fetchReal4(outputsFromForward.color, b, y, x);
	int terminationIndex = outputsFromForward.terminationIndex[b][y][x];

	//fetch output adjoint and initialize input adjoint
	real4 adj_outputColor = fetchReal4(adj_color, b, y, x);
	real_t adj_stepSize = 0;
	real3 adj_rayStart = make_real3(0);
	real3 adj_rayDir = make_real3(0);

	VolumeInterpolation<volumeFilterMode, true, HasVolumeDerivative> volumeInterpolation(
		b, inputs.volumeSize, adj_outputs.adj_volume);

	TransferFunctionEval<tfMode> transferFunctionEval;
#ifdef __CUDA_ARCH__
	real_t* tf_shared = nullptr;
	if constexpr (TfDelayedAccumulation)
	{
		tf_shared = shared + (threadIdx.x * inputs.tf.size(1) * inputs.tf.size(2));
		transferFunctionEval.adjointInit(adj_outputs.adj_tf, tf_shared);
	}
#else
	static_assert(!TfDelayedAccumulation, "delayed accumulation only supported in CUDA");
	real_t* tf_shared = nullptr;
#endif
	
	--terminationIndex; //terminationIndex is the index at termination,
	//  hence terminationIndex-1 was the last index that added a color
	for (; terminationIndex>=0; --terminationIndex)
	{
		//run part of the forward code again
		real_t tcurrent = tmin + terminationIndex * stepsize;

		//fetch density
		real3 worldPos = rayStart + tcurrent * rayDir;
		real3 volumePos = (worldPos - boxMin) / voxelSize;
		auto density = volumeInterpolation.fetch(
			inputs.volume, inputs.volumeSize, b, volumePos);
		using density_t = decltype(density);

		//evaluate transfer function
		real4 colorAbsorption = transferFunctionEval.eval(inputs.tf, b, density);

		//adjoint blending
		real4 adj_colorAbsorption;
		outputColor = Blending<blendMode>::adjoint(
			outputColor, colorAbsorption, stepsize, adj_outputColor,
			adj_outputColor, adj_colorAbsorption, adj_stepSize);

		//adjoint transfer function
		density_t adj_density;
		transferFunctionEval.template adjoint<HasTFDerivative, TfDelayedAccumulation>(
			inputs.tf, b, density, adj_colorAbsorption,
			adj_density, adj_outputs.adj_tf, tf_shared);
		
		//adjoint density
		real3 adj_volumePos;
		volumeInterpolation.adjoint(adj_density, adj_volumePos);

		//adjoint stepping
		real3 adj_worldPos = adj_volumePos / voxelSize;
		adj_rayStart += adj_worldPos;
		adj_rayDir += adj_worldPos * tcurrent;
		real_t adj_tcurrent = dot(adj_worldPos, rayDir);
		adj_stepSize += adj_tcurrent * terminationIndex;
	}

	//output
	if constexpr(HasStepsizeDerivative)
	{
		if (adj_outputs.stepSizeHasBroadcasting)
			kernel::atomicAdd(&adj_outputs.adj_stepSize[b][y][x], adj_stepSize);
		else
			adj_outputs.adj_stepSize[b][y][x] = adj_stepSize;
	}
	if constexpr(HasCameraDerivative)
	{
		if (adj_outputs.cameraHasBroadcasting)
		{
			kernel::atomicAdd(&adj_outputs.adj_cameraRayStart[b][y][x][0], adj_rayStart.x);
			kernel::atomicAdd(&adj_outputs.adj_cameraRayStart[b][y][x][1], adj_rayStart.y);
			kernel::atomicAdd(&adj_outputs.adj_cameraRayStart[b][y][x][2], adj_rayStart.z);
			kernel::atomicAdd(&adj_outputs.adj_cameraRayDir[b][y][x][0], adj_rayDir.x);
			kernel::atomicAdd(&adj_outputs.adj_cameraRayDir[b][y][x][1], adj_rayDir.y);
			kernel::atomicAdd(&adj_outputs.adj_cameraRayDir[b][y][x][2], adj_rayDir.z);
		}
		else
		{
			writeReal3(adj_rayStart, adj_outputs.adj_cameraRayStart, b, y, x);
			writeReal3(adj_rayDir, adj_outputs.adj_cameraRayDir, b, y, x);
		}
	}
	if constexpr (TfDelayedAccumulation)
	{
		transferFunctionEval.adjointAccumulate(b, adj_outputs.adj_tf, tf_shared);
	}
}
	

template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative, bool TfDelayedAccumulation,
	bool HasVolumeDerivative>
__global__ void DvrKernelAdjointDevice(dim3 virtual_size,
		RendererInputs inputs, RendererOutputsAsInput outputsFromForward,
		AdjointColor_t adj_color, AdjointOutputs adj_outputs)
{
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		DvrKernelAdjointImpl<
			volumeFilterMode, cameraMode, tfMode, blendMode,
			HasStepsizeDerivative, HasCameraDerivative, HasTFDerivative, TfDelayedAccumulation, HasVolumeDerivative>(
				x, y, b, inputs, outputsFromForward, adj_color, adj_outputs);
	}
	KERNEL_3D_LOOP_END
}

#ifndef CUDA_NO_HOST
template<
	VolumeFilterMode volumeFilterMode,
	CameraMode cameraMode,
	TFMode tfMode,
	BlendMode blendMode,
	bool HasStepsizeDerivative,
	bool HasCameraDerivative,
	bool HasTFDerivative,
	bool HasVolumeDerivative>
__host__ void DvrKernelAdjointHost(dim3 virtual_size,
		RendererInputs inputs, RendererOutputsAsInput outputsFromForward,
		const AdjointColor_t adj_color, AdjointOutputs adj_outputs)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int b = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * b);
		DvrKernelAdjointImpl<
			volumeFilterMode, cameraMode, tfMode, blendMode,
			HasStepsizeDerivative, HasCameraDerivative, HasTFDerivative, false, HasVolumeDerivative>(
				x, y, b, inputs, outputsFromForward, adj_color, adj_outputs);
	}
}
#endif


//=======================================================
// forward variables to adjoint variables
//=======================================================
	
__host__ __device__ inline void ForwardVariablesToGradientsKernelImpl(
		int x, int y, int b,
		const ForwardDifferencesOutput& forwardVariables,
		const AdjointColor_t& gradientOutputColor,
		const ForwardDifferencesSettings& differencesSettings,
		AdjointOutputs& adj_outputs)
{
	real4 grad_color = fetchReal4(gradientOutputColor, b, y, x);

	// STEPSIZE
	if (differencesSettings.d_stepsize>=0)
	{
		adj_outputs.adj_stepSize[b][y][x] = dot(
			grad_color,
			fetchReal4(forwardVariables.gradients, b, y, x,
				differencesSettings.d_stepsize));
	}

	// CAMERA
#define combineCamera(d_param, adj_param, channel_name, channel_index)		\
	if (differencesSettings . d_param . channel_name >= 0)				\
	{																		\
		adj_outputs . adj_param [b][y][x][channel_index] = dot(				\
			grad_color,														\
			fetchReal4(forwardVariables.gradients, b, y, x,					\
				differencesSettings . d_param . channel_name));			\
	}

#define combineCamera3(d_param, adj_param)	\
	combineCamera(d_param, adj_param, x, 0)	\
	combineCamera(d_param, adj_param, y, 1)	\
	combineCamera(d_param, adj_param, z, 2)

	combineCamera3(d_rayStart, adj_cameraRayStart)
	combineCamera3(d_rayDir, adj_cameraRayDir)

#undef combineCamera3
#undef combineCamera

	// TF
	if (differencesSettings.hasTFDerivatives)
	{
		int R = differencesSettings.d_tf.size(1);
		int C = differencesSettings.d_tf.size(2);
		for (int r=0; r<R; ++r) for (int c=0; c<C; ++c)
		{
			int idx = differencesSettings.d_tf[b][r][c];
			if (idx >= 0)
			{
				::kernel::atomicAdd(
					&adj_outputs.adj_tf[b][r][c],
					dot(
						grad_color,
						fetchReal4(forwardVariables.gradients, b, y, x, idx)));
			}
		}
	}
}

#if __NVCC__==1
__global__ void ForwardVariablesToGradientsDevice(dim3 virtual_size,
	const ForwardDifferencesOutput forwardVariables,
	const AdjointColor_t gradientOutputColor,
	const ForwardDifferencesSettings differencesSettings,
	AdjointOutputs adj_outputs)
{
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		ForwardVariablesToGradientsKernelImpl(
				x, y, b, forwardVariables, gradientOutputColor, differencesSettings, adj_outputs);
	}
	KERNEL_3D_LOOP_END
}
#endif

#ifndef CUDA_NO_HOST
inline
__host__ void ForwardVariablesToGradientsHost(dim3 virtual_size,
	const ForwardDifferencesOutput forwardVariables,
	const AdjointColor_t gradientOutputColor,
	const ForwardDifferencesSettings differencesSettings,
	AdjointOutputs adj_outputs)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int b = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * b);
		ForwardVariablesToGradientsKernelImpl(
			x, y, b, forwardVariables, gradientOutputColor, differencesSettings, adj_outputs);
	}
}
#endif

	
}