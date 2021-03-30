#ifdef _WIN32
//HACK: at::native::gpu_kernel_multiple_outputs
//uses std::apply internally, but somehow, NVCC does not recognize
//this properly as device function, even though it is constexpr and
//--expt-relaxed-constexpr is defined.
//This hack "unlearns" std::apply so that ATen uses a custom replacement.
#include <yvals_core.h> //internal MSVC header that defines which standard functions are available
#undef __cpp_lib_apply
#endif

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "renderer_tf.cuh"
#include "renderer_interpolation.cuh"

#include <math.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cpu/Loops.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <curand_kernel.h>
#include <third-party/cub/cub.cuh>
#include <cuMat/src/Context.h>
#include <cuMat/src/DevicePointer.h>

#ifndef FLT_EPSILON
#define FLT_EPSILON      1.192092896e-07F
#endif

namespace kernel
{
	template<typename scalar_t>
	__host__ __device__ __forceinline__ scalar_t my_log2(scalar_t s);
	template<>
	__host__ __device__ __forceinline__ float my_log2<float>(float s) { return ::log2f(s); }
	template<>
	__host__ __device__ __forceinline__ double my_log2<double>(double s) { return ::log2(s); }

	template<typename scalar_t>
	struct MullogForwardFunctor {
		__host__ __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
			return (a < FLT_EPSILON) ? scalar_t(0) : (a * my_log2(a));
		}
	};

	template<typename scalar_t>
	struct MullogBackwardFunctor {
		__host__ __device__ __forceinline__ scalar_t operator() (
				const scalar_t t, const scalar_t grad_output) const {
			//1/log(2)
			static constexpr scalar_t oneOverLog2 = 1.4426950408889634073599246810018921374266459541529859341354494069;
			return (t < FLT_EPSILON)
				? scalar_t(0)
				: (grad_output * (my_log2(t)+ oneOverLog2));
		}
	};
	
	void mulLogForward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mullog_cuda", [&]()
			{
				at::native::gpu_kernel(iter, MullogForwardFunctor<scalar_t>());
			});
		} else
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mullog_cpu", [&]()
			{
				at::native::cpu_kernel(iter, MullogForwardFunctor<scalar_t>());
			});
		}
	}

	void mulLogBackward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mullog_cuda_backward", [&]()
			{
				at::native::gpu_kernel(iter, MullogBackwardFunctor<scalar_t>());
			});
		}
		else
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "mullog_cpu_backward", [&]()
			{
				at::native::cpu_kernel(iter, MullogBackwardFunctor<scalar_t>());
			});
		}
	}



	template<typename scalar_t>
	struct LogMSEForwardFunctor {
		__host__ __device__ __forceinline__ scalar_t operator() (
			const scalar_t logX, const scalar_t logY) const
		{
			return kernel::logMSE(logX, logY);
		}
	};

	template<typename scalar_t>
	struct LogMSEBackwardFunctor {
		__host__ __device__ __forceinline__ thrust::tuple<scalar_t, scalar_t> operator() (
			const scalar_t logX, const scalar_t logY, const scalar_t grad_output) const
		{
			scalar_t adjLogX, adjLogY;
			kernel::adjLogMSE(logX, logY, grad_output, adjLogX, adjLogY);
			return { adjLogX, adjLogY };
		}
	};

	void logMSEForward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logmse_cuda", [&]()
				{
					at::native::gpu_kernel(iter, LogMSEForwardFunctor<scalar_t>());
				});
		}
		else
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logmse_cpu", [&]()
				{
					at::native::cpu_kernel(iter, LogMSEForwardFunctor<scalar_t>());
				});
		}
	}

	void logMSEBackward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			throw std::runtime_error(R"(
CUDA version of logMSEBackward not available.
Would cause the following error (bug in PyTorch?):

error: identifier "std::apply< ::kernel::logMSEBackward( ::at::TensorIterator &, bool)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2)]::operator ()() const   ::[lambda(double, double, double) (instance 1)] &,     ::std::tuple<double, double, double >  &> " is undefined in device code
				)");
			//AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logmse_cuda_backward", [&]()
			//	{
			//		at::native::gpu_kernel_multiple_outputs(
			//			iter, [=] GPU_LAMBDA(scalar_t logX, scalar_t logY, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
			//			return LogMSEBackwardFunctor<scalar_t>()(logX, logY, grad);
			//		});
			//	});
		}
		else
		{
			throw std::runtime_error("CPU version of logMSEBackward not implemented, at::native::cpu_kernel_multiple_outputs is not available");
			//AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logmse_cpu_backward", [&]()
			//	{
			//		at::native::cpu_kernel_multiple_outputs(iter, LogMSEBackwardFunctor<scalar_t>());
			//	});
		}
	}


	template<typename scalar_t>
	struct LogL1ForwardFunctor {
		__host__ __device__ __forceinline__ scalar_t operator() (
			const scalar_t logX, const scalar_t logY) const
		{
			return kernel::logL1(logX, logY);
		}
	};

	template<typename scalar_t>
	struct LogL1BackwardFunctor {
		__host__ __device__ __forceinline__ thrust::tuple<scalar_t, scalar_t> operator() (
			const scalar_t logX, const scalar_t logY, const scalar_t grad_output) const
		{
			scalar_t adjLogX, adjLogY;
			kernel::adjLogL1(logX, logY, grad_output, adjLogX, adjLogY);
			return { adjLogX, adjLogY };
		}
	};

	void logL1Forward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logl1_cuda", [&]()
				{
					at::native::gpu_kernel(iter, LogL1ForwardFunctor<scalar_t>());
				});
		}
		else
		{
			AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logl1_cpu", [&]()
				{
					at::native::cpu_kernel(iter, LogL1ForwardFunctor<scalar_t>());
				});
		}
	}

	void logL1Backward(
		at::TensorIterator& iter,
		bool cuda)
	{
		if (cuda)
		{
			throw std::runtime_error(R"(
CUDA version of logL1Backward not available.
Would cause the following error (bug in PyTorch?):

error: identifier "std::apply< ::kernel::logMSEBackward( ::at::TensorIterator &, bool)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2)]::operator ()() const   ::[lambda(double, double, double) (instance 1)] &,     ::std::tuple<double, double, double >  &> " is undefined in device code
				)");
			//AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logl1_cuda_backward", [&]()
			//	{
			//		at::native::gpu_kernel_multiple_outputs(
			//			iter, [=] GPU_LAMBDA(scalar_t logX, scalar_t logY, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
			//			return LogL1BackwardFunctor<scalar_t>()(logX, logY, grad);
			//		});
			//	});
		}
		else
		{
			throw std::runtime_error("CPU version of logL1Backward not implemented, at::native::cpu_kernel_multiple_outputs is not available");
			//AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logl1_cpu_backward", [&]()
			//	{
			//		at::native::cpu_kernel_multiple_outputs(iter, LogL1BackwardFunctor<scalar_t>());
			//	});
		}
	}


	typedef PackedTensorAccessor32<bool, 1, DefaultPtrTraits> BTensor1RW;

	//before we can sample, we first need to query the maximal values for the opacity+gradient

	struct MaxReal3
	{
		__device__ __forceinline__ real3 operator()(const real3& a, const real3& b) const
		{
			return rmax(a, b);
		}
	};
	template<TFMode tfMode>
	struct GradientComputationOp
	{
		const Tensor4Read densityVolume;
		const Tensor3Read tf;
		__device__ __forceinline__ real3 operator()(const int& idx) const
		{
			const int X = densityVolume.size(1);
			const int Y = densityVolume.size(2);
			const int Z = densityVolume.size(3);
			const int z = idx / (X * Y);
			const int y = (idx - (z * X * Y)) / X;
			const int x = idx - X * (y + Y * z);

			real_t dCenter = densityVolume[0][x][y][z];
			real_t dXm = densityVolume[0][max(0, x - 1)][y][z];
			real_t dXp = densityVolume[0][min(X - 1, x + 1)][y][z];
			real_t dYm = densityVolume[0][x][max(0, y - 1)][z];
			real_t dYp = densityVolume[0][x][min(Y - 1, y + 1)][z];
			real_t dZm = densityVolume[0][x][y][max(0, z - 1)];
			real_t dZp = densityVolume[0][x][y][min(Z - 1, z + 1)];

			TransferFunctionEval<tfMode> tfEval;
			real_t oCenter = tfEval.eval(tf, 0, dCenter).w;
			real_t oXm = tfEval.eval(tf, 0, dXm).w;
			real_t oXp = tfEval.eval(tf, 0, dXp).w;
			real_t oYm = tfEval.eval(tf, 0, dYm).w;
			real_t oYp = tfEval.eval(tf, 0, dYp).w;
			real_t oZm = tfEval.eval(tf, 0, dZm).w;
			real_t oZp = tfEval.eval(tf, 0, dZp).w;

			real3 densityGradient =
				make_real3(dXp - dXm, dYp - dYm, dZp - dZm) / 2.0f;
			real3 opacityGradient =
				make_real3(oXp - oXm, oYp - oYm, oZp - oZm) / 2.0f;
			return make_real3(
				length(densityGradient),
				oCenter,
				length(opacityGradient));
		}
	};
	
	template<TFMode tfMode>
	__global__ void SampleImportanceKernel(dim3 virtual_size,
		BTensor1RW output,
		const Tensor2Read sampleLocations,
		const Tensor4Read densityVolume,
		const Tensor3Read tf,
		float weightUniform, float weightDensityGradient,
		float weightOpacity, float weightOpacityGradient,
		real3 maxValues, int seed)
	{
		const int3 volumeSize = make_int3(densityVolume.size(1), densityVolume.size(2), densityVolume.size(3));
		const real3 volumeSizeF = make_real3(volumeSize - make_int3(1));
		TransferFunctionEval<tfMode> tfEval;
		curandState_t state;
		curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
		CUMAT_KERNEL_1D_LOOP(i, virtual_size)
		{
			//get sample position
			const real3 posWorld = fetchReal3(sampleLocations, i);
			const real3 posVol = posWorld * volumeSizeF;

			//random numbers
			float rndSample = curand_uniform(&state);
			float rndReject = curand_uniform(&state);
			//printf("random [%05d]: %.5f, %.5f\n", int(i), rndSample, rndReject);
			
			//fetch densities
			real_t dCenter = fetchTrilinear(densityVolume, volumeSize, 0, posVol);
			real_t dXm = fetchTrilinear(densityVolume, volumeSize, 0, 
				posVol - make_real3(1, 0, 0));
			real_t dXp = fetchTrilinear(densityVolume, volumeSize, 0,
				posVol + make_real3(1, 0, 0));
			real_t dYm = fetchTrilinear(densityVolume, volumeSize, 0,
				posVol - make_real3(0, 1, 0));
			real_t dYp = fetchTrilinear(densityVolume, volumeSize, 0,
				posVol + make_real3(0, 1, 0));
			real_t dZm = fetchTrilinear(densityVolume, volumeSize, 0,
				posVol - make_real3(0, 0, 1));
			real_t dZp = fetchTrilinear(densityVolume, volumeSize, 0,
				posVol + make_real3(0, 0, 1));

			//apply TF
			real_t oCenter = tfEval.eval(tf, 0, dCenter).w;
			real_t oXm = tfEval.eval(tf, 0, dXm).w;
			real_t oXp = tfEval.eval(tf, 0, dXp).w;
			real_t oYm = tfEval.eval(tf, 0, dYm).w;
			real_t oYp = tfEval.eval(tf, 0, dYp).w;
			real_t oZm = tfEval.eval(tf, 0, dZm).w;
			real_t oZp = tfEval.eval(tf, 0, dZp).w;

			//perform sampling
			bool sampled;
			if (rndSample <= weightUniform)
				sampled = true;
			else if (rndSample <= weightDensityGradient)
			{
				real3 densityGradient =
					make_real3(dXp - dXm, dYp - dYm, dZp - dZm) / 2.0f;
				real_t norm = length(densityGradient) / maxValues.x;
				sampled = norm >= rndReject;
			}
			else if (rndSample <= weightOpacity)
			{
				sampled = (oCenter / maxValues.y) >= rndReject;
			}
			else //(rndSample <= weightOpacityGradient==1)
			{
				real3 opacityGradient =
					make_real3(oXp - oXm, oYp - oYm, oZp - oZm) / 2.0f;
				real_t norm = length(opacityGradient) / maxValues.z;
				sampled = norm >= rndReject;
			}
			//printf("sampled [%05d]: %d\n", int(i), sampled ? 1 : 0);
			output[i] = sampled;
		}
		CUMAT_KERNEL_1D_LOOP_END
	}
	
	void sampleImportanceCUDA(
		BTensor1RW& output,
		const Tensor2Read& sampleLocations,
		const Tensor4Read& densityVolume,
		const Tensor3Read& tf, TFMode tfMode,
		float weightUniform, float weightDensityGradient,
		float weightOpacity, float weightOpacityGradient,
		int seed)
	{
		unsigned int X = densityVolume.size(1), Y = densityVolume.size(2), Z = densityVolume.size(3);
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		SWITCH_TF_MODE(tfMode, [&ctx, stream, X, Y, Z, &output, &sampleLocations, &densityVolume, &tf, weightUniform, weightDensityGradient, weightOpacity, weightOpacityGradient, seed]()
			{
				int XYZ = X * Y * Z;
				cub::CountingInputIterator<int> it1(0);
				GradientComputationOp<tfMode> op{ densityVolume, tf };
				cub::TransformInputIterator<real3, decltype(op), decltype(it1)> it2(it1, op);
				real3* maxValuesDevice = reinterpret_cast<real3*>(ctx.mallocDevice(sizeof(real3)));

				//reduce max values
				void* d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(
					d_temp_storage, temp_storage_bytes,
					it2, maxValuesDevice, XYZ, MaxReal3(), real3{ 0,0,0 }, stream));
				d_temp_storage = ctx.mallocDevice(temp_storage_bytes);
				CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(
					d_temp_storage, temp_storage_bytes,
					it2, maxValuesDevice, XYZ, MaxReal3(), real3{ 0,0,0 }, stream));
				ctx.freeDevice(d_temp_storage);

				//copy to host
				real3 maxValuesHost;
				CUMAT_SAFE_CALL(cudaMemcpy(
					&maxValuesHost, maxValuesDevice, sizeof(real3), cudaMemcpyDeviceToHost));
				ctx.freeDevice(maxValuesDevice);
				//std::cout << "max density gradient norm: " << maxValuesHost.x <<
				//	"\nmax opacity: " << maxValuesHost.y <<
				//	"\nmax opacity gradient norm: " << maxValuesHost.z << std::endl;

				//main sampling kernel
				int N = sampleLocations.size(0);
				auto cfg = ctx.createLaunchConfig1D(N, SampleImportanceKernel<tfMode>);
				//std::cout << "now sample " << N << " points (thread per block: " << cfg.thread_per_block.x << ")" << std::endl;
				SampleImportanceKernel<tfMode>
					<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
					(cfg.virtual_size, output, sampleLocations, densityVolume, tf,
						weightUniform, weightDensityGradient, weightOpacity, weightOpacityGradient,
						maxValuesHost, seed);
				CUMAT_CHECK_ERROR();

				//CUMAT_SAFE_CALL(cudaDeviceSynchronize()); //test
			});
	}
	
}
