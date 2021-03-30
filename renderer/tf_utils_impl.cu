#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <omp.h>

#include "renderer_settings.cuh"
#include "renderer_tf.cuh"
#include "renderer_utils.cuh"

static __host__ __device__ __forceinline__ real_t sqr(real_t x)
{
	return x * x;
}

namespace kernel
{
template<TFMode tfMode>
__host__ __device__ void PreshadeVolumeImpl(
	int x, int y, int z,
	const Tensor4Read& volume,
	const Tensor3Read& tf,
	Tensor4RW& color)
{
	real_t density = volume[0][x][y][z];
	TransferFunctionEval<tfMode> tfEval;
	real4 c = tfEval.eval(tf, 0, density);
	//printf("d=%.4f -> r=%.4f, g=%4.f, b=%.4f, a=%.7f\n",
	//	density, c.x, c.y, c.z, c.w);
	color[0][x][y][z] = c.x;
	color[1][x][y][z] = c.y;
	color[2][x][y][z] = c.z;
	color[3][x][y][z] = c.w;
}

template<TFMode tfMode>
__global__ void PreshadeVolumeDevice(dim3 virtual_size,
	const Tensor4Read volume, const Tensor3Read tf, Tensor4RW color) //no references, copied to GPU by value
{
	KERNEL_3D_LOOP(x,y,z, virtual_size)
	{
		PreshadeVolumeImpl<tfMode>(x, y, z, volume, tf, color);
	}
	KERNEL_3D_LOOP_END
}

template<TFMode tfMode>
__host__ void PreshadeVolumeHost(dim3 virtual_size,
	const Tensor4Read& volume, const Tensor3Read& tf, Tensor4RW& color)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int z = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (z * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * z);
		PreshadeVolumeImpl<tfMode>(x, y, z, volume, tf, color);
	}
}

void PreshadeVolume(
	const Tensor4Read& volume, const Tensor3Read& tf, Tensor4RW& color,
	TFMode tfMode, bool cuda)
{
	unsigned int X = volume.size(1), Y = volume.size(2), Z = volume.size(3);
	if (cuda)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		SWITCH_TF_MODE_WITH(tfMode, TFGaussianLog, [&ctx, stream, X, Y, Z, &volume, &tf, &color]()
			{
				auto cfg = ctx.createLaunchConfig3D(
					X, Y, Z, PreshadeVolumeDevice<tfMode>);
				PreshadeVolumeDevice<tfMode>
					<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
					(cfg.virtual_size, volume, tf, color);
			});
		CUMAT_CHECK_ERROR();
		
	} else
	{
		const dim3 virtual_size{ X, Y, Z };
		SWITCH_TF_MODE_WITH(tfMode, TFGaussianLog, [&virtual_size, &volume, &tf, &color]()
			{
				PreshadeVolumeHost<tfMode>
					(virtual_size, volume, tf, color);
			});
	}
}




template<TFMode tfMode>
__host__ __device__ void PreshadeVolumeAdjImpl(
	int x, int y, int z,
	const Tensor4Read& volume,
	const Tensor3Read& tf,
	const Tensor4Read& adj_color,
	BTensor3RW& dummy_adj_tf,
	Tensor4RW& adj_volume)
{
	real_t density = volume[0][x][y][z];
	TransferFunctionEval<tfMode> tfEval;
	real4 adj_c = make_real4(
		adj_color[0][x][y][z],
		adj_color[1][x][y][z],
		adj_color[2][x][y][z],
		adj_color[3][x][y][z]);
	real_t adj_density = 0;
	//BTensor3RW adj_tf; //TODO: why do I need to use an external dummy_adj_tf?
	                     //Using this local version leads to an unspecified launch failure
	tfEval.template adjoint<false, false>(
		tf, 0, density, adj_c, adj_density, dummy_adj_tf, nullptr);
	//if (fabs(adj_density)>1e-5)
	//	printf("[%03x, %03d, %03d] adj_color=(%f, %f, %f, %f) -> adj_density=%f\n",
	//		x, y, z, adj_c.x, adj_c.y, adj_c.z, adj_c.w, adj_density);
	adj_volume[0][x][y][z] = adj_density;

	//printf("[%03x, %03d, %03d] density=%.7f, adj_color=(%.7f, %.7f, %.7f, %.7f) -> adj_density=%.7f\n",
	//	x, y, z, density, adj_c.x, adj_c.y, adj_c.z, adj_c.w, adj_density);
}

template<TFMode tfMode>
__global__ void PreshadeVolumeAdjDevice(dim3 virtual_size,
	const Tensor4Read volume, const Tensor3Read tf, Tensor4Read adj_color, 
	BTensor3RW dummy_adj_tf, Tensor4RW adj_volume) //no references, copied to GPU by value
{
	KERNEL_3D_LOOP(x,y,z, virtual_size)
	{
		PreshadeVolumeAdjImpl<tfMode>(x, y, z, volume, tf, adj_color, dummy_adj_tf, adj_volume);
	}
	KERNEL_3D_LOOP_END
}

template<TFMode tfMode>
__host__ void PreshadeVolumeAdjHost(dim3 virtual_size,
	const Tensor4Read& volume, const Tensor3Read& tf, const Tensor4Read& adj_color, 
	BTensor3RW& dummy_adj_tf, Tensor4RW& adj_volume)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(guided)
	for (int __i = 0; __i < count; ++__i)
	{
		int z = __i / (virtual_size.x * virtual_size.y);
		int y = (__i - (z * virtual_size.x * virtual_size.y)) / virtual_size.x;
		int x = __i - virtual_size.x * (y + virtual_size.y * z);
		PreshadeVolumeAdjImpl<tfMode>(x, y, z, volume, tf, adj_color, dummy_adj_tf, adj_volume);
	}
}

void PreshadeVolumeAdj(
	const Tensor4Read& volume, const Tensor3Read& tf, const Tensor4Read& adj_color,
	Tensor4RW& adj_volume, TFMode tfMode, bool cuda)
{
	unsigned int X = volume.size(1), Y = volume.size(2), Z = volume.size(3);
	if (cuda)
	{
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		SWITCH_TF_MODE_WITH(tfMode, TFGaussianLog, [&ctx, stream, X, Y, Z, &volume, &tf, &adj_color, &adj_volume]()
			{
				auto cfg = ctx.createLaunchConfig3D(
					X, Y, Z, PreshadeVolumeAdjDevice<tfMode>);
				BTensor3RW dummy_adj_tf;
				PreshadeVolumeAdjDevice<tfMode>
					<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
					(cfg.virtual_size, volume, tf, adj_color, dummy_adj_tf, adj_volume);
			});
		CUMAT_CHECK_ERROR();
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	}
	else
	{
		const dim3 virtual_size{ X, Y, Z };
		SWITCH_TF_MODE_WITH(tfMode, TFGaussianLog, [&virtual_size, &volume, &tf, &adj_color, &adj_volume]()
			{
				BTensor3RW dummy_adj_tf;
				PreshadeVolumeAdjHost<tfMode>
					(virtual_size, volume, tf, adj_color, dummy_adj_tf, adj_volume);
			});
	}
}



namespace
{
	typedef unsigned long long state_t;

	struct RandNextHelper {
		static __host__ __device__ __forceinline__ int eval(int bits, state_t* seed)
		{
			*seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1LL << 48) - 1);
			return (int)(*seed >> (48 - bits));
		}
	};

	template<typename S>
	struct RandNext {
		static __host__ __device__ __forceinline__ S eval(state_t* seed, S min, S max);
	};

	template<>
	struct RandNext<float> {
		static __host__ __device__ __forceinline__ float eval(state_t* seed, float min, float max)
		{
			float v = RandNextHelper::eval(24, seed) / ((float)(1 << 24));
			return (v * (max - min)) + min;
		}
	};

	template<>
	struct RandNext<double> {
		static __host__ __device__ __forceinline__ double eval(state_t* seed, double min, double max)
		{
			double v = (((long)(RandNextHelper::eval(26, seed)) << 27) + RandNextHelper::eval(27, seed))
				/ (double)(1LL << 53);
			return (v * (max - min)) + min;
		}
	};
}

template<TFMode tfMode>
__host__ __device__ void FindBestFitImpl(
	int x, int y, int z, state_t* seed,
	const Tensor4Read& colorVolume,
	const Tensor3Read& tf,
	Tensor4RW& density,
	int numSamples, real_t opacityWeighting)
{
	real4 targetColor = make_real4(
		colorVolume[0][x][y][z],
		colorVolume[1][x][y][z],
		colorVolume[2][x][y][z],
		colorVolume[3][x][y][z]);

	real_t minCost = FLT_MAX;
	real_t bestDensity = 0;
	TransferFunctionEval<tfMode> tfEval;
	RandNext<real_t> rnd;
	
	for (int i = 0; i < numSamples; ++i)
	{
		real_t density = rnd.eval(seed, real_t(0), real_t(1));
		real4 currentColor = tfEval.eval(tf, 0, density);
		//compute difference
		real_t colorDifference = lengthSquared(
			make_real3(targetColor) - make_real3(currentColor));
		real_t opacityDifference = logf(1+abs(targetColor.w - currentColor.w));

		////Test
		//if (x==40 && y==39 && z==120)
		//{
		//	printf("[%d,%d,%d] target=(%.2f, %.2f, %.2f, %.2f), density=%.3f, current=(%.2f, %.2f, %.2f, %.2f), cd=%.3f, od=%.3f\n",
		//		x, y, z, targetColor.x, targetColor.y, targetColor.z, targetColor.w,
		//		density, currentColor.x, currentColor.y, currentColor.z, currentColor.w,
		//		colorDifference, opacityDifference);
		//}
		
		real_t cost = colorDifference + opacityWeighting * opacityDifference;
		if (cost < minCost)
		{
			minCost = cost;
			bestDensity = density;
		}
	}

	////Test
	//if (targetColor.z>0.8 && targetColor.x<0.2 && targetColor.y<0.2)
	//{
	//	printf("Blue found at (%d,%d,%d), color=(%.2f, %.2f, %.2f, %.2f), density=%.3f\n",
	//		x, y, z, targetColor.x, targetColor.y, targetColor.z, targetColor.w, bestDensity);
	//}
	
	density[0][x][y][z] = bestDensity;
}

template<TFMode tfMode>
__global__ void FindBestFitDevice(dim3 virtual_size,
	const LTensor1Read seeds,
	const Tensor4Read colorVolume, const Tensor3Read tf, Tensor4RW density,
	int numSamples, real_t opacityWeighting)
{
	state_t seed = seeds[threadIdx.x];
	KERNEL_3D_LOOP(x, y, z, virtual_size)
	{
		FindBestFitImpl<tfMode>(x, y, z, &seed, 
			colorVolume, tf, density, numSamples, opacityWeighting);
	}
	KERNEL_3D_LOOP_END
}

template<TFMode tfMode>
__host__ void FindBestFitHost(dim3 virtual_size,
	const LTensor1Read& seeds,
	const Tensor4Read& colorVolume, const Tensor3Read& tf, Tensor4RW& density,
	int numSamples, real_t opacityWeighting)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel
	{
		state_t seed = seeds[omp_get_thread_num()];
#pragma omp for
		for (int __i = 0; __i < count; ++__i)
		{
			int z = __i / (virtual_size.x * virtual_size.y);
			int y = (__i - (z * virtual_size.x * virtual_size.y)) / virtual_size.x;
			int x = __i - virtual_size.x * (y + virtual_size.y * z);
			FindBestFitImpl<tfMode>(x, y, z, &seed,
				colorVolume, tf, density, numSamples, opacityWeighting);
		}
	}
}



void FindBestFit(
	const Tensor4Read& colorVolume, const Tensor3Read& tf,
	Tensor4RW& densityOut, const LTensor1Read& seeds,
	kernel::TFMode tfMode, int numSamples, real_t opacityWeighting,
	bool cuda)
{
	unsigned int X = colorVolume.size(1), Y = colorVolume.size(2), Z = colorVolume.size(3);
	if (cuda)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		SWITCH_TF_MODE(tfMode, [&ctx, stream, X, Y, Z, &seeds, &colorVolume, &tf, &densityOut, &numSamples, &opacityWeighting]()
			{
				auto cfg = ctx.createLaunchConfig3D(
					X, Y, Z, FindBestFitDevice<tfMode>);
				FindBestFitDevice<tfMode>
					<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
					(cfg.virtual_size, seeds, colorVolume, tf, densityOut, numSamples, opacityWeighting);
			});
		CUMAT_CHECK_ERROR();

	}
	else
	{
		const dim3 virtual_size{ X, Y, Z };
		SWITCH_TF_MODE(tfMode, [&virtual_size, &seeds, &colorVolume, &tf, &densityOut, &numSamples, &opacityWeighting]()
			{
				FindBestFitHost<tfMode>
					(virtual_size, seeds, colorVolume, tf, densityOut, numSamples, opacityWeighting);
			});
	}
}


	

template<TFMode tfMode>
__host__ __device__ void FindBestFitWithComparisonImpl(
	int x, int y, int z, state_t* seed,
	const Tensor4Read& colorVolume,
	const Tensor3Read& tf,
	Tensor4RW& density, const Tensor4RW& previousDensity,
	int numSamples, real_t opacityWeighting, real_t neighborWeighting)
{
	real4 targetColor = make_real4(
		colorVolume[0][x][y][z],
		colorVolume[1][x][y][z],
		colorVolume[2][x][y][z],
		colorVolume[3][x][y][z]);

	real_t minCost = FLT_MAX;
	real_t bestDensity = 0;
	TransferFunctionEval<tfMode> tfEval;
	RandNext<real_t> rnd;

	bool hasNeighbor[6];
	real_t neighbor[6];
	if (x > 0)
	{
		hasNeighbor[0] = true;
		neighbor[0] = previousDensity[0][x - 1][y][z];
	}
	else
		hasNeighbor[0] = false;
	if (x < previousDensity.size(1)-1)
	{
		hasNeighbor[1] = true;
		neighbor[1] = previousDensity[0][x + 1][y][z];
	}
	else
		hasNeighbor[1] = false;
	if (y > 0)
	{
		hasNeighbor[2] = true;
		neighbor[2] = previousDensity[0][x][y - 1][z];
	}
	else
		hasNeighbor[2] = false;
	if (y < previousDensity.size(2) - 1)
	{
		hasNeighbor[3] = true;
		neighbor[3] = previousDensity[0][x][y + 1][z];
	}
	else
		hasNeighbor[3] = false;
	if (z > 0)
	{
		hasNeighbor[4] = true;
		neighbor[4] = previousDensity[0][x][y][z - 1];
	}
	else
		hasNeighbor[4] = false;
	if (z < previousDensity.size(3) - 1)
	{
		hasNeighbor[5] = true;
		neighbor[5] = previousDensity[0][x][y][z + 1];
	}
	else
		hasNeighbor[5] = false;
	
	for (int i = 0; i < numSamples; ++i)
	{
		real_t density = rnd.eval(seed, real_t(0), real_t(1));
		real4 currentColor = tfEval.eval(tf, 0, density);
		//compute difference
		real_t colorDifference = lengthSquared(
			make_real3(targetColor) - make_real3(currentColor));
		real_t opacityDifference = logf(1 + abs(targetColor.w - currentColor.w));
		real_t cost = colorDifference + opacityWeighting * opacityDifference;
		//add neighbor cost
#pragma unroll
		for (int j=0; j<6; ++j)
		{
			if (hasNeighbor[j])
				cost += neighborWeighting * sqr(density - neighbor[j]);
		}
		//check if best value
		if (cost < minCost)
		{
			minCost = cost;
			bestDensity = density;
		}
	}
	density[0][x][y][z] = bestDensity;
}

template<TFMode tfMode>
__global__ void FindBestFitWithComparisonDevice(dim3 virtual_size,
	const LTensor1Read seeds,
	const Tensor4Read colorVolume, const Tensor3Read tf, Tensor4RW density,
	const Tensor4RW previousDensity,
	int numSamples, real_t opacityWeighting, real_t neighborWeighting)
{
	state_t seed = seeds[threadIdx.x];
	KERNEL_3D_LOOP(x, y, z, virtual_size)
	{
		FindBestFitWithComparisonImpl<tfMode>(x, y, z, &seed,
			colorVolume, tf, density, previousDensity, numSamples, opacityWeighting, neighborWeighting);
	}
	KERNEL_3D_LOOP_END
}

template<TFMode tfMode>
__host__ void FindBestFitWithComparisonHost(dim3 virtual_size,
	const LTensor1Read& seeds,
	const Tensor4Read& colorVolume, const Tensor3Read& tf, Tensor4RW& density,
	const Tensor4RW& previousDensity,
	int numSamples, real_t opacityWeighting, real_t neighborWeighting)
{
	int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel
	{
		state_t seed = seeds[omp_get_thread_num()];
#pragma omp for
		for (int __i = 0; __i < count; ++__i)
		{
			int z = __i / (virtual_size.x * virtual_size.y);
			int y = (__i - (z * virtual_size.x * virtual_size.y)) / virtual_size.x;
			int x = __i - virtual_size.x * (y + virtual_size.y * z);
			FindBestFitWithComparisonImpl<tfMode>(x, y, z, &seed,
				colorVolume, tf, density, previousDensity, numSamples, opacityWeighting, neighborWeighting);
		}
	}
}



void FindBestFitWithComparison(
	const Tensor4Read& colorVolume, const Tensor3Read& tf,
	const Tensor4RW& previousDensity,
	Tensor4RW& densityOut, const LTensor1Read& seeds,
	kernel::TFMode tfMode, int numSamples, real_t opacityWeighting,
	real_t neighborWeighting, bool cuda)
{
	unsigned int X = colorVolume.size(1), Y = colorVolume.size(2), Z = colorVolume.size(3);
	if (cuda)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		SWITCH_TF_MODE(tfMode, [&ctx, stream, X, Y, Z, &seeds, &colorVolume, &tf, &densityOut, &previousDensity, &numSamples, &opacityWeighting, &neighborWeighting]()
			{
				auto cfg = ctx.createLaunchConfig3D(
					X, Y, Z, FindBestFitWithComparisonDevice<tfMode>);
				FindBestFitWithComparisonDevice<tfMode>
					<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
					(cfg.virtual_size, seeds, colorVolume, tf, densityOut, previousDensity, numSamples, opacityWeighting, neighborWeighting);
			});
		CUMAT_CHECK_ERROR();

	}
	else
	{
		const dim3 virtual_size{ X, Y, Z };
		SWITCH_TF_MODE(tfMode, [&virtual_size, &seeds, &colorVolume, &tf, &densityOut, &previousDensity, &numSamples, &opacityWeighting, &neighborWeighting]()
			{
				FindBestFitWithComparisonHost<tfMode>
					(virtual_size, seeds, colorVolume, tf, densityOut, previousDensity, numSamples, opacityWeighting, neighborWeighting);
			});
	}
}
	
	
}
