#pragma once

/*
 * Utilities for adjoint code
 */

#include "renderer_settings.cuh"

#ifndef __NVCC__
#ifndef __noinline
#define __noinline__
#endif
#endif

namespace kernel
{
	template<typename T>
	__host__ __device__ __noinline__ void atomicAdd(T* address, T val)
	{
#ifdef __CUDA_ARCH__
		::atomicAdd(address, val);
#else
		#pragma omp atomic
		(*address) += val;
#endif
	}
	
	__host__ __device__ __forceinline__ real3 adjNormalize(
		const real3& input, const real3& adj_output)
	{
		//real_t length = sqrt(input.x * input.x + input.y * input.y + input.z * input.z)
		//real3 output = input / length

		real3 adj_input = make_real3(0);
		real_t len = length(input);

		//real3 output = input / length
		adj_input += adj_output / len;
		real_t adj_length = dot(adj_output, -input / (len * len));

		//real_t length = sqrt(input.x * input.x + input.y * input.y + input.z * input.z)
		adj_input += adj_length * input / len;
		
		return adj_input;
	}

	__host__ __device__ __forceinline__ void adjCross(
		const real3& a, const real3& b, const real3& adj_c, real3& adj_a, real3& adj_b)
	{
		//c.x = a.y*b.z - a.z*b.y
		adj_a.y += b.z * adj_c.x;
		adj_b.z += a.y * adj_c.x;
		adj_a.z -= b.y * adj_c.x;
		adj_b.y -= a.z * adj_c.x;
		//c.y = a.z*b.x - a.x*b.z
		adj_a.z += b.x * adj_c.y;
		adj_b.x += a.z * adj_c.y;
		adj_a.x -= b.z * adj_c.y;
		adj_b.z -= a.x * adj_c.y;
		//c.z = a.x*b.y - a.y*b.x
		adj_a.x += b.y * adj_c.z;
		adj_b.y += a.x * adj_c.z;
		adj_a.y -= b.x * adj_c.z;
		adj_b.x -= a.y * adj_c.z;
	}

}
