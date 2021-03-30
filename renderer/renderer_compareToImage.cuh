#pragma once

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "forward_vector.h"

namespace kernel
{
	template<int D>
	struct CompareToImage
	{
		static __host__ __device__ __inline__
		cudAD::fvar<real4, D> readInput(
			const kernel::Tensor4Read& colorInput,
			const kernel::Tensor5Read& gradientsInput,
			int b, int y, int x)
		{
			real4 value = fetchReal4(colorInput, b, y, x);
			cudAD::internal::array<real4, D> derivatives;
			for (int d = 0; d < D; ++d)
				derivatives[d] = fetchReal4(gradientsInput, b, y, x, d);
			return cudAD::fvar<real4, D>(value, derivatives);
		}

		static __host__ __device__ __inline__
		real4 readReference(
			const kernel::Tensor4Read& colorReference,
			int b, int y, int x)
		{
			return fetchReal4(colorReference, b, y, x);
		}

		static __host__ __device__ __inline__
		void writeOutput(
			const cudAD::fvar<real_t, D>& output,
			kernel::Tensor4RW& differenceOut,
			kernel::Tensor4RW& gradientsOut,
			int b, int y, int x)
		{
			//non-reduced
			differenceOut[b][y][x][0] = output.value();
			for (int d = 0; d < D; ++d)
				gradientsOut[b][y][x][d] = output.derivative(d);
		}
		
		static __host__ __device__ __inline__
		void writeOutput(
			const cudAD::fvar<real_t, D>& output,
			kernel::Tensor1RW& differenceOut,
			kernel::Tensor1RW& gradientsOut)
		{
			//reduced
			differenceOut[0] = output.value();
			for (int d = 0; d < D; ++d)
				gradientsOut[d] = output.derivative(d);
		}

		//computes the L2-norm between the input and the reference
		static __host__ __device__ __inline__
		cudAD::fvar<real_t, D> compare(
			const cudAD::fvar<real4, D>& input, const real4& reference)
		{
			return cudAD::length(input - reference);
		}
	};

}

