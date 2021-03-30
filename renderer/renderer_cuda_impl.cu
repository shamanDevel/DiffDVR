#include "renderer_cuda.h"

#include "renderer_compareToImage.cuh"

#include <cuMat/src/Context.h>
#include <cuMat/../third-party/cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

template<int D>
__global__ void CompareToImage_NoReduce_CUDA_Kernel(
	dim3 virtual_size, kernel::Tensor4Read colorInput,
	kernel::Tensor5Read gradientsInput,
	kernel::Tensor4Read colorReference,
	kernel::Tensor4RW differenceOut,
	kernel::Tensor4RW gradientsOut)
{
	using namespace kernel;
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		cudAD::fvar<real4, D> input = CompareToImage<D>::readInput(
			colorInput, gradientsInput, b, y, x);
		real4 reference = CompareToImage<D>::readReference(
			colorReference, b, y, x);
		cudAD::fvar<real_t, D> result = CompareToImage<D>::compare(input, reference);
		CompareToImage<D>::writeOutput(result, differenceOut, gradientsOut, b, y, x);
	}
	KERNEL_3D_LOOP_END
}

template<int D>
struct CompareToImage_CUDA_InputOp
{
	const int3 strides;
	const int3 sizes;
	const real_t normalization;
	const kernel::Tensor4Read colorInput;
	const kernel::Tensor5Read gradientsInput;
	const kernel::Tensor4Read colorReference;

	__host__ __device__ CompareToImage_CUDA_InputOp(int B, int H, int W, const kernel::Tensor4Read& colorInput,
		const kernel::Tensor5Read& gradientsInput, const kernel::Tensor4Read& colorReference)
		: strides({H*W, W, 1}),
		  sizes({B, H, W}),
		  normalization(real_t(1) / real_t(B*H*W)),
		  colorInput(colorInput),
		  gradientsInput(gradientsInput),
		  colorReference(colorReference)
	{}

	__host__ __device__ __forceinline__
	cudAD::fvar<real_t, D> operator()(const int& idx) const
	{
		int b = (idx / strides.x) % sizes.x;
		int y = (idx / strides.y) % sizes.y;
		int x = (idx / strides.z) % sizes.z;
		using namespace kernel;
		cudAD::fvar<real4, D> input = CompareToImage<D>::readInput(
			colorInput, gradientsInput, b, y, x);
		real4 reference = CompareToImage<D>::readReference(
			colorReference, b, y, x);
		cudAD::fvar<real_t, D> result = CompareToImage<D>::compare(input, reference);
		return result * normalization;
	}
};
template<int D>
using CompareToImage_CUDA_InputIterator = cub::TransformInputIterator<
	cudAD::fvar<real_t, D>,
	CompareToImage_CUDA_InputOp<D>,
	cub::CountingInputIterator<int>>;

template<int D>
struct CompareToImage_CUDA_OutputIterator
{
	// Required iterator traits
	typedef CompareToImage_CUDA_OutputIterator<D> Base;
	typedef CompareToImage_CUDA_OutputIterator<D> self_type; ///< My own type
	typedef int difference_type; ///< Type to express the result of subtracting one iterator from another
	using ValueType = cudAD::fvar<real_t, D>;
	using value_type = ValueType; ///< The type of the element the iterator can point to
	using pointer = ValueType*; ///< The type of a pointer to an element the iterator can point to
	using reference = self_type&; ///< The type of a reference to an element the iterator can point to
	using iterator_category = std::random_access_iterator_tag;

	kernel::Tensor1RW differenceOut;
	kernel::Tensor1RW gradientsOut;

	__host__ __device__ CompareToImage_CUDA_OutputIterator(
		const kernel::Tensor1RW& differenceOut, const kernel::Tensor1RW& gradientsOut)
		: differenceOut(differenceOut),
		  gradientsOut(gradientsOut)
	{}

	template <typename Distance>
	__host__ __device__ __forceinline__ reference operator[](Distance n)
	{
		assert(n == 0);
		return *this;
	}

	__host__ __device__ __forceinline__ reference operator*()
	{
		return *this;
	}

	__host__ __device__ __forceinline__ self_type& operator=(const ValueType& value)
	{
		kernel::CompareToImage<D>::writeOutput(value, differenceOut, gradientsOut);
		return *this;
	}
};

#define SWITCH_N(N, VariableName, ...)	\
	switch (N) { \
		case 1: {static constexpr int VariableName = 1; __VA_ARGS__(); } break;	\
		default: {std::cerr << "Only 1 variable is supported at the moment, not " << N << std::endl;} break;	\
	}

void renderer::RendererCuda::compareToImage_NoReduce(const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput, const kernel::Tensor4Read& colorReference,
	kernel::Tensor4RW& differenceOut, kernel::Tensor4RW& gradientsOut, int B, int W, int H, int D)
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	SWITCH_N(D, NumDerivatives, 
		[colorInput, gradientsInput, colorReference, differenceOut, gradientsOut, B, W, H]()
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
		auto cfg = ctx.createLaunchConfig3D(W, H, B, CompareToImage_NoReduce_CUDA_Kernel<NumDerivatives>);
		CompareToImage_NoReduce_CUDA_Kernel<NumDerivatives>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, colorInput, gradientsInput, colorReference, differenceOut, gradientsOut);
		CUMAT_CHECK_ERROR();
	});
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

void renderer::RendererCuda::compareToImage_WithReduce(const kernel::Tensor4Read& colorInput,
	const kernel::Tensor5Read& gradientsInput, const kernel::Tensor4Read& colorReference,
	kernel::Tensor1RW& differenceOut, kernel::Tensor1RW& gradientsOut, int B, int W, int H, int D)
{
	SWITCH_N(D, NumDerivatives,
		[colorInput, gradientsInput, colorReference, differenceOut, gradientsOut, B, W, H]()
	{
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();
			
		int numItems = B * W * H;
		CompareToImage_CUDA_InputIterator<NumDerivatives> input =
			CompareToImage_CUDA_InputIterator<NumDerivatives>(
				cub::CountingInputIterator<int>(0),
				CompareToImage_CUDA_InputOp<NumDerivatives>(
					B, H, W, colorInput, gradientsInput, colorReference)
				);
		CompareToImage_CUDA_OutputIterator<NumDerivatives> output =
			CompareToImage_CUDA_OutputIterator<NumDerivatives>(
				differenceOut, gradientsOut);

		// Determine temporary device storage requirements
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		CUMAT_SAFE_CALL(
			cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output, numItems, stream));

		// Allocate temporary storage
		auto alloc = at::cuda::getCUDADeviceAllocator();
		d_temp_storage = alloc->raw_allocate(temp_storage_bytes);

		CUMAT_SAFE_CALL(
			cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output, numItems, stream));
		alloc->raw_deallocate(d_temp_storage);
	});
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}
