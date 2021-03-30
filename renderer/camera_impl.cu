//no include of camera.h
//torch/types.h and cuda don't mix well

#include "renderer_utils.cuh"

#include <cuMat/src/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuMat/../third-party/cub/cub.cuh>
#include <cuMat/src/Iterator.h>

#include "renderer_adjoint.cuh"

namespace kernel
{
	namespace
	{
		__host__ __device__ __forceinline__ void
		generateRaysForwardImpl(
			const Tensor3Read& viewport,
			float tanFovX, float tanFovY, int b, int y, int x,
			int H, int W,
			Tensor4RW& rayStart, Tensor4RW& rayDir)
		{
			real3 eye = fetchReal3(viewport, b, 0);
			real3 right = fetchReal3(viewport, b, 1);
			real3 up = fetchReal3(viewport, b, 2);
			real3 front = cross(up, right);

			//to normalized coordinates
			const real_t fx = 2 * (x + 0.5f) / real_t(W) - 1; //NDC in [-1,+1]
			const real_t fy = 2 * (y + 0.5f) / real_t(H) - 1;

			real3 dir = front + fx * tanFovX * right + fy * tanFovY * up;
			dir = normalize(dir);
			writeReal3(eye, rayStart, b, y, x);
			writeReal3(dir, rayDir, b, y, x);
		}

		__global__ void GenerateRaysForwardImplDevice(dim3 virtual_size,
			Tensor3Read viewport,
			float tanFovX, float tanFovY, int H, int W,
			Tensor4RW rayStart, Tensor4RW rayDir)
		{
			KERNEL_3D_LOOP(x, y, b, virtual_size)
			{
				generateRaysForwardImpl(viewport, tanFovX, tanFovY, b, y, x, H, W, rayStart, rayDir);
			}
			KERNEL_3D_LOOP_END
		}

		__host__ void GenerateRaysForwardImplHost(dim3 virtual_size,
			Tensor3Read viewport,
			float tanFovX, float tanFovY, int H, int W,
			Tensor4RW rayStart, Tensor4RW rayDir)
		{
			int count = virtual_size.x * virtual_size.y * virtual_size.z;
#pragma omp parallel for schedule(static)
			for (int __i = 0; __i < count; ++__i)
			{
				int b = __i / (virtual_size.x * virtual_size.y);
				int y = (__i - (b * virtual_size.x * virtual_size.y)) / virtual_size.x;
				int x = __i - virtual_size.x * (y + virtual_size.y * b);
				generateRaysForwardImpl(viewport, tanFovX, tanFovY, b, y, x, H, W, rayStart, rayDir);
			}
		}
	}

	void generateRaysForward(
		const Tensor3Read& viewport,
		float fovY, int B, int H, int W,
		Tensor4RW& rayStart, Tensor4RW& rayDir,
		bool cuda)
	{
		float fovX = fovY * float(W) / float(H);
		float tanFovX = tanf(fovX / 2);
		float tanFovY = tanf(fovY / 2);

		if (cuda)
		{
			cuMat::Context& ctx = cuMat::Context::current();
			cudaStream_t stream = at::cuda::getCurrentCUDAStream();
			auto cfg = ctx.createLaunchConfig3D(W, H, B, GenerateRaysForwardImplDevice);
			GenerateRaysForwardImplDevice
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, viewport, tanFovX, tanFovY, H, W, rayStart, rayDir);
			CUMAT_CHECK_ERROR();
		} else
		{
			dim3 virtual_size{ static_cast<unsigned>(W), static_cast<unsigned>(H), static_cast<unsigned>(B) };
			GenerateRaysForwardImplHost
				(virtual_size, viewport, tanFovX, tanFovY, H, W, rayStart, rayDir);
		}
	}

	namespace
	{
		struct viewport_matrix
		{
			real3 eye{ 0,0,0 };
			real3 right{ 0,0,0 };
			real3 up{ 0,0,0 };
			friend __host__ __device__ __inline__ viewport_matrix operator+(
				const viewport_matrix& m1, const viewport_matrix& m2)
			{
				return {
					m1.eye + m2.eye,
					m1.right + m2.right,
					m1.up + m2.up
				};
			}
		};

		struct GenerateRaysAdjoint_InputOp
		{
			const int3 strides;
			const int3 sizes;
			const Tensor3Read viewport;
			const Tensor4Read adj_rayStart;
			const Tensor4Read adj_rayDir;
			const float tanFovX;
			const float tanFovY;

			__host__ __device__ __inline__ GenerateRaysAdjoint_InputOp(const Tensor3Read& viewport,
				const Tensor4Read& adj_rayStart, const Tensor4Read& adj_rayDir, float tanFovX, float tanFovY)
				: strides({ adj_rayStart.size(1) * adj_rayStart.size(2), adj_rayStart.size(2), 1 }),
				sizes({ adj_rayStart.size(0), adj_rayStart.size(1), adj_rayStart.size(2) }),
				viewport(viewport),
				adj_rayStart(adj_rayStart),
				adj_rayDir(adj_rayDir),
				tanFovX(tanFovX),
				tanFovY(tanFovY)
			{}

			__host__ __device__ __forceinline__
				viewport_matrix operator()(const int& idx) const
			{
				int b = (idx / strides.x) % sizes.x;
				int y = (idx / strides.y) % sizes.y;
				int x = (idx / strides.z) % sizes.z;
				int H = sizes.y, W = sizes.z;

				real3 eye = fetchReal3(viewport, b, 0);
				real3 right = fetchReal3(viewport, b, 1);
				real3 up = fetchReal3(viewport, b, 2);
				real3 front = cross(up, right);

				//to normalized coordinates
				const real_t fx = 2 * (x + 0.5f) / real_t(W) - 1; //NDC in [-1,+1]
				const real_t fy = 2 * (y + 0.5f) / real_t(H) - 1;

				//forward code
				real3 dir = front + fx * tanFovX * right + fy * tanFovY * up;
				real3 dirNormalized = normalize(dir);
				//dirNormalized is the output

				//adjoint code
				real3 adj_eye = fetchReal3(adj_rayStart, b, y, x);
				real3 adj_dirNormalized = fetchReal3(adj_rayDir, b, y, x);

				//real3 dirNormalized = normalize(dir);
				real3 adj_dir = adjNormalize(dir, adj_dirNormalized);
				//real3 dir = front + fx * tanFovX * right + fy * tanFovY * up;
				real3 adj_front = adj_dir;
				real3 adj_right = adj_dir * (fx * tanFovX);
				real3 adj_up = adj_dir * (fy * tanFovY);
				//real3 front = cross(up, right);
				adjCross(up, right, adj_front, adj_up, adj_right);

				//output
				return viewport_matrix{
					adj_eye,
					adj_right,
					adj_up };
			}
		};
		using GenerateRaysAdjoint_InputIterator = cub::TransformInputIterator<
			viewport_matrix,
			GenerateRaysAdjoint_InputOp,
			cub::CountingInputIterator<int>>;

		struct GenerateRaysAdjoint_OutputAccessor
		{
			typedef GenerateRaysAdjoint_OutputAccessor self_type;
			typedef viewport_matrix value_type;
			
			kernel::Tensor3RW adjViewportOut;
			int index;

			__host__ __device__ __forceinline__ GenerateRaysAdjoint_OutputAccessor(const kernel::Tensor3RW& adjViewportOut, int index)
				: adjViewportOut(adjViewportOut),
				  index(index)
			{
			}
			__host__ __device__ __forceinline__ self_type& operator=(const value_type& value)
			{
				writeReal3(value.eye, adjViewportOut, index, 0);
				writeReal3(value.right, adjViewportOut, index, 1);
				writeReal3(value.up, adjViewportOut, index, 2);
				return *this;
			}
		};

		struct GenerateRaysAdjoint_OutputIterator
		{
			// Required iterator traits
			typedef GenerateRaysAdjoint_OutputIterator Base;
			typedef GenerateRaysAdjoint_OutputIterator self_type; ///< My own type
			typedef int difference_type; ///< Type to express the result of subtracting one iterator from another
			using ValueType = viewport_matrix;
			using value_type = void; ///< The type of the element the iterator can point to
			using pointer = void; ///< The type of a pointer to an element the iterator can point to
			using reference = GenerateRaysAdjoint_OutputAccessor; ///< The type of a reference to an element the iterator can point to
			using iterator_category = std::random_access_iterator_tag;
			
			kernel::Tensor3RW adjViewportOut;
			int index;

			__host__ __device__ GenerateRaysAdjoint_OutputIterator(
				const kernel::Tensor3RW& adjViewportOut, int index=0)
				: adjViewportOut(adjViewportOut)
				, index(index)
			{}

			template <typename Distance>
			__host__ __device__ __forceinline__ reference operator[](Distance n)
			{
				return GenerateRaysAdjoint_OutputAccessor(adjViewportOut, index + n);
			}

			__host__ __device__ __forceinline__ reference operator*()
			{
				return GenerateRaysAdjoint_OutputAccessor(adjViewportOut, index);
			}

			/// Postfix increment
			__host__ __device__ __forceinline__ self_type operator++(int)
			{
				self_type retval = *this;
				index++;
				return retval;
			}


			/// Prefix increment
			__host__ __device__ __forceinline__ self_type operator++()
			{
				index++;
				return *this;
			}

			/// Addition
			template <typename Distance>
			__host__ __device__ __forceinline__ self_type operator+(Distance n) const
			{
				return self_type(adjViewportOut, index + n);
			}

			/// Addition assignment
			template <typename Distance>
			__host__ __device__ __forceinline__ self_type& operator+=(Distance n)
			{
				index += n;
				return *this;
			}

			/// Subtraction
			template <typename Distance>
			__host__ __device__ __forceinline__ self_type operator-(Distance n) const
			{
				return self_type(adjViewportOut, index - n);
			}

			/// Subtraction assignment
			template <typename Distance>
			__host__ __device__ __forceinline__ self_type& operator-=(Distance n)
			{
				index -= n;
				return *this;
			}

			/// Distance
			__host__ __device__ __forceinline__ difference_type operator-(self_type other) const
			{
				return index - other.index;
			}

			/// Equal to
			__host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
			{
				return (index == rhs.index);
			}

			/// Not equal to
			__host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
			{
				return (index != rhs.index);
			}
		};
		
	}

	void generateRaysAdjoint(
		const Tensor3Read& viewport,
		float fovY, int B, int H, int W,
		const Tensor4Read& adj_rayStart, const Tensor4Read& adj_rayDir,
		Tensor3RW& adj_viewport,
		bool cuda)
	{
		float fovX = fovY * float(W) / float(H);
		float tanFovX = tanf(fovX / 2);
		float tanFovY = tanf(fovY / 2);
		
		const auto input = GenerateRaysAdjoint_InputIterator(
			cub::CountingInputIterator<int>(0),
			GenerateRaysAdjoint_InputOp(viewport, adj_rayStart, adj_rayDir, tanFovX, tanFovY)
		);
		const auto output = GenerateRaysAdjoint_OutputIterator(adj_viewport);

		const int numEntries = H * W;
		const int numBatches = B;
		const cuMat::CountingInputIterator<int> iterOffsets(0, numEntries);

		if (cuda)
		{
			cudaStream_t stream = at::cuda::getCurrentCUDAStream();
			
			// Determine temporary device storage requirements
			void* d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			CUMAT_SAFE_CALL(
				cub::DeviceSegmentedReduce::Sum(
					d_temp_storage, temp_storage_bytes,
					input, output, numBatches, iterOffsets, iterOffsets + 1,
					stream));

			// Allocate temporary storage
			auto alloc = at::cuda::getCUDADeviceAllocator();
			d_temp_storage = alloc->raw_allocate(temp_storage_bytes);

			CUMAT_SAFE_CALL(
				cub::DeviceSegmentedReduce::Sum(
					d_temp_storage, temp_storage_bytes,
					input, output, numBatches, iterOffsets, iterOffsets + 1,
					stream));
			alloc->raw_deallocate(d_temp_storage);
		}
		else
		{
#pragma omp parallel for
			for (int b=0; b<B; ++b)
			{
				auto input_local = input + b * numEntries;
				auto output_local = output + b;
				*output_local = std::accumulate(input_local, input_local + numEntries, viewport_matrix());
			}
		}
	}
}
