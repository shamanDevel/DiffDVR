#pragma once

#ifndef CUDA_NO_HOST
#include <type_traits>
#include <algorithm>
#else

//define standard types manually
typedef int                int32_t;
typedef long long          int64_t;

#endif

//=========================================
// (modified) copy of PyTorch's TensorAccessor.h
// for runtime compilation, it is easier to have no external dependencies
// (apart from cuda_runtime and standard libraries)
//=========================================


namespace kernel
{
	// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
	// is used to enable the __restrict__ keyword/modifier for the data
	// passed to cuda.
	template <typename T>
	struct DefaultPtrTraits {
		typedef T* PtrType;
	};
	template <typename T>
	struct RestrictPtrTraits {
		typedef T* __restrict__ PtrType;
	};

    // TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
	// For CUDA tensors it is used in device code (only). This means that we restrict ourselves
	// to functions and types available there (e.g. IntArrayRef isn't).

	// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class TensorAccessorBase {
    public:
        typedef typename PtrTraits<T>::PtrType PtrType;

        __host__ __device__ TensorAccessorBase(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : data_(data_), sizes_(sizes_), strides_(strides_) {}
        __host__ __device__ index_t stride(index_t i) const {
            return strides_[i];
        }
        __host__ __device__ index_t size(index_t i) const {
            return sizes_[i];
        }
        __host__ __device__ PtrType data() {
            return data_;
        }
        __host__ __device__ const PtrType data() const {
            return data_;
        }
    protected:
        PtrType data_;
        const index_t* sizes_;
        const index_t* strides_;
    };

    // The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
    // `Tensor.accessor<T, N>()`.
    // For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and only
    // indexing on the device uses `TensorAccessor`s.
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
    public:
        typedef typename PtrTraits<T>::PtrType PtrType;

        __host__ __device__ TensorAccessor(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : TensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

        __host__ __device__ TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
            return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + this->strides_[0] * i, this->sizes_ + 1, this->strides_ + 1);
        }

        __host__ __device__ const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
            return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + this->strides_[0] * i, this->sizes_ + 1, this->strides_ + 1);
        }
    };

    template<typename T, template <typename U> class PtrTraits, typename index_t>
    class TensorAccessor<T, 1, PtrTraits, index_t> : public TensorAccessorBase<T, 1, PtrTraits, index_t> {
    public:
        typedef typename PtrTraits<T>::PtrType PtrType;

        __host__ __device__ TensorAccessor(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
        __host__ __device__ T& operator[](index_t i) {
            return this->data_[this->strides_[0] * i];
        }
        __host__ __device__ const T& operator[](index_t i) const {
            return this->data_[this->strides_[0] * i];
        }
    };


    // GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on for CUDA `Tensor`s on the host
	// and as
	// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
	// in order to transfer them on the device when calling kernels.
	// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
	// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
	// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
	// on the device, so those functions are host only.
	//
	// If BroadcastEmptyDimensions is true, the stride of dimensions of size 1 is set to zero,
	// i.e. disabling any index computation for that dimension.
    template<typename T, size_t N,
		template <typename U> class PtrTraits = DefaultPtrTraits,
		typename index_t = int64_t,
		bool BroadcastEmptyDimensions = false>
    class GenericPackedTensorAccessorBase {
    public:
        typedef typename PtrTraits<T>::PtrType PtrType;
#ifndef CUDA_NO_HOST
        __host__ GenericPackedTensorAccessorBase()
	        : data_(nullptr) {}
    	
        __host__ GenericPackedTensorAccessorBase(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : data_(data_) {
            std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
            std::copy(strides_, strides_ + N, std::begin(this->strides_));
        	if constexpr(BroadcastEmptyDimensions)
        	{
                for (size_t i = 0; i < N; ++i)
                    if (this->sizes_[i] == 1) this->strides_[i] = 0;
        	}
        }

        // if index_t is not int64_t, we want to have an int64_t constructor
        template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
        __host__ GenericPackedTensorAccessorBase(
            PtrType data_,
            const source_index_t* sizes_,
            const source_index_t* strides_)
            : data_(data_) {
            for (int i = 0; i < N; i++) {
                this->sizes_[i] = sizes_[i];
                this->strides_[i] = strides_[i];
            }
            if constexpr (BroadcastEmptyDimensions)
            {
                for (size_t i = 0; i < N; ++i)
                    if (this->sizes_[i] == 1) this->strides_[i] = 0;
            }
        }
#endif

        __host__ __device__ index_t stride(index_t i) const {
            return strides_[i];
        }
        __host__ __device__ index_t size(index_t i) const {
            return sizes_[i];
        }
        __host__ __device__ PtrType data() {
            return data_;
        }
        __host__ __device__ const PtrType data() const {
            return data_;
        }
    protected:
        PtrType data_;
        index_t sizes_[N];
        index_t strides_[N];
    };

    template<typename T, size_t N,
		template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t,
        bool BroadcastEmptyDimensions = false>
    class GenericPackedTensorAccessor
		: public GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t, BroadcastEmptyDimensions> {
    public:
        using Type = T;
        typedef typename PtrTraits<T>::PtrType PtrType;

#ifndef CUDA_NO_HOST
        __host__ GenericPackedTensorAccessor() {} //uninitialized
    	
        __host__ GenericPackedTensorAccessor(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t, BroadcastEmptyDimensions>(data_, sizes_, strides_) {}

        // if index_t is not int64_t, we want to have an int64_t constructor
        template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
        __host__ GenericPackedTensorAccessor(
            PtrType data_,
            const source_index_t* sizes_,
            const source_index_t* strides_)
            : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t, BroadcastEmptyDimensions>(data_, sizes_, strides_) {}
#endif
    	
        __host__ __device__ TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
            index_t* new_sizes = this->sizes_ + 1;
            index_t* new_strides = this->strides_ + 1;
            return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + this->strides_[0] * i, new_sizes, new_strides);
        }

        __host__ __device__ const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
            const index_t* new_sizes = this->sizes_ + 1;
            const index_t* new_strides = this->strides_ + 1;
            return TensorAccessor<T, N - 1, PtrTraits, index_t>(this->data_ + this->strides_[0] * i, new_sizes, new_strides);
        }
    };

    template<typename T,
		template <typename U> class PtrTraits, typename index_t,
		bool BroadcastEmptyDimensions>
    class GenericPackedTensorAccessor<T, 1, PtrTraits, index_t, BroadcastEmptyDimensions>
		: public GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t, BroadcastEmptyDimensions> {
    public:
        using Type = T;
        typedef typename PtrTraits<T>::PtrType PtrType;

#ifndef CUDA_NO_HOST
        __host__ GenericPackedTensorAccessor() {} //uninitialized
    	
        __host__ GenericPackedTensorAccessor(
            PtrType data_,
            const index_t* sizes_,
            const index_t* strides_)
            : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t, BroadcastEmptyDimensions>(data_, sizes_, strides_) {}

        // if index_t is not int64_t, we want to have an int64_t constructor
        template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
        __host__ GenericPackedTensorAccessor(
            PtrType data_,
            const source_index_t* sizes_,
            const source_index_t* strides_)
            : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t, BroadcastEmptyDimensions>(data_, sizes_, strides_) {}
#endif
    	
        __host__ __device__ T& operator[](index_t i) {
            return this->data_[this->strides_[0] * i];
        }
        __host__ __device__ const T& operator[](index_t i) const {
            return this->data_[this->strides_[0] * i];
        }
    };

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t, false>;

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t, false>;

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using BroadcastingTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t, true>;

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using BroadcastingTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t, true>;
}
