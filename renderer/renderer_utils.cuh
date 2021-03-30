#pragma once

#ifndef CUDA_NO_HOST
#include <assert.h>
#else
#define assert(x)
#endif

#include "helper_math.cuh"
#include "renderer_settings.cuh"

namespace kernel
{

	//=========================================
	// RENDERER UTILITIES
	//=========================================

	inline __host__ __device__ unsigned int rgbaToInt(float r, float g, float b, float a)
	{
		r = clamp(r * 255, 0.0f, 255.0f);
		g = clamp(g * 255, 0.0f, 255.0f);
		b = clamp(b * 255, 0.0f, 255.0f);
		a = clamp(a * 255, 0.0f, 255.0f);
		return (unsigned(a) << 24) | (unsigned(b) << 16)
			| (unsigned(g) << 8) | (unsigned(r));
		//return 0xff000000 | (int(b) << 16) | (int(g) << 8) | int(r);
	}
	inline __host__ __device__ unsigned int float4ToInt(float4 rgba)
	{
		return rgbaToInt(rgba.x, rgba.y, rgba.z, rgba.w);
	}
	inline __host__ __device__ float4 intToFloat4(unsigned int rgba)
	{
		return make_float4(
			(rgba & 0xff) / 255.0f,
			((rgba >> 8) & 0xff) / 255.0f,
			((rgba >> 16) & 0xff) / 255.0f,
			((rgba >> 24) & 0xff) / 255.0f
		);
	}
	
	inline __host__ __device__ real4 matmul(const real4 mat[4], real4 v)
	{
		return make_real4(
			dot(mat[0], v),
			dot(mat[1], v),
			dot(mat[2], v),
			dot(mat[3], v)
		);
	}

	__host__ __device__ __forceinline__ real3 matmul_proj(const real4 mat[4], real_t x, real_t y, real_t z)
	{
		real4 screen = make_real4(x, y, z, 1);
		real4 world = matmul(mat, screen);
		return make_real3(world / world.w);
	}
	
	__host__ __device__ __forceinline__ void intersectionRayAABB(
		const real3& rayStart, const real3& rayDir,
		const real3& boxMin, const real3& boxSize,
		real_t& tmin, real_t& tmax)
	{
		real3 invRayDir = 1.0f / rayDir;
		real_t t1 = (boxMin.x - rayStart.x) * invRayDir.x;
		real_t t2 = (boxMin.x + boxSize.x - rayStart.x) * invRayDir.x;
		real_t t3 = (boxMin.y - rayStart.y) * invRayDir.y;
		real_t t4 = (boxMin.y + boxSize.y - rayStart.y) * invRayDir.y;
		real_t t5 = (boxMin.z - rayStart.z) * invRayDir.z;
		real_t t6 = (boxMin.z + boxSize.z - rayStart.z) * invRayDir.z;
		tmin = rmax(rmax(rmin(t1, t2), rmin(t3, t4)), rmin(t5, t6));
		tmax = rmin(rmin(rmax(t1, t2), rmax(t3, t4)), rmax(t5, t6));
	}

	template<typename Tensor>
	__host__ __device__ __forceinline__ real3 fetchReal3(const Tensor& t, int b)
	{
		return make_real3(t[b][0], t[b][1], t[b][2]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ real3 fetchReal3(const Tensor& t, int b, int i)
	{
		return make_real3(t[b][i][0], t[b][i][1], t[b][i][2]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ real3 fetchReal3(const Tensor& t, int b, int y, int x)
	{
		return make_real3(t[b][y][x][0], t[b][y][x][1], t[b][y][x][2]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ real4 fetchReal4(const Tensor& t, int b, int i)
	{
		return make_real4(t[b][i][0], t[b][i][1], t[b][i][2], t[b][i][3]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ real4 fetchReal4(const Tensor& t, int b, int y, int x)
	{
		return make_real4(t[b][y][x][0], t[b][y][x][1], t[b][y][x][2], t[b][y][x][3]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ real4 fetchReal4(const Tensor& t, int b, int y, int x, int i)
	{
		return make_real4(t[b][y][x][i][0], t[b][y][x][i][1], t[b][y][x][i][2], t[b][y][x][i][3]);
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ int4 fetchInt4(const Tensor& t, int b, int i)
	{
		return make_int4(t[b][i][0], t[b][i][1], t[b][i][2], t[b][i][3]);
	}

	template<typename Tensor>
	__host__ __device__ __forceinline__ void writeReal3(const real3& v, Tensor& t, int b, int i)
	{
		t[b][i][0] = v.x;
		t[b][i][1] = v.y;
		t[b][i][2] = v.z;
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ void writeReal3(const real3& v, Tensor& t, int b, int y, int x)
	{
		t[b][y][x][0] = v.x;
		t[b][y][x][1] = v.y;
		t[b][y][x][2] = v.z;
	}
	template<typename Tensor>
	__host__ __device__ __forceinline__ void writeReal4(const real4& v, Tensor& t, int b, int y, int x)
	{
		t[b][y][x][0] = v.x;
		t[b][y][x][1] = v.y;
		t[b][y][x][2] = v.z;
		t[b][y][x][3] = v.w;
	}

	//=========================================
	// MATH UTILITIES
	//=========================================

	/**
	 * \brief Computes log-sum-exp of the input values 'values'.
	 * In detail: the result is
	 *
	 * \f[
	 *    y = \log(\sum_{i=0}^{N-1}(\exp(x(i))))
	 * \f]
	 *    
	 * where x(i)=values(i).
	 * Note that this function is computed numerically more stable
	 * than the naive sum.
	 * 
	 * The values need to be traversed twice, hence they are
	 * provided as a functor/lambda:
	 * F must define an operator()(int i) that returns the value
	 * at index 'i' of type 'T'.
	 *
	 * Supported types for T: float, double, float[2,3,4], double[2,3,4].
	 * 
	 * \tparam T the return type
	 * \tparam F the functor/lambda type that returns the value at index 'i'
	 * \param N the number of entries
	 * \param values the functor/lambda that returns the value at index 'i'
	 * \return the result of log-sum-exp
	 */
	template<typename T, typename F>
	__host__ __device__ __forceinline__ T logSumExp(int N, const F& values)
	{
		assert(N > 0);
		//1st pass: compute max
		T xmax = values(0);
		for (int i = 1; i < N; ++i)
			xmax = rmax(xmax, values(i));
		//2nd pass: compute sum
		T xsum = rexp(values(0) - xmax);
		for (int i = 1; i < N; ++i)
			xsum += rexp(values(i) - xmax);
		return xmax + rlog(xsum);
	}

	/**
	 * \brief Computes log-sum-exp of the input values 'values'
	 * with additional scaling.
	 * <b>Assumes that the sum is positive!</b>
	 * In detail: the result is
	 *
	 * \f[
	 *    y = \log(\sum_{i=0}^{N-1}(b(i)*\exp(x(i))))
	 * \f]
	 *
	 * where x(i)=values(i), b(i)=scaling(i).
	 * Note that this function is computed numerically more stable
	 * than the naive sum.
	 *
	 * The values need to be traversed twice, hence they are
	 * provided as a functor/lambda:
	 * F must define an operator()(int i) that returns the value
	 * at index 'i' of type 'T'.
	 *
	 * Supported types for T: float, double, float[2,3,4], double[2,3,4].
	 *
	 * \tparam T the return type
	 * \tparam F the functor/lambda type that returns the value at index 'i'
	 * \tparam B the functor/lambda type that returns the scaling at index 'i'
	 * \param N the number of entries
	 * \param values the functor/lambda that returns the value at index 'i'
	 * \param scaling the functor/lambda that returns the scaling at index 'i'
	 * \return the result of log-sum-exp
	 */
	template<typename T, typename F, typename B>
	__host__ __device__ __forceinline__ T logSumExpWithScaling(
		int N, const F& values, const B& scaling)
	{
		assert(N > 0);
		//1st pass: compute max
		T xmax = values(0);
		for (int i = 1; i < N; ++i)
			xmax = rmax(xmax, values(i));
		//2nd pass: compute sum
		T xsum = scaling(0)*rexp(values(0) - xmax);
		for (int i = 1; i < N; ++i)
			xsum += scaling(i)*rexp(values(i) - xmax);
		return xmax + rlog(xsum);
	}

	/**
	 * \brief Computes the log-MSE loss, that is, for inputs x and y it holds:
	 *
	 * \f[
	 *    (x-y)^2 = MSE(x,y) = \exp(logMSE(\log(x), \log(y)))
	 * \f]
	 *
	 * This function takes care of negative values.
	 * 
	 * \tparam T the scalar type, can be float or double
	 * \param logX the log of the first argument
	 * \param logY the log of the second argument
	 * \return log((exp(logX)-exp(logY))^2)
	 */
	template<typename T>
	__host__ __device__ __forceinline__ T logMSE(const T& logX, const T& logY)
	{
		T logA = rmax(logX, logY);
		T logB = rmin(logX, logY);

		return 2 * (rlog(1 - rexp(logB - logA)) + logA);
	}

	template<typename T>
	__host__ __device__ __forceinline__ void adjLogMSE(
		const T& logX, const T& logY, const T& adjOut,
		T& adjLogX, T& adjLogY)
	{
		//naive:
		//T denom = rexp(logX) - rexp(logY);
		//adjLogX = adjOut * rexp(logX) / denom;
		//adjLogY = -adjOut * rexp(logY) / denom;

		//accurate
		adjLogX = 2 * adjOut / (1 - rexp(logY - logX));
		adjLogY = 2 * adjOut / (1 - rexp(logX - logY));
	}
	
	/**
	 * \brief Computes the log-L1 loss, that is, for inputs x and y it holds:
	 *
	 * \f[
	 *    |x-y| = L1Loss(x,y) = \exp(logL1(\log(x), \log(y)))
	 * \f]
	 *
	 * This function takes care of negative values.
	 *
	 * \tparam T the scalar type, can be float or double
	 * \param logX the log of the first argument
	 * \param logY the log of the second argument
	 * \return log(|exp(logX)-exp(logY)|)
	 */
	template<typename T>
	__host__ __device__ __forceinline__ T logL1(const T& logX, const T& logY)
	{
		T logA = rmax(logX, logY);
		T logB = rmin(logX, logY);

		return (rlog(1 - rexp(logB - logA)) + logA);
	}

	template<typename T>
	__host__ __device__ __forceinline__ void adjLogL1(
		const T& logX, const T& logY, const T& adjOut,
		T& adjLogX, T& adjLogY)
	{
		//naive:
		//T denom = rexp(logX) - rexp(logY);
		//adjLogX = adjOut * rexp(logX) / denom;
		//adjLogY = -adjOut * rexp(logY) / denom;

		//accurate
		adjLogX = adjOut / (1 - rexp(logY - logX));
		adjLogY = adjOut / (1 - rexp(logX - logY));
	}
	
}

#ifdef __NVCC__
#define KERNEL_3D_LOOP(i, j, k, virtual_size) 												\
	for (ptrdiff_t __i = blockIdx.x * blockDim.x + threadIdx.x;							\
		 __i < virtual_size.x*virtual_size.y*virtual_size.z;										\
		 __i += blockDim.x * gridDim.x) {															\
		 ptrdiff_t k = __i / (virtual_size.x*virtual_size.y);							\
		 ptrdiff_t j = (__i - (k * virtual_size.x*virtual_size.y)) / virtual_size.x;	\
		 ptrdiff_t i = __i - virtual_size.x * (j + virtual_size.y * k);
#define KERNEL_3D_LOOP_END }
#else
#define KERNEL_3D_LOOP(i, j, k, virtual_size) assert(false); {ptrdiff_t i,j,k;
#define KERNEL_3D_LOOP_END }
#endif
