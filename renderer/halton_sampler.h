#pragma once
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <cuda_runtime_api.h>

#include "commons.h"

BEGIN_RENDERER_NAMESPACE

namespace detail
{
	uint32_t ReverseBits32(uint32_t n) {
		n = (n << 16) | (n >> 16);
		n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
		n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
		n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
		n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
		return n;
	}

	uint64_t ReverseBits64(uint64_t n) {
		uint64_t n0 = ReverseBits32((uint32_t)n);
		uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
		return (n0 << 32) | n1;
	}
	
	static constexpr double OneMinusEpsilon = 1.0 - 1e-7;
	template <int base, typename Float>
	__host__ __device__ Float RadicalInverseSpecialized(uint64_t a) {
		const Float invBase = (Float)1 / (Float)base;
		uint64_t reversedDigits = 0;
		Float invBaseN = 1;
		while (a) {
			uint64_t next = a / base;
			uint64_t digit = a - next * base;
			reversedDigits = reversedDigits * base + digit;
			invBaseN *= invBase;
			a = next;
		}
		assert(reversedDigits * invBaseN <= 1.00001);
		return std::min(
			invBaseN * reversedDigits,
			(Float)OneMinusEpsilon);
	}
	template<>
	__host__ __device__ float RadicalInverseSpecialized<2, float>(uint64_t a)
	{
		return static_cast<float>(ReverseBits64(a) * 0x1p-64);
	}
	template<>
	__host__ __device__ double RadicalInverseSpecialized<2, double>(uint64_t a)
	{
		return ReverseBits64(a) * 0x1p-64;
	}
}

class HaltonSampler
{
public:
	//Source: https://github.com/mmp/pbrt-v3/blob/master/src/core/lowdiscrepancy.cpp
	static constexpr int PrimeTableSize = 100;
	static constexpr int Primes[PrimeTableSize] = {
		2, 3, 5, 7, 11,
		13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
		97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
		173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
		257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
		349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
		439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523,
		541
	};
	static constexpr int PrimeSums[PrimeTableSize] = {
		0, 2, 5, 10, 17,
		28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639,
		712, 791, 874, 963, 1060, 1161, 1264, 1371, 1480, 1593, 1720, 1851, 1988,
		2127, 2276, 2427, 2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028,
		4227, 4438, 4661, 4888, 5117, 5350, 5589, 5830, 6081, 6338, 6601, 6870,
		7141, 7418, 7699, 7982, 8275, 8582, 8893, 9206, 9523, 9854, 10191, 10538,
		10887, 11240, 11599, 11966, 12339, 12718, 13101, 13490, 13887, 14288, 14697,
		15116, 15537, 15968, 16401, 16840, 17283, 17732, 18189, 18650, 19113, 19580,
		20059, 20546, 21037, 21536, 22039, 22548, 23069, 23592
	};

	template <int base>
	static uint64_t InverseRadicalInverse(uint64_t inverse, int nDigits) {
		uint64_t index = 0;
		for (int i = 0; i < nDigits; ++i) {
			uint64_t digit = inverse % base;
			inverse /= base;
			index = index * base + digit;
		}
		return index;
	}
	
	/**
	 * \brief Generates low-discrepancy pseudo-random numbers in [0,1]
	 * \tparam base the base, must be a prime
	 * \tparam Float the return type, float or double
	 * \param a the sample index, a non-negative integer
	 * \return the uniform pseudo-random number.
	 */
	template <int base, typename Float>
	__host__ __device__ static Float Sample(uint64_t a) {
		return detail::RadicalInverseSpecialized<base, Float>(a);
	}
};

END_RENDERER_NAMESPACE

