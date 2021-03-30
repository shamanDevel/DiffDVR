#pragma once

#include "helper_math.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef USE_DOUBLE_PRECISION
#error "Must define USE_DOUBLE_PRECISION with value 1 (double) or 0 (float)"
#else

#if USE_DOUBLE_PRECISION==0
typedef float real_t;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#define make_real2 make_float2
#define make_real3 make_float3
#define make_real4 make_float4
#define make_real2in make_float2in
#define make_real3in make_float3in
#define make_real4in make_float4in

#else
typedef double real_t;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#define make_real2 make_double2
#define make_real3 make_double3
#define make_real4 make_double4
#define make_real2in make_double2in
#define make_real3in make_double3in
#define make_real4in make_double4in

#endif
#endif

namespace kernel {
	// STRUCT TEMPLATE integral_constant
	template<class _Ty, _Ty _Val>
	struct integral_constant
	{	// convenient template for integral constant types
		enum
		{
			value = _Val
		};
	};
	// STRUCT TEMPLATE conditional
	template <bool _Test, class _Ty1, class _Ty2>
	struct conditional { // Choose _Ty1 if _Test is true, and _Ty2 otherwise
		using type = _Ty1;
	};
	template <class _Ty1, class _Ty2>
	struct conditional<false, _Ty1, _Ty2> {
		using type = _Ty2;
	};

	// Choose _Ty1 if _Test is true, and _Ty2 otherwise
	template <bool _Test, class _Ty1, class _Ty2>
	using conditional_t = typename conditional<_Test, _Ty1, _Ty2>::type;
}

