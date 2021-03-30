#include <catch.hpp>
#include <Eigen/Core>
#include <random>

#include "check_adjoint.h"
#include "test_utils.h"
#include "renderer_adjoint.cuh"



TEST_CASE("Adjoint-Cross", "[adjoint]")
{
	typedef empty TmpStorage_t;
	typedef VectorXr Vector_t;
	
	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		const real3 a = fromEigen3(x.head(3));
		const real3 b = fromEigen3(x.tail(3));
		const real3 c = cross(a, b);
		return toEigen(c);
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		const real3 a = fromEigen3(x.head(3));
		const real3 b = fromEigen3(x.tail(3));
		const real3 adj_c = fromEigen3(g);
		real3 adj_a = make_real3(0), adj_b = make_real3(0);
		kernel::adjCross(a, b, adj_c, adj_a, adj_b);
		z.head(3) = toEigen(adj_a);
		z.tail(3) = toEigen(adj_b);
	};

	std::default_random_engine rnd(42);
	std::normal_distribution<real_t> distr;
	int N = 20;
	for (int i=0; i<N; ++i)
	{
		INFO("N=" << i);
		const real3 a = make_real3(
			distr(rnd), distr(rnd), distr(rnd));
		const real3 b = make_real3(
			distr(rnd), distr(rnd), distr(rnd));
		Vector_t x(6);
		x.head(3) = toEigen(a);
		x.tail(3) = toEigen(b);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint);
	}
}


TEST_CASE("Adjoint-Normalize", "[adjoint]")
{
	typedef empty TmpStorage_t;
	typedef Vector3r Vector_t;

	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		const real3 a = fromEigen3(x);
		const real3 c = normalize(a);
		return toEigen(c);
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		const real3 a = fromEigen3(x);
		const real3 adj_c = fromEigen3(g);
		real3 adj_a = kernel::adjNormalize(a, adj_c);
		z = toEigen(adj_a);
	};

	std::default_random_engine rnd(42);
	std::normal_distribution<real_t> distr;
	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		const real3 a = make_real3(
			distr(rnd), distr(rnd), distr(rnd));
		Vector_t x = toEigen(a);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-6, 1e-3, 1e-5);
	}
}

