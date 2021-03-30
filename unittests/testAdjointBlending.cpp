#include <catch.hpp>
#include <Eigen/Core>
#include <random>

#include "check_adjoint.h"
#include "test_utils.h"
#include "renderer_blending.cuh"


template<kernel::BlendMode blendMode>
void testAdjointBlending()
{
	typedef empty TmpStorage_t;
	typedef VectorXr Vector_t;

	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		const real4 acc = fromEigen4(x.segment(0, 4));
		const real4 current = fromEigen4(x.segment(4, 4));
		const real_t stepsize = x[8];

		const real4 output = kernel::Blending<blendMode>::blend(
			acc, current, stepsize);

		return toEigen(output);
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		const real4 acc = fromEigen4(x.segment(0, 4));
		const real4 current = fromEigen4(x.segment(4, 4));
		const real_t stepsize = x[8];

		const real4 output = fromEigen4(e);
		const real4 adj_output = fromEigen4(g);

		real4 adj_acc, adj_current;
		real_t adj_stepsize;

		real4 acc_in = kernel::Blending<blendMode>::adjoint(
			output, current, stepsize, adj_output,
			adj_acc, adj_current, adj_stepsize);

		INFO("input-acc: " << acc);
		INFO("reconstructed-acc: " << acc_in);
		REQUIRE(acc.x == Approx(acc_in.x));
		REQUIRE(acc.y == Approx(acc_in.y));
		REQUIRE(acc.z == Approx(acc_in.z));
		REQUIRE(acc.w == Approx(acc_in.w));

		z.segment<4>(0) = toEigen(adj_acc);
		z.segment<4>(4) = toEigen(adj_current);
		z[8] = adj_stepsize;
	};

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<real_t> distr(0.01, 0.99);
	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		Vector_t x(9);
		for (int j = 0; j < 9; ++j) x[j] = distr(rnd);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-5, 1e-5, 1e-6);
	}
}

TEST_CASE("Adjoint-Blending-BeerLambert", "[adjoint]")
{
	testAdjointBlending<kernel::BlendBeerLambert>();
}

TEST_CASE("Adjoint-Blending-Alpha", "[adjoint]")
{
	testAdjointBlending<kernel::BlendAlpha>();
}

