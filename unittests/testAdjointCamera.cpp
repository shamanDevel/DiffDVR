#include <catch.hpp>
#include <Eigen/Core>
#include <random>

#include "check_adjoint.h"
#include "test_utils.h"
#include "renderer_blending.cuh"


static void testAdjointCamera()
{
	typedef empty TmpStorage_t;
	typedef VectorXr Vector_t;
	static int numSteps = 10;
	static const real3 boxMin = make_real3(-0.2f, -0.3f, -0.4f);
	static const real3 voxelSize = make_real3(0.1f, 0.05f, 0.07f);

	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		const real3 rayStart = fromEigen3(x.segment(0, 3));
		const real3 rayDir = fromEigen3(x.segment(3, 3));
		const real_t stepsize = x[6];
		Vector_t result = Vector_t::Zero(numSteps * 3);
		for (int i=0; i<numSteps; ++i)
		{
			real_t tcurrent = i * stepsize;
			real3 worldPos = rayStart + tcurrent * rayDir;
			real3 volumePos = (worldPos - boxMin) / voxelSize;
			result.segment(3 * i, 3) = toEigen(volumePos);
		}
		return result;
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		const real3 rayStart = fromEigen3(x.segment(0, 3));
		const real3 rayDir = fromEigen3(x.segment(3, 3));
		const real_t stepsize = x[6];
		real3 adj_rayStart = make_real3(0);
		real3 adj_rayDir = make_real3(0);
		real_t adj_stepSize = 0;
		for (int i=numSteps-1; i>=0; --i)
		{
			//run part of the forward code again
			real_t tcurrent = i * stepsize;
			real3 worldPos = rayStart + tcurrent * rayDir;
			real3 volumePos = (worldPos - boxMin) / voxelSize;

			real3 adj_volumePos = fromEigen3(g.segment(3 * i, 3));
			
			//adjoint stepping
			real3 adj_worldPos = adj_volumePos / voxelSize;
			adj_rayStart += adj_worldPos;
			adj_rayDir += adj_worldPos * tcurrent;
			real_t adj_tcurrent = dot(adj_worldPos, rayDir);
			adj_stepSize += adj_tcurrent * i;
		}

		z.segment(0, 3) = toEigen(adj_rayStart);
		z.segment(3, 3) = toEigen(adj_rayDir);
		z[6] = adj_stepSize;
	};

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<real_t> distr(0.01, 0.99);
	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		Vector_t x(7);
		for (int j = 0; j < 7; ++j) x[j] = distr(rnd);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-5, 1e-5, 1e-6);
	}
}

TEST_CASE("Adjoint-Camera", "[adjoint]")
{
	testAdjointCamera();
}


