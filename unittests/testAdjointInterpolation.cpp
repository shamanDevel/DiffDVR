#include <catch.hpp>
#include <Eigen/Core>
#include <random>

#include "check_adjoint.h"
#include "test_utils.h"
#include "renderer_interpolation.cuh"


template<typename Tensor_t>
Tensor_t accessor4D(const VectorXr& tf, const int3& xyz)
{
	const int64_t sizes[] = {
		1, xyz.x, xyz.y, xyz.z
	};
	const int64_t strides[] = {
		0, xyz.y*xyz.z, xyz.z, 1
	};
	return Tensor_t(
		const_cast<real_t*>(tf.data()), sizes, strides);
}

template<bool hasVolumeDerivative>
void testAdjointInterpolation()
{
	typedef empty TmpStorage_t;
	typedef VectorXr Vector_t;

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<real_t> distr(0.01, 0.99);
	static const int3 resolution = make_int3(2, 3, 4);
	static const int resolution_prod = resolution.x * resolution.y * resolution.z;
	static const real3 voxelSize = make_real3(1.0f) / make_real3(resolution);
	Vector_t baseVolume(resolution_prod);
	for (int j = 0; j < baseVolume.size(); ++j) baseVolume[j] = distr(rnd);

	auto forward = [&baseVolume](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		const real3 pos = fromEigen3(x.segment(0, 3));

		const Vector_t volume = hasVolumeDerivative
			? x.segment(3, resolution_prod)
			: baseVolume;
		const auto volume_acc = accessor4D<kernel::Tensor4Read>(volume, resolution);

		kernel::VolumeInterpolation<kernel::FilterTrilinear> volumeInterpolation;
		real_t density = volumeInterpolation.fetch(volume_acc, resolution, 0, pos);

		Vector_t result(1);
		result[0] = density;
		return result;
	};
	auto adjoint = [baseVolume](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		const real3 pos = fromEigen3(x.segment(0, 3));

		const Vector_t volume = hasVolumeDerivative
			? x.segment(3, resolution_prod)
			: baseVolume;
		const auto volume_acc = accessor4D<kernel::Tensor4Read>(volume, resolution);

		real_t density = e[0];
		real_t adj_density = g[0];
		real3 adj_pos;
		Vector_t adjVolume = Vector_t::Zero(resolution_prod);
		auto adjVolume_acc = accessor4D<kernel::BTensor4RW>(adjVolume, resolution);

		{
			kernel::VolumeInterpolation<kernel::FilterTrilinear, true, hasVolumeDerivative> volumeInterpolation(
				0, resolution, adjVolume_acc);
			volumeInterpolation.fetch(volume_acc, resolution, 0, pos);
			volumeInterpolation.adjoint(adj_density, adj_pos);
		}

		z.segment(0, 3) = toEigen(adj_pos);
		if (hasVolumeDerivative) z.segment(3, resolution_prod) = adjVolume;
	};

	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		Vector_t x;
		if (hasVolumeDerivative) {
			x.resize(3 + resolution_prod);
			x.segment(3, resolution_prod) = baseVolume;
		}
		else {
			x.resize(3);
		}
		x[0] = distr(rnd) / voxelSize.x;
		x[1] = distr(rnd) / voxelSize.y;
		x[2] = distr(rnd) / voxelSize.z;
		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-5, 1e-5, 1e-6);
	}
}

TEST_CASE("Adjoint-Interpolation-NoVolumeDerivatives", "[adjoint]")
{
	testAdjointInterpolation<false>();
}

TEST_CASE("Adjoint-Interpolation-withVolumeDerivatives", "[adjoint]")
{
	testAdjointInterpolation<true>();
}


