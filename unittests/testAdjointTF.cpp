#include <catch.hpp>
#include <map>
#include <Eigen/Core>
#include <random>

#include "check_adjoint.h"
#include "test_utils.h"
#include "renderer_tf.cuh"

template<kernel::TFMode tfMode>
MatrixXr createTF(std::default_random_engine& rnd);

template<>
MatrixXr createTF<kernel::TFIdentity>(std::default_random_engine& rnd)
{
	MatrixXr tf(1, 2);
	std::uniform_real_distribution<real_t> distr;
	tf(0, 0) = distr(rnd);
	tf(0, 1) = distr(rnd);
	return tf;
}

template<>
MatrixXr createTF<kernel::TFTexture>(std::default_random_engine& rnd)
{
	static const int R = 8;
	MatrixXr tf(R, 4);
	std::uniform_real_distribution<real_t> distr;
	for (int r = 0; r < R; ++r)
		for (int c = 0; c < 4; ++c)
			tf(r, c) = distr(rnd);
	return tf;
}

template<>
MatrixXr createTF<kernel::TFLinear>(std::default_random_engine& rnd)
{
	static const int R = 5;
	std::uniform_real_distribution<real_t> distr(0.1, 0.9);
	std::vector<real_t> positions(R);
	positions[0] = 0.0;
	positions[R - 1] = 1.0;
	for (int r = 1; r < R - 1; ++r)
		positions[r] = distr(rnd);
	std::sort(positions.begin(), positions.end());
	
	MatrixXr tf(R, 5);
	for (int r=0; r<R; ++r)
	{
		for (int c = 0; c < 4; ++c)
			tf(r, c) = distr(rnd);
		tf(r, 4) = positions[r];
	}
	return tf;
}

template<>
MatrixXr createTF<kernel::TFGaussian>(std::default_random_engine& rnd)
{
	static const int R = 4;
	MatrixXr tf(R, 6);
	std::uniform_real_distribution<real_t> distr;
	for (int r = 0; r < R; ++r) {
		for (int c = 0; c < 4; ++c)
			tf(r, c) = distr(rnd);
		tf(r, 4) = distr(rnd); //mean
		tf(r, 5) = 0.1 + distr(rnd); //sigma
	}
	return tf;
}
template<>
MatrixXr createTF<kernel::TFGaussianLog>(std::default_random_engine& rnd)
{
	return createTF<kernel::TFGaussian>(rnd);
}

static VectorXr linearize(const MatrixXr& mat)
{
	Eigen::Map<const VectorXr> v(mat.data(), mat.size());
	return v;
}
static MatrixXr delinearize(const VectorXr& input, const MatrixXr& sizeReference)
{
	Eigen::Map<const MatrixXr> m(input.data(), sizeReference.rows(), sizeReference.cols());
	return m;
}
template<typename Tensor_t>
Tensor_t accessor3D(const MatrixXr& tf)
{
	const int64_t sizes[] = {
		1, tf.rows(), tf.cols()
	};
	const int64_t strides[] = {
		0, tf.rowStride(), tf.colStride()
	};
	return Tensor_t(
		const_cast<real_t*>(tf.data()), sizes, strides);
}

template<kernel::TFMode tfMode, bool hasTFDerivative>
void testAdjointTF(double epsilon=1e-6, double precision=1e-4, double margin=1e-6)
{
	typedef empty TmpStorage_t;
	typedef VectorXr Vector_t;

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<real_t> distr(0.01, 0.99);
	int N1 = 10; //TFs
	int N2 = 10; //densities
	for (int i1 = 0; i1 < N1; ++i1)
	{
		INFO("N1=" << i1);
		const MatrixXr tf = createTF<tfMode>(rnd);
		INFO("tf:\n" << tf);

		const VectorXr tfLinear = linearize(tf);
		const MatrixXr tfTest = delinearize(tfLinear, MatrixXr::Zero(tf.rows(), tf.cols()));
		REQUIRE(tf == tfTest);
		
		kernel::TransferFunctionEval<tfMode> tfEval;
		
		auto forward = [&tf, &tfEval](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
		{
			MatrixXr myTF;
			real_t myDensity = x[0];
			if constexpr (hasTFDerivative)
				myTF = delinearize(x.tail(x.size() - 1), tf);
			else
				myTF = tf;
			const kernel::Tensor3Read myTFacc = accessor3D<kernel::Tensor3Read>(myTF);
			
			const real4 color = tfEval.eval(myTFacc, 0, myDensity);
			return toEigen(color);
		};
		auto adjoint = [&tf, &tfEval](const Vector_t& x, const Vector_t& e, const Vector_t& g,
			Vector_t& z, const TmpStorage_t& tmp)
		{
			MatrixXr myTF;
			real_t myDensity = x[0];
			if constexpr (hasTFDerivative)
				myTF = delinearize(x.tail(x.size() - 1), tf);
			else
				myTF = tf;
			const auto myTFacc = accessor3D<kernel::Tensor3Read>(myTF);

			const real4 adj_color = fromEigen4(g);

			real_t adj_density;
			MatrixXr adj_tf = MatrixXr::Zero(tf.rows(), tf.cols());
			auto adj_tfAcc = accessor3D<kernel::BTensor3RW>(adj_tf);

			tfEval.adjoint<hasTFDerivative, false>(
				myTFacc, 0, myDensity, adj_color, adj_density, adj_tfAcc, nullptr);

			z[0] = adj_density;
			if constexpr(hasTFDerivative)
			{
				z.tail(z.size() - 1) = linearize(adj_tf);
			} else
			{
				REQUIRE(adj_tf.isZero());
			}
		};
		
		for (int i2 = 0; i2 < N2; ++i2)
		{
			INFO("N2=" << i2);
			real_t density = distr(rnd);
			INFO("density: " << density);

			Vector_t x;
			if constexpr(hasTFDerivative)
			{
				x.resize(1 + tf.rows() * tf.cols());
				x[0] = density;
				x.tail(x.size() - 1) = tfLinear;
			} else
			{
				x.resize(1);
				x[0] = density;
			}

			checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
				epsilon, precision, margin);
		}
	}
}

TEST_CASE("Adjoint-TFIdentity-NoTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFIdentity, false>();
}
TEST_CASE("Adjoint-TFIdentity-WithTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFIdentity, true>();
}

TEST_CASE("Adjoint-TFTexture-NoTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFTexture, false>();
}
TEST_CASE("Adjoint-TFTexture-WithTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFTexture, true>();
}

TEST_CASE("Adjoint-TFLinear-NoTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFLinear, false>();
}
TEST_CASE("Adjoint-TFLinear-WithTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFLinear, true>();
}

TEST_CASE("Adjoint-TFGauss-NoTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFGaussian, false>(1e-5, 1e-3, 1e-4);
}
TEST_CASE("Adjoint-TFGauss-WithTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFGaussian, true>(1e-5, 1e-3, 1e-3);
}

TEST_CASE("Adjoint-TFGaussLog-NoTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFGaussianLog, false>(1e-5, 1e-3, 1e-4);
}
TEST_CASE("Adjoint-TFGaussLog-WithTFDerivatives", "[adjoint]")
{
	testAdjointTF<kernel::TFGaussianLog, true>(1e-6, 1e-3, 1e-4);
}


TEST_CASE("TFGaussianLog-vs-TFGaussian", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<real_t> distr(0.01, 0.99);
	int N1 = 10; //TFs
	int N2 = 20; //densities
	for (int i1 = 0; i1 < N1; ++i1)
	{
		INFO("N1=" << i1);
		const MatrixXr tf = createTF<kernel::TFGaussian>(rnd);
		INFO("tf:\n" << tf);

		kernel::TransferFunctionEval<kernel::TFGaussian> tfEvalGauss;
		kernel::TransferFunctionEval<kernel::TFGaussianLog> tfEvalGaussLog;
		
		for (int i2 = 0; i2 < N2; ++i2)
		{
			INFO("N2=" << i2);
			real_t density = distr(rnd);
			INFO("density: " << density);

			const kernel::Tensor3Read myTFacc = accessor3D<kernel::Tensor3Read>(tf);
			real4 colorExpected = tfEvalGauss.eval(myTFacc, 0, density);
			real4 colorActualLog = tfEvalGaussLog.eval(myTFacc, 0, density);
			real4 colorActual = rexp(colorActualLog);

			REQUIRE(colorExpected.x == Approx(colorActual.x));
			REQUIRE(colorExpected.y == Approx(colorActual.y));
			REQUIRE(colorExpected.z == Approx(colorActual.z));
			REQUIRE(colorExpected.w == Approx(colorActual.w));
		}
	}
}