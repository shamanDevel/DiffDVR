#include <catch.hpp>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <torch/torch.h>

#include <renderer_utils.cuh>
#include <losses.h>
#include "check_adjoint.h"
#include "test_utils.h"

//namespace std {
//	template < class T >
//	inline std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
//	{
//		os << "[";
//		for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
//		{
//			os << " " << *ii;
//		}
//		os << " ]";
//		return os;
//	}
//}
	
TEST_CASE("LogSumExp-double", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int N=1; N<=10; ++N)
	{
		DYNAMIC_SECTION("N=" << N)
		{
			std::vector<double> values(N);
			for (int i = 0; i < N; ++i) values[i] = distr(rnd);

			const auto valuesLambda = [values](int i) {return values[i]; };
			double resultActual = kernel::logSumExp<double>(N, valuesLambda);

			//naive
			double resultExpected = 0;
			for (int i = 0; i < N; ++i) resultExpected += rexp(values[i]);
			resultExpected = rlog(resultExpected);

			REQUIRE(resultActual == Approx(resultExpected));
		}
	}
}
TEST_CASE("LogSumExp-double4", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int N = 1; N <= 10; ++N)
	{
		DYNAMIC_SECTION("N=" << N)
		{
			std::vector<double4> values(N);
			for (int i = 0; i < N; ++i) {
				values[i] = make_double4(
					distr(rnd), distr(rnd), 
					distr(rnd), distr(rnd));
			}

			const auto valuesLambda = [values](int i) {return values[i]; };
			double4 resultActual = kernel::logSumExp<double4>(N, valuesLambda);

			//naive
			double4 resultExpected = make_double4(0);
			for (int i = 0; i < N; ++i) resultExpected += rexp(values[i]);
			resultExpected = rlog(resultExpected);

			REQUIRE(resultActual.x == Approx(resultExpected.x));
			REQUIRE(resultActual.y == Approx(resultExpected.y));
			REQUIRE(resultActual.z == Approx(resultExpected.z));
			REQUIRE(resultActual.w == Approx(resultExpected.w));
		}
	}
}

TEST_CASE("LogSumExp-scaling-double", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int N = 1; N <= 10; ++N)
	{
		DYNAMIC_SECTION("N=" << N)
		{
			std::vector<double> values(N);
			std::vector<double> scaling(N);
			for (int i = 0; i < N; ++i) {
				values[i] = distr(rnd);
				scaling[i] = fabs(distr(rnd));
			}
			INFO("values: " << values);
			INFO("scaling: " << scaling);

			const auto valuesLambda = [values](int i) {return values[i]; };
			const auto scalingLambda = [scaling](int i) {return scaling[i]; };
			double resultActual = kernel::logSumExpWithScaling<double>(
				N, valuesLambda, scalingLambda);

			//naive
			double resultExpected = 0;
			for (int i = 0; i < N; ++i) 
				resultExpected += scaling[i] * rexp(values[i]);
			resultExpected = rlog(resultExpected);

			REQUIRE(resultActual == Approx(resultExpected));
		}
	}
}
TEST_CASE("LogSumExp-scaling-double4", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int N = 1; N <= 10; ++N)
	{
		DYNAMIC_SECTION("N=" << N)
		{
			std::vector<double4> values(N);
			std::vector<double> scaling(N);
			for (int i = 0; i < N; ++i) {
				values[i] = make_double4(
					distr(rnd), distr(rnd),
					distr(rnd), distr(rnd));
				scaling[i] = fabs(distr(rnd));
			}

			const auto valuesLambda = [values](int i) {return values[i]; };
			const auto scalingLambda = [scaling](int i) {return scaling[i]; };
			double4 resultActual = kernel::logSumExpWithScaling<double4>(
				N, valuesLambda, scalingLambda);

			//naive
			double4 resultExpected = make_double4(0);
			for (int i = 0; i < N; ++i) 
				resultExpected += scaling[i] * rexp(values[i]);
			resultExpected = rlog(resultExpected);

			REQUIRE(resultActual.x == Approx(resultExpected.x));
			REQUIRE(resultActual.y == Approx(resultExpected.y));
			REQUIRE(resultActual.z == Approx(resultExpected.z));
			REQUIRE(resultActual.w == Approx(resultExpected.w));
		}
	}
}

TEST_CASE("LogMSE-double", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int i = 1; i <= 50; ++i)
	{
		INFO("i: " << i);
		double logX = distr(rnd);
		double logY = distr(rnd);
		double x = exp(logX), y = exp(logY);
		INFO("x=" << x << ", y=" << y);

		double mseExpected = (x - y) * (x - y);

		double mseActualLog = kernel::logMSE(logX, logY);
		double mseActual = exp(mseActualLog);

		REQUIRE(mseActual == Approx(mseExpected));
	}
}

TEST_CASE("LogL1-double", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);

	for (int i = 1; i <= 50; ++i)
	{
		INFO("i: " << i);
		double logX = distr(rnd);
		double logY = distr(rnd);
		double x = exp(logX), y = exp(logY);
		INFO("x=" << x << ", y=" << y);

		double l1Expected = fabs(x - y);

		double l1ActualLog = kernel::logL1(logX, logY);
		double l1Actual = exp(l1ActualLog);

		REQUIRE(l1Actual == Approx(l1Expected));
	}
}


TEST_CASE("Adjoint-LogMSE", "[adjoint]")
{
	typedef empty TmpStorage_t;
	typedef Eigen::VectorXd Vector_t;

	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		double logX = x[0], logY = x[1];
		double res = kernel::logMSE(logX, logY);
		Vector_t rese(1);
		rese[0] = res;
		return rese;
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		double logX = x[0], logY = x[1];
		double adjOut = g[0];
		double adjLogX, adjLogY;
		kernel::adjLogMSE(logX, logY, adjOut, adjLogX, adjLogY);
		z[0] = adjLogX;
		z[1] = adjLogY;
	};

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);
	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		Vector_t x(2);
		for (int j = 0; j < 2; ++j) x[j] = distr(rnd);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-5, 1e-5, 1e-6);
	}
}

TEST_CASE("Adjoint-Log1", "[adjoint]")
{
	typedef empty TmpStorage_t;
	typedef Eigen::VectorXd Vector_t;

	auto forward = [](const Vector_t& x, TmpStorage_t* tmp) -> Vector_t
	{
		double logX = x[0], logY = x[1];
		double res = kernel::logL1(logX, logY);
		Vector_t rese(1);
		rese[0] = res;
		return rese;
	};
	auto adjoint = [](const Vector_t& x, const Vector_t& e, const Vector_t& g,
		Vector_t& z, const TmpStorage_t& tmp)
	{
		double logX = x[0], logY = x[1];
		double adjOut = g[0];
		double adjLogX, adjLogY;
		kernel::adjLogL1(logX, logY, adjOut, adjLogX, adjLogY);
		z[0] = adjLogX;
		z[1] = adjLogY;
	};

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distr(-5, +5);
	int N = 20;
	for (int i = 0; i < N; ++i)
	{
		INFO("N=" << i);
		Vector_t x(2);
		for (int j = 0; j < 2; ++j) x[j] = distr(rnd);

		checkAdjoint<Vector_t, TmpStorage_t>(x, forward, adjoint,
			1e-5, 1e-5, 1e-6);
	}
}

TEST_CASE("Adjoint-LogMSE-Full", "[adjoint]")
{
	torch::Tensor logX = torch::randn({ 4,5 },
		at::TensorOptions().dtype(c10::kDouble).device(c10::kCUDA))
		.requires_grad_(true);
	torch::Tensor logY = torch::randn({ 4,5 },
		at::TensorOptions().dtype(c10::kDouble).device(c10::kCUDA))
		.requires_grad_(true);

	torch::Tensor out = renderer::logMSE(logX, logY);
	torch::Tensor grad_out = torch::rand_like(out);
	auto grad_inputs = torch::autograd::grad(
		{ out }, { logX, logY }, { grad_out });
	torch::Tensor grad_logX = grad_inputs[0];
	torch::Tensor grad_logY = grad_inputs[1];

	std::cout << "grad_logX:\n" << grad_logX << std::endl;
}

