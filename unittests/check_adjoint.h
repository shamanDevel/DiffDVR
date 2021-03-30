#pragma once

#include <catch.hpp>

#include "renderer_settings.cuh"

//Forward function: e <- f(x)
//  x: input state
//  e: output state
//if tmp!=null, this forward pass is for the adjoint computation and you can save there some data that will be sent to the adjoint
//if tmp==null, this forward pass is used to compute the gradient numerically (no adjoint call)
template<typename Vector_t, typename TmpStorage_t>
using ForwardFunction = std::function<Vector_t(const Vector_t& x, TmpStorage_t* tmp)>;

//Adjoint function: f'(x, e, g, z'&)
//  x: input state
//  e: output state from the forward pass
//  g: gradient of the output energy with respect to the output (dJ/de)
//  z': output partial gradient. Let A=dEdx, F=dEdz. Solve Ax'=g. Return x'^T * F. Has to be applied additively
//tmp: additional storage from the forward pass
template<typename Vector_t, typename TmpStorage_t>
using AdjointFunction = std::function<void(const Vector_t& x, const Vector_t& e, const Vector_t& g, Vector_t& z, const TmpStorage_t& tmp)>;

/**
 * \brief Checks the adjoint code for correctness.
 * \tparam Vector_t the vector type
 * \tparam TmpStorage_t temporal storage
 * \param x the input vector
 * \param forward the forward function
 * \param adjoint the adjoint function
 * \param epsilon the distance for the numerical differences
 * \param precision the allowed relative error
 * \param margin the allowed absolute error
 */
template<typename Vector_t, typename TmpStorage_t>
void checkAdjoint(const Vector_t& x, const ForwardFunction<Vector_t, TmpStorage_t>& forward, 
    const AdjointFunction<Vector_t, TmpStorage_t>& adjoint,
    real_t epsilon = 1e-5, real_t precision = 1e-6, real_t margin = 1e-10)
{
    int n = x.size();

    INFO("initial x: " << x.transpose());

    TmpStorage_t mem;
    const Vector_t ref = forward(x, &mem);
    int m = ref.size();
    INFO("reference result e: " << ref.transpose());
    int numDirections = 2 * n;

    for (int trial = 0; trial < numDirections + m; ++trial)
    {
        INFO("trial " << trial << " of " << m << "+" << numDirections);
        Vector_t adjE;
        if (trial < m) {
            adjE = Vector_t::Zero(m); adjE[trial] = 1;
        }
        else
            adjE = Vector_t::Random(m);
        INFO("adjE / search direction: " << adjE.transpose());

        Vector_t adjZ = Vector_t::LinSpaced(n, 42, 42 + n - 1) * 0.0;
        Vector_t adjZCopy = adjZ;

        //evaluate adjoint
        adjoint(x, ref, adjE, adjZ, mem);
        INFO("backward: adjZ=" << (adjZ - adjZCopy).transpose())

        //check against numerical value
        for (int i = 0; i < n; ++i)
        {
            INFO("x': i=" << i);
            Vector_t x2 = x;
            x2[i] += epsilon;
            CHECK(adjZ[i] - adjZCopy[i] == Approx(((forward(x2, nullptr) - ref) / epsilon).dot(adjE)).epsilon(precision).margin(margin));
        }
        //for (int i = 0; i<n; ++i)
        //{
        //    INFO("x': i=" << i);
        //    VectorX x2 = x;
        //    x2[i] += epsilon;
        //    REQUIRE(adjX[i] == Approx(((forward(x2, z, nullptr) - ref) / epsilon).dot(adjXCopy)).epsilon(precision).margin(margin));
        //}
    }
}

template<typename Vector_t>
using ScalarFunction = std::function<real_t(const Vector_t&)>;

template<typename Vector_t>
using ScalarFunctionDerivative = std::function<Vector_t(const Vector_t&)>;

template<typename Vector_t>
inline void checkDerivative(const Vector_t& x, const ScalarFunction<Vector_t>& fun, 
    const ScalarFunctionDerivative<Vector_t>& derivative,
    real_t epsilon = 1e-5, real_t precision = 1e-4)
{
    int n = x.size();

    real_t ref = fun(x);
    Vector_t gradAna = derivative(x);

    for (int i = 0; i < n; ++i)
    {
        Vector_t x2 = x;
        x2[i] += epsilon;
        real_t ref2 = fun(x2);
        CAPTURE(i);
        REQUIRE(gradAna[i] == Approx((ref2 - ref) / epsilon).epsilon(precision));
    }
}

