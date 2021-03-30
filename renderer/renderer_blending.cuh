#pragma once

#include "renderer_settings.cuh"
#include "forward_vector.h"

namespace kernel
{
	/**
	 * \brief Performs blending of the current color+absorption onto the accumulator.
	 * \tparam blendMode the blend mode
	 */
	template<BlendMode blendMode>
	struct Blending
	{
		/**
		 * \brief Blends the new value 'current' into the
		 *   accumulator 'acc' and returns the new accumulated value.
		 * \param acc the accumulator (color rgb, alpha)
		 * \param current the current value (color rgb, absorption)
		 * \param stepsize the stepsize
		 * \return the new accumulated value (color rgb, alpha)
		 */
		static __host__ __device__ __inline__ real4 blend(
			const real4& acc, const real4& current, real_t stepsize);

		/**
		 * \brief Adjoint code of the blending.
		 * Returns the state of the accumulator *before* the current cell
		 * would have been blended.
		 */
		static __host__ __device__ __inline__ real4 adjoint(
			const real4& acc, const real4& current, real_t stepsize,
			const real4& adj_Acc, real4& adj_current, real_t& adj_Stepsize);
	};

	template<>
	struct Blending<BlendBeerLambert>
	{
		//forward-derivatives
		template<typename acc_t, typename current_t, typename step_t>
		static __host__ __device__ __inline__ acc_t blend(
			const acc_t& acc, const current_t& current, step_t stepsize)
		{
			using namespace cudAD;
			auto alpha = 1 - exp(-getW(current) * stepsize);
			auto colorOutR = getX(acc) + (real_t(1) - getW(acc)) * getX(current) * alpha;
			auto colorOutG = getY(acc) + (real_t(1) - getW(acc)) * getY(current) * alpha;
			auto colorOutB = getZ(acc) + (real_t(1) - getW(acc)) * getZ(current) * alpha;
			auto alphaOut = getW(acc) + (real_t(1) - getW(acc)) * alpha;
			return make_real4(colorOutR, colorOutG, colorOutB, alphaOut);
		}

		//simple forward mode, no derivatives
		static __host__ __device__ __inline__ real4 blend(
			const real4& acc, const real4& current, real_t stepsize)
		{
			real_t currentAlpha = 1 - real_t(std::exp(-current.w * stepsize));
			real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * currentAlpha;
			real_t alphaOut = acc.w + (1 - acc.w) * currentAlpha;
			return make_real4(colorOut, alphaOut);
		}

		static __host__ __device__ __inline__ real4 adjoint(
			const real4& acc_output, const real4& current, real_t stepsize,
			const real4& adj_accOut, real4& adj_accIn, real4& adj_current, real_t& adj_stepsize)
		{
			const real3 currentColor = make_real3(current);
			const real3 accColor = make_real3(acc_output);
			const real3 adj_accOutColor = make_real3(adj_accOut);
			
			real_t currentAlpha = 1 - real_t(std::exp(-current.w * stepsize));
			//real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * alpha;
			//real_t alphaOut = acc.w + (1 - acc.w) * alpha;

			//invert to reconstruct the accumulator before blending
			const real_t alphaIn = (currentAlpha - acc_output.w) / (currentAlpha - 1);
			const real3 colorIn = accColor - (1 - alphaIn) * currentColor * currentAlpha;

			//adjoint code

			//real_t alphaOut = acc.w + (1 - acc.w) * alpha;
			real_t adj_alpha = adj_accOut.w * (1 - alphaIn);
			real_t adj_accInAlpha = adj_accOut.w * (1 - currentAlpha);
			
			//real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * alpha;
			adj_alpha += dot(adj_accOutColor, currentColor - currentColor * alphaIn);
			adj_accInAlpha += dot(adj_accOutColor, - currentColor * currentAlpha);
			real3 adj_currentColor = adj_accOutColor * (currentAlpha * (1 - alphaIn));
			real3 adj_accInColor = adj_accOutColor;

			//real_t currentAlpha = 1 - real_t(std::exp(-current.w * stepsize));
			real_t adj_currentW = adj_alpha * stepsize * std::exp(-current.w * stepsize);
			adj_stepsize = adj_alpha * current.w * std::exp(-current.w * stepsize);

			adj_current = make_real4(adj_currentColor, adj_currentW);
			adj_accIn = make_real4(adj_accInColor, adj_accInAlpha);
			return make_real4(colorIn, alphaIn);
		}
	};

	template<>
	struct Blending<BlendAlpha>
	{
		//forward-derivatives
		template<typename acc_t, typename current_t, typename step_t>
		static __host__ __device__ __inline__ acc_t blend(
			const acc_t& acc, const current_t& current, step_t stepsize)
		{
			using namespace cudAD;
			auto alpha = fmin(real_t(1), getW(current) * stepsize);
			auto colorOutR = getX(acc) + (real_t(1) - getW(acc)) * getX(current) * alpha;
			auto colorOutG = getY(acc) + (real_t(1) - getW(acc)) * getY(current) * alpha;
			auto colorOutB = getZ(acc) + (real_t(1) - getW(acc)) * getZ(current) * alpha;
			auto alphaOut = getW(acc) + (real_t(1) - getW(acc)) * alpha;
			return make_real4(colorOutR, colorOutG, colorOutB, alphaOut);
		}
		
		static __host__ __device__ __inline__ real4 blend(
			const real4& acc, const real4& current, real_t stepsize)
		{
			real_t currentAlpha = rmin(real_t(1), current.w * stepsize);
			real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * currentAlpha;
			real_t alphaOut = acc.w + (1 - acc.w) * currentAlpha;
			return make_real4(colorOut, alphaOut);
		}

		static constexpr real_t EPS = real_t(1e-5);
		static __host__ __device__ __inline__ real4 adjoint(
			const real4& acc_output, const real4& current, real_t stepsize,
			const real4& adj_accOut, real4& adj_accIn, real4& adj_current, real_t& adj_stepsize)
		{
			const real3 currentColor = make_real3(current);
			const real3 accColor = make_real3(acc_output);
			const real3 adj_accOutColor = make_real3(adj_accOut);

			real_t currentAlpha = rmin(real_t(1-EPS), current.w * stepsize);
			//real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * currentAlpha;
			//real_t alphaOut = acc.w + (1 - acc.w) * currentAlpha;

			//invert to reconstruct the accumulator before blending
			const real_t alphaIn = (currentAlpha - acc_output.w) / (currentAlpha - 1);
			const real3 colorIn = accColor - (1 - alphaIn) * currentColor * currentAlpha;

			//adjoint code

			//real_t alphaOut = acc.w + (1 - acc.w) * alpha;
			real_t adj_alpha = adj_accOut.w * (1 - alphaIn);
			real_t adj_accInAlpha = adj_accOut.w * (1 - currentAlpha);
			
			//real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * alpha;
			adj_alpha += dot(adj_accOutColor, currentColor - currentColor * alphaIn);
			adj_accInAlpha += dot(adj_accOutColor, -currentColor * currentAlpha);
			real3 adj_currentColor = adj_accOutColor * (currentAlpha * (1 - alphaIn));
			real3 adj_accInColor = adj_accOutColor;

			//real_t currentAlpha = rmin(real_t(1-EPS), current.w * stepsize);
			real_t adj_currentW;
			adj_stepsize = (current.w * stepsize < 1) ? (adj_alpha * current.w) : 0;
			adj_currentW = (current.w * stepsize < 1) ? (adj_alpha * stepsize) : 0;

			adj_current = make_real4(adj_currentColor, adj_currentW);
			adj_accIn = make_real4(adj_accInColor, adj_accInAlpha);
			return make_real4(colorIn, alphaIn);
		}
	};
}
