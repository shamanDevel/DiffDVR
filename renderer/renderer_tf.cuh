#pragma once

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_adjoint.cuh"

#ifdef __CUDACC__
#include "cooperative_groups.cuh"
#endif
//#include <cooperative_groups/reduce.h>

namespace kernel
{

	/**
	 * \brief Transfer function evaluation.
	 * Maps the density in [0,1] to a real4 of red, green, blue, absorption.
	 * \tparam tfMode the transfer function mode
	 */
	template<TFMode tfMode>
	struct TransferFunctionEval
	{
		/**
		 * \brief Evaluates the transfer function.
		 * \param tf the transfer function of shape B*R*C
		 *    where C depends on the type of transfer function,
		 *    R is the result and B the batch
		 * \param batch the batch index
		 * \param density the density in [0,1]
		 * \return the color as rgb in xyz and the absorption in w.
		 */
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const;

		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> hasTFDerivative) const;
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> hasTFDerivative) const;

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, real_t density,
			const real4& adj_color, real_t& adj_density, BTensor3RW& adj_tf,
			real_t* sharedData) const;
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const;
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const;
	};

	template<>
	struct TransferFunctionEval<TFIdentity>
	{
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const
		{
			density = clamp(density, real_t(0), real_t(1));
			const real_t scaleAbsorption = tf[batch][0][0];
			const real_t scaleColor = tf[batch][0][1];
			return make_real4(
				density * scaleColor, //red
				density * scaleColor, //green
				density * scaleColor, //blue
				density * scaleAbsorption); //absorption
		}

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, real_t density,
			const real4& adj_color, real_t& adj_density, 
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			real_t densityClamped = clamp(density, real_t(0), real_t(1));
			const real_t scaleAbsorption = tf[batch][0][0];
			const real_t scaleColor = tf[batch][0][1];

			real_t adj_densityClamped =
				dot(make_real3(adj_color), make_real3(scaleColor)) +
				adj_color.w * scaleAbsorption;
			if constexpr (HasTFDerivative)
			{
				real_t adj_scaleColor = dot(make_real3(adj_color), make_real3(densityClamped));
				real_t adj_scaleAbsorption = adj_color.w * densityClamped;
				if constexpr(DelayedAccumulation)
				{
					sharedData[0] += adj_scaleAbsorption;
					sharedData[1] += adj_scaleColor;
				}
				else {
					kernel::atomicAdd(&adj_tf[batch][0][0], adj_scaleAbsorption);
					kernel::atomicAdd(&adj_tf[batch][0][1], adj_scaleColor);
				}
			}

			adj_density = (density > 0 && density < 1) ? adj_densityClamped : 0;
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
			//TODO: optimize via warp reductions?
			kernel::atomicAdd(&adj_tf[batch][0][0], sharedData[0]);
			kernel::atomicAdd(&adj_tf[batch][0][1], sharedData[1]);
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 2;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}
		
		template<int D, typename density_t>
		__host__ __device__ __inline__ auto evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			density = clamp(density, real_t(0), real_t(1));
			const real_t scaleAbsorption = tf[batch][0][0];
			const real_t scaleColor = tf[batch][0][1];
			return make_real4(
				density * scaleColor, //red
				density * scaleColor, //green
				density * scaleColor, //blue
				density * scaleAbsorption); //absorption
		}
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			density = clamp(density, real_t(0), real_t(1));
			const fvar<real_t, D> scaleAbsorption = fvar<real_t, D>::input(
				tf[batch][0][0], d_tf[batch][0][0]);
			const fvar<real_t, D> scaleColor = fvar<real_t, D>::input(
				tf[batch][0][1], d_tf[batch][0][1]);
			return make_real4(
				density * scaleColor, //red
				density * scaleColor, //green
				density * scaleColor, //blue
				density * scaleAbsorption); //absorption
		}
	};

	template<>
	struct TransferFunctionEval<TFTexture>
	{
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const
		{
			const int R = tf.size(1);
			//texture linear interpolation
			const real_t d = density * R - real_t(0.5);
			const int di = int(floorf(d));
			const real_t df = d - di;
			const real4 val0 = fetchReal4(tf, batch, clamp(di, 0, R - 1));
			const real4 val1 = fetchReal4(tf, batch, clamp(di+1, 0, R - 1));
			return lerp(val0, val1, df);
		}

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, real_t density,
			const real4& adj_color, real_t& adj_density, 
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = tf.size(1);
			//texture linear interpolation
			const real_t d = density * R - real_t(0.5);
			const int di = int(floorf(d));
			const real_t df = d - di;
			const int idx0 = clamp(di, 0, R - 1);
			const int idx1 = clamp(di + 1, 0, R - 1);
			const real4 val0 = fetchReal4(tf, batch, idx0);
			const real4 val1 = fetchReal4(tf, batch, idx1);
			//output = lerp(val0, val1, df);

			const real_t adj_df = dot(adj_color, val1 - val0);
			if constexpr (HasTFDerivative)
			{
				const real4 adj_val0 = adj_color * (1 - df);
				const real4 adj_val1 = adj_color * df;

				if constexpr (DelayedAccumulation) {
					sharedData[4 * idx0 + 0] += adj_val0.x;
					sharedData[4 * idx0 + 1] += adj_val0.y;
					sharedData[4 * idx0 + 2] += adj_val0.z;
					sharedData[4 * idx0 + 3] += adj_val0.w;

					sharedData[4 * idx1 + 0] += adj_val1.x;
					sharedData[4 * idx1 + 1] += adj_val1.y;
					sharedData[4 * idx1 + 2] += adj_val1.z;
					sharedData[4 * idx1 + 3] += adj_val1.w;
				}
				else {
					kernel::atomicAdd(&adj_tf[batch][idx0][0], adj_val0.x);
					kernel::atomicAdd(&adj_tf[batch][idx0][1], adj_val0.y);
					kernel::atomicAdd(&adj_tf[batch][idx0][2], adj_val0.z);
					kernel::atomicAdd(&adj_tf[batch][idx0][3], adj_val0.w);

					kernel::atomicAdd(&adj_tf[batch][idx1][0], adj_val1.x);
					kernel::atomicAdd(&adj_tf[batch][idx1][1], adj_val1.y);
					kernel::atomicAdd(&adj_tf[batch][idx1][2], adj_val1.z);
					kernel::atomicAdd(&adj_tf[batch][idx1][3], adj_val1.w);
				}
			}
			adj_density = adj_df * R;
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 4;
#ifdef __CUDA_ARCH__
			coalesced_group active = coalesced_threads();
			const int tia = active.thread_rank();
#endif
			for (int r=0; r<R; ++r) for (int c=0; c<C; ++c)
			{
				real_t val = sharedData[4 * r + c];
#ifdef __CUDA_ARCH__
				real_t reduction = reduce(active, val, plus<real_t>());
				active.sync();
				if (tia==0)
				{ //leader accumulates
					kernel::atomicAdd(&adj_tf[batch][r][c], reduction);
				}
#else
				//fallback accumulation
				kernel::atomicAdd(&adj_tf[batch][r][c], val);
#endif
			}
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 4;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}
		
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			const auto d = density * real_t(R) - real_t(0.5);
			const int di = int(floorf(static_cast<real_t>(d)));
			const auto df = d - di;
			const real4 val0 = fetchReal4(tf, batch, clamp(di, 0, R - 1));
			const real4 val1 = fetchReal4(tf, batch, clamp(di + 1, 0, R - 1));
			return lerp(val0, val1, broadcast4(df));
		}
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			const auto d = density * real_t(R) - real_t(0.5);
			const int di = int(floorf(static_cast<real_t>(d)));
			const auto df = d - di;
			const fvar<real4, D> val0 = make_real4in<D>(
				fetchReal4(tf, batch, clamp(di, 0, R - 1)),
				fetchInt4(d_tf, batch, clamp(di, 0, R - 1)));
			const fvar<real4, D> val1 = make_real4in<D>(
				fetchReal4(tf, batch, clamp(di+1, 0, R - 1)),
				fetchInt4(d_tf, batch, clamp(di+1, 0, R - 1)));
			return lerp(val0, val1, broadcast4(df));
		}
	};

	template<>
	struct TransferFunctionEval<TFLinear>
	{
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const
		{
			const int R = tf.size(1);
			//find control point interval
			int i;
			for (i = 0; i < R - 2; ++i)
				if (tf[batch][i+1][4] > density) break;
			//fetch values
			const real4 val0 = fetchReal4(tf, batch, i);
			const real4 val1 = fetchReal4(tf, batch, i+1);
			const real_t pos0 = tf[batch][i][4];
			const real_t pos1 = tf[batch][i + 1][4];
			//linear interpolation
			//density<=pos0 -> pos0 ELSE density>pos1 -> pos1 ELSE density
			density = clamp(density, pos0, pos1);
			const real_t frac = (density - pos0) / (pos1 - pos0);
			//val0 + frac * (val1 - val0) = (1-frac)*val0 + frac*val1
			return lerp(val0, val1, frac);
		}

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, const real_t density,
			const real4& adj_color, real_t& adj_density, 
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			//FORWARD
			
			const int R = tf.size(1);
			//find control point interval
			int i;
			for (i = 0; i < R - 2; ++i)
				if (tf[batch][i + 1][4] > density) break;
			//fetch values
			const real4 val0 = fetchReal4(tf, batch, i);
			const real4 val1 = fetchReal4(tf, batch, i + 1);
			const real_t pos0 = tf[batch][i][4];
			const real_t pos1 = tf[batch][i + 1][4];
			//linear interpolation
			const real_t density2 = clamp(density, pos0, pos1);
			const real_t frac = (density2 - pos0) / (pos1 - pos0);
			//output = lerp(val0, val1, frac) = val0 + frac * (val1 - val0);

			//ADJOINT

			//adj: output = lerp(val0, val1, frac);
			const real_t adj_frac = dot(adj_color, val1 - val0);
			const real4 adj_val0 = adj_color * (1 - frac);
			const real4 adj_val1 = adj_color * frac;
			//adj: frac = (density2 - pos0) / (pos1 - pos0);
			const real_t tmp_denom = (pos1 - pos0) * (pos1 - pos0);
			real_t adj_pos0 = adj_frac * (density2 - pos1) / tmp_denom;
			real_t adj_pos1 = adj_frac * (pos0 - density2) / tmp_denom;
			const real_t adj_density2 = adj_frac / (pos1 - pos0);
			//adj: density2 = clamp(density, pos0, pos1);
#if 1
			adj_density = adj_density2;
#else
			adj_density = 0;
			if (density <= pos0)
			{
				//adj: density2 = pos0
				adj_pos0 += adj_density2;
			} else if (density > pos1)
			{
				//adj: density2 = pos1
				adj_pos1 += adj_density2;
			} else
			{
				//adj: density2 = density
				adj_density = density2;
			}
#endif
			
			if constexpr (HasTFDerivative)
			{

				if (DelayedAccumulation)
				{
					sharedData[5 * i + 0] += adj_val0.x;
					sharedData[5 * i + 1] += adj_val0.y;
					sharedData[5 * i + 2] += adj_val0.z;
					sharedData[5 * i + 3] += adj_val0.w;
					sharedData[5 * i + 4] += adj_pos0;
					int j = i + 1;
					sharedData[5 * j + 0] += adj_val1.x;
					sharedData[5 * j + 1] += adj_val1.y;
					sharedData[5 * j + 2] += adj_val1.z;
					sharedData[5 * j + 3] += adj_val1.w;
					sharedData[5 * j + 4] += adj_pos1;
				}
				else {
					kernel::atomicAdd(&adj_tf[batch][i][0], adj_val0.x);
					kernel::atomicAdd(&adj_tf[batch][i][1], adj_val0.y);
					kernel::atomicAdd(&adj_tf[batch][i][2], adj_val0.z);
					kernel::atomicAdd(&adj_tf[batch][i][3], adj_val0.w);
					kernel::atomicAdd(&adj_tf[batch][i][4], adj_pos0);

					kernel::atomicAdd(&adj_tf[batch][i + 1][0], adj_val1.x);
					kernel::atomicAdd(&adj_tf[batch][i + 1][1], adj_val1.y);
					kernel::atomicAdd(&adj_tf[batch][i + 1][2], adj_val1.z);
					kernel::atomicAdd(&adj_tf[batch][i + 1][3], adj_val1.w);
					kernel::atomicAdd(&adj_tf[batch][i + 1][4], adj_pos1);
				}
			}
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 5;
#ifdef __CUDA_ARCH__
			coalesced_group active = coalesced_threads();
			const int tia = active.thread_rank();
#endif
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
			{
				real_t val = sharedData[5 * r + c];
#ifdef __CUDA_ARCH__
				real_t reduction = reduce(active, val, plus<real_t>());
				active.sync();
				if (tia == 0)
				{ //leader accumulates
					kernel::atomicAdd(&adj_tf[batch][r][c], reduction);
				}
#else
				//fallback accumulation
				kernel::atomicAdd(&adj_tf[batch][r][c], val);
#endif
			}
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 5;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}


		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			//find control point interval
			int i;
			for (i = 0; i < R - 2; ++i)
				if (tf[batch][i + 1][4] > static_cast<real_t>(density)) break;
			//fetch values
			const real4 val0 = fetchReal4(tf, batch, i);
			const real4 val1 = fetchReal4(tf, batch, i + 1);
			const real_t pos0 = tf[batch][i][4];
			const real_t pos1 = tf[batch][i + 1][4];
			//linear interpolation
			const auto density2 = clamp(density, pos0, pos1);
			const auto frac = (density2 - pos0) / (pos1 - pos0);
			return lerp(val0, val1, broadcast4(frac));
		}
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			//find control point interval
			int i;
			for (i = 0; i < R - 2; ++i)
				if (tf[batch][i + 1][4] > static_cast<real_t>(density)) break;
			//fetch values
			const fvar<real4, D> val0 = make_real4in<D>(
				fetchReal4(tf, batch, i),
				fetchInt4(d_tf, batch, i));
			const fvar<real4, D> val1 = make_real4in<D>(
				fetchReal4(tf, batch, i+1),
				fetchInt4(d_tf, batch, i+1));
			const fvar<real_t, D> pos0 = fvar<real_t, D>::input(
				tf[batch][i][4], d_tf[batch][i][4]);
			const fvar<real_t, D> pos1 = fvar<real_t, D>::input(
				tf[batch][i+1][4], d_tf[batch][i+1][4]);
			//linear interpolation
			const auto density2 = clamp(density, pos0, pos1);
			const auto frac = (density2 - pos0) / (pos1 - pos0);
			return lerp(val0, val1, broadcast4(frac));
		}
	};

	template<>
	struct TransferFunctionEval<TFGaussian>
	{
		template<typename D, typename M, typename S>
		static __host__ __device__ __inline__ auto normal(const D& d, const M& mu, const S& sigma)
		{
			using namespace cudAD;
			return exp(-(d - mu) * (d - mu) / (2 * sigma * sigma));
		}
		
		/**
		 * \brief Evaluates the transfer function.
		 * \param tf the transfer function of shape B*R*C
		 *    where C depends on the type of transfer function,
		 *    R is the result and B the batch
		 * \param batch the batch index
		 * \param density the density in [0,1]
		 * \return the color as rgb in xyz and the absorption in w.
		 */
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const
		{
			const int R = tf.size(1);
			real4 c = make_real4(0);
			for (int i=0; i<R; ++i)
			{
				real4 ci = fetchReal4(tf, batch, i);
				real_t ni = normal(density, tf[batch][i][4], tf[batch][i][5]);
				c += ci * ni;
			}
			return c;
		}

		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> hasTFDerivative) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			fvar<real4, D> c = fvar<real4, D>::constant(make_real4(0));
			for (int i = 0; i < R; ++i)
			{
				real4 ci = fetchReal4(tf, batch, i);
				auto ni = normal(density, tf[batch][i][4], tf[batch][i][5]);
				c += fvar<real4, D>::constant(ci) * ni;
			}
			return c;
		}
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> hasTFDerivative) const
		{
			using namespace cudAD;
			const int R = tf.size(1);
			fvar<real4, D> c = fvar<real4, D>::constant(make_real4(0));
			for (int i = 0; i < R; ++i)
			{
				const fvar<real4, D> ci = make_real4in<D>(
					fetchReal4(tf, batch, i),
					fetchInt4(d_tf, batch, i));
				const fvar<real_t, D> mu = fvar<real_t, D>::input(
					tf[batch][i][4], d_tf[batch][i][4]);
				const fvar<real_t, D> sigma = fvar<real_t, D>::input(
					tf[batch][i][5], d_tf[batch][i][5]);
				auto ni = normal(density, mu, sigma);
				c += ci * ni;
			}
			return c;
		}

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, const real_t density,
			const real4& adj_color, real_t& adj_density,
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = tf.size(1);
			adj_density = 0;

			for (int i = 0; i < R; ++i)
			{
				//forward
				const real4 ci = fetchReal4(tf, batch, i);
				const real_t mu = tf[batch][i][4];
				const real_t sigma = tf[batch][i][5];
				const real_t sigma2 = sigma * sigma;
				const real_t ni = normal(density, mu, sigma);
				//c += ci * ni;

				//ADJOINT: density
				//printf("mu=%.7f, sigma=%.7f, ni=%.7f\n", mu, sigma, ni);
				real_t f1 = (mu - density) / sigma2 * ni;
				adj_density += dot(adj_color, ci * f1);

				//ADJOINT: tf
				if constexpr(HasTFDerivative)
				{
					real4 adj_ci = adj_color * ni;
					real_t adj_mu = dot(adj_color, ci * -f1);
					const float f2 = f1 * (mu - density) / sigma;
					real_t adj_sigma = dot(adj_color, ci * f2);

					if constexpr (DelayedAccumulation)
					{
						sharedData[6 * i + 0] += adj_ci.x;
						sharedData[6 * i + 1] += adj_ci.y;
						sharedData[6 * i + 2] += adj_ci.z;
						sharedData[6 * i + 3] += adj_ci.w;
						sharedData[6 * i + 4] += adj_mu;
						sharedData[6 * i + 5] += adj_sigma;
					}
					else
					{
						kernel::atomicAdd(&adj_tf[batch][i][0], adj_ci.x);
						kernel::atomicAdd(&adj_tf[batch][i][1], adj_ci.y);
						kernel::atomicAdd(&adj_tf[batch][i][2], adj_ci.z);
						kernel::atomicAdd(&adj_tf[batch][i][3], adj_ci.w);
						kernel::atomicAdd(&adj_tf[batch][i][4], adj_mu);
						kernel::atomicAdd(&adj_tf[batch][i][5], adj_sigma);
					}
				}
			}
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 6;
#ifdef __CUDA_ARCH__
			coalesced_group active = coalesced_threads();
			const int tia = active.thread_rank();
#endif
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
			{
				real_t val = sharedData[C * r + c];
#ifdef __CUDA_ARCH__
				real_t reduction = reduce(active, val, plus<real_t>());
				active.sync();
				if (tia == 0)
				{ //leader accumulates
					kernel::atomicAdd(&adj_tf[batch][r][c], reduction);
				}
#else
				//fallback accumulation
				kernel::atomicAdd(&adj_tf[batch][r][c], val);
#endif
			}
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 6;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}
	};




	template<>
	struct TransferFunctionEval<TFGaussianLog>
	{
		template<typename D, typename M, typename S>
		static __host__ __device__ __inline__ auto logNormal(const D& d, const M& mu, const S& sigma)
		{
			using namespace cudAD;
			return -(d - mu) * (d - mu) / (2 * sigma * sigma);
		}
		
		template<typename D, typename M, typename S>
		static __host__ __device__ __inline__ auto normal(const D& d, const M& mu, const S& sigma)
		{
			using namespace cudAD;
			return exp(logNormal(d, mu, sigma));
		}

		/**
		 * \brief Evaluates the transfer function.
		 * \param tf the transfer function of shape B*R*C
		 *    where C depends on the type of transfer function,
		 *    R is the result and B the batch
		 * \param batch the batch index
		 * \param density the density in [0,1]
		 * \return the color as rgb in xyz and the absorption in w.
		 */
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real_t density) const
		{
			const int R = tf.size(1);
			const auto values = [&](int i)
			{
				real4 ci = fetchReal4(tf, batch, i);
				real_t log_ni = logNormal(density, tf[batch][i][4], tf[batch][i][5]);
				return rlog(ci) + log_ni;
			};
			return logSumExp<real4>(R, values);
		}

		//forward derivatives not supported
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> hasTFDerivative) const = delete;
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> hasTFDerivative) const = delete;

		
		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, const real_t density,
			const real4& adj_color, real_t& adj_density,
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = tf.size(1);
			adj_density = 0;

			// out = LogSumExp_i^R (t_i)
			//where
			// t_i = log(c_i) - (x-mu_i)^2/(sigma_i)^2
			for (int i = 0; i < R; ++i)
			{
				//fetch
				const real4 ci = fetchReal4(tf, batch, i);
				const real4 log_ci = rlog(ci);
				const real_t mu = tf[batch][i][4];
				const real_t sigma = tf[batch][i][5];
				const real_t sigma2 = sigma * sigma;
				const real_t log_ni = logNormal(density, mu, sigma);

				//cache t_i
				real4 ti = log_ci + log_ni;

				//compute adj_ti
				real4 denom = make_real4(1);
				for (int j=0; j<R; ++j)
				{
					if (j!=i) //j==i -> exp(tj-ti)=1
					{
						real4 log_cj = rlog(fetchReal4(tf, batch, j));
						const real_t muj = tf[batch][j][4];
						const real_t sigmaj = tf[batch][j][5];
						const real_t log_nj = logNormal(density, muj, sigmaj);
						real4 tj = log_cj + log_nj;
						denom += rexp(tj - ti);
					}
				}
				real4 adj_ti = adj_color / denom;

				//ADJOINT: density
				//printf("mu=%.7f, sigma=%.7f, ni=%.7f\n", mu, sigma, ni);
				real_t f1 = (mu - density) / sigma2;
				real_t adj_xmu = sum(adj_ti * f1);
				adj_density += adj_xmu;

				//ADJOINT: tf
				if constexpr (HasTFDerivative)
				{
					real4 adj_ci = adj_ti / ci;
					real_t adj_mu = -adj_xmu;
					const float f2 = f1 * (mu - density) / sigma;
					real_t adj_sigma = sum(adj_ti * f2);

					if constexpr (DelayedAccumulation)
					{
						sharedData[6 * i + 0] += adj_ci.x;
						sharedData[6 * i + 1] += adj_ci.y;
						sharedData[6 * i + 2] += adj_ci.z;
						sharedData[6 * i + 3] += adj_ci.w;
						sharedData[6 * i + 4] += adj_mu;
						sharedData[6 * i + 5] += adj_sigma;
					}
					else
					{
						kernel::atomicAdd(&adj_tf[batch][i][0], adj_ci.x);
						kernel::atomicAdd(&adj_tf[batch][i][1], adj_ci.y);
						kernel::atomicAdd(&adj_tf[batch][i][2], adj_ci.z);
						kernel::atomicAdd(&adj_tf[batch][i][3], adj_ci.w);
						kernel::atomicAdd(&adj_tf[batch][i][4], adj_mu);
						kernel::atomicAdd(&adj_tf[batch][i][5], adj_sigma);
					}
				}
			}
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 6;
#ifdef __CUDA_ARCH__
			coalesced_group active = coalesced_threads();
			const int tia = active.thread_rank();
#endif
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
			{
				real_t val = sharedData[C * r + c];
#ifdef __CUDA_ARCH__
				real_t reduction = reduce(active, val, plus<real_t>());
				active.sync();
				if (tia == 0)
				{ //leader accumulates
					kernel::atomicAdd(&adj_tf[batch][r][c], reduction);
				}
#else
				//fallback accumulation
				kernel::atomicAdd(&adj_tf[batch][r][c], val);
#endif
			}
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 6;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}
	};
	
	
	
	template<>
	struct TransferFunctionEval<TFPreshaded>
	{
		__host__ __device__ __inline__ real4 eval(
			const Tensor3Read& tf, int batch, real4 density) const
		{
			return density;
		}

		template<bool HasTFDerivative, bool DelayedAccumulation>
		__host__ __device__ __inline__ void adjoint(
			const Tensor3Read& tf, int batch, real4 density,
			const real4& adj_color, real4& adj_density,
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
			adj_density = adj_color;
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW& adj_tf, real_t* sharedData) const
		{
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW& adj_tf, real_t* sharedData) const
		{
		}

		template<int D, typename density_t, bool hasTFDerivative>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, hasTFDerivative> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			return density;
		}

	};
	
}

#define CALL_KERNEL_TF(tf_static, ...)	\
	do {	\
		static constexpr kernel::TFMode tfMode = tf_static;	\
		__VA_ARGS__();	\
	} while(0)

#define SWITCH_TF_MODE(tf_dynamic, ...)	\
	switch(tf_dynamic) {	\
		case kernel::TFMode::TFIdentity:	\
			CALL_KERNEL_TF(kernel::TFMode::TFIdentity, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFTexture:	\
			CALL_KERNEL_TF(kernel::TFMode::TFTexture, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFLinear:	\
			CALL_KERNEL_TF(kernel::TFMode::TFLinear, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFGaussian:	\
			CALL_KERNEL_TF(kernel::TFMode::TFGaussian, __VA_ARGS__);	\
			break;	\
		default:	\
			throw std::runtime_error("unknown TF mode");	\
	}
#define SWITCH_TF_MODE_WITH(tf_dynamic, extra_mode, ...)	\
	switch(tf_dynamic) {	\
		case kernel::TFMode::TFIdentity:	\
			CALL_KERNEL_TF(kernel::TFMode::TFIdentity, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFTexture:	\
			CALL_KERNEL_TF(kernel::TFMode::TFTexture, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFLinear:	\
			CALL_KERNEL_TF(kernel::TFMode::TFLinear, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::TFGaussian:	\
			CALL_KERNEL_TF(kernel::TFMode::TFGaussian, __VA_ARGS__);	\
			break;	\
		case kernel::TFMode::extra_mode:	\
			CALL_KERNEL_TF(kernel::TFMode::extra_mode, __VA_ARGS__);	\
			break;	\
		default:	\
			throw std::runtime_error("unknown TF mode");	\
	}
