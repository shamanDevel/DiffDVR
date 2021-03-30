#pragma once

#include "renderer_settings.cuh"
#include "forward_vector.h"
#include "renderer_adjoint.cuh"

namespace kernel {
	
	/**
	 * \brief Helper for volume interpolation.
	 * \tparam volumeFilterMode the interpolation mode
	 */
	template<VolumeFilterMode volumeFilterMode,
		bool hasAdjoint=false, bool hasVolumeAdjoint=false>
	struct VolumeInterpolation
	{
		/**
		 * \brief Fetches the density from the volume
		 * \tparam pos_t the typoe of the volume position, either
		 *   real3 or cudAD::fvar<real3, D> if forward derivatives are traced
		 * \param volume the volume tensor B*X*Y*Z
		 * \param volumeSize the volume size / resolution
		 * \param batch the batch index
		 * \param volumePos the volume position in [0, X-1]*[0, Y-1]*[0, Z-1]
		 * \return the sampled, interpolated density
		 */
		template<typename pos_t>
		__host__ __device__ __inline__ real_t fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const pos_t& volumePos) const;
	};

	template<>
	struct VolumeInterpolation<FilterNearest, false, false>
	{
		static constexpr real_t EPS = real_t(1e-7);

		__host__ __device__ __inline__ real_t fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const real3& volumePos) const
		{
			//nearest-neighbor interpolation casts to integer
			//-> no gradients are propagated.
			//Hence, I can immediately cast to real3 to remove the forward variable
			real3 volumePosClamped = clamp(volumePos,
				make_real3(EPS), make_real3(volumeSize) - (1 + EPS));
			//nearest-neighbor interpolation
			int3 ipos = make_int3(round(volumePosClamped));
			return volume[batch][ipos.x][ipos.y][ipos.z];
		}

		template<int D>
		__host__ __device__ __inline__ cudAD::fvar<real_t, D> fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const cudAD::fvar<real3, D>& volumePos) const
		{
			//nearest-neighbor interpolation casts to integer
			//-> no gradients are propagated.
			//Hence, I can immediately cast to real3 to remove the forward variable
			real3 volumePosClamped = clamp(static_cast<real3>(volumePos),
				make_real3(EPS), make_real3(volumeSize) - (1 + EPS));
			//nearest-neighbor interpolation
			int3 ipos = make_int3(round(volumePosClamped));
			return cudAD::fvar<real_t, D>::constant(volume[batch][ipos.x][ipos.y][ipos.z]);
		}
	};

	template<bool hasVolumeAdjoint>
	struct VolumeInterpolation<FilterNearest, true, hasVolumeAdjoint>
	{
		const int batch_;
		int3 lastIndex_ = make_int3(-1);
		real_t density_;
		real_t adjDensity_;
		BTensor4RW adj_Volume_;
		
		static constexpr real_t EPS = real_t(1e-7);

		__host__ __device__ __inline__ explicit VolumeInterpolation(
			int batch, const int3& volumeSize, const BTensor4RW& adjVolume)
			: adj_Volume_(adjVolume)
			, batch_(batch)
			, lastIndex_(make_int3(-1))
			, density_(0)
			, adjDensity_(0)
		{}

	private:
		/**
		 * \brief The adjoint values for the current voxel are cached
		 * until the next voxel is entered.
		 * Call this after the end of the tracing loop to explicitly
		 * finish the current voxel.
		 */
		__host__ __device__ __inline__ void finishAdjoint()
		{
			if constexpr (hasVolumeAdjoint) {
				if (adjDensity_ != 0)
				{
					kernel::atomicAdd(
						&(adj_Volume_[batch_][lastIndex_.x][lastIndex_.y][lastIndex_.z]),
						adjDensity_);
					adjDensity_ = 0;
				}
			}
		}

	public:
		__host__ __device__ __inline__ ~VolumeInterpolation() { finishAdjoint(); }
		
		__host__ __device__ __inline__ real_t fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const real3& volumePos)
		{
			//nearest-neighbor interpolation casts to integer
			//-> no gradients are propagated.
			//Hence, I can immediately cast to real3 to remove the forward variable
			real3 volumePosClamped = clamp(volumePos,
			                               make_real3(EPS), make_real3(volumeSize) - (1 + EPS));
			//nearest-neighbor interpolation
			int3 ipos = make_int3(round(volumePosClamped));

			if (any(ipos != lastIndex_))
			{
				//we entered a new cell
				finishAdjoint();
				lastIndex_ = ipos;
				density_ = volume[batch][ipos.x][ipos.y][ipos.z];
			}
			
			return density_;
		}

		//adjoint code, must be preceded by fetch() at the same position
		__host__ __device__ __inline__ void adjoint(
			const real_t& adj_density, real3& adj_volumePos)
		{
			adjDensity_ += adj_density;
			//nearest neighbor does not allow gradients for the position
			adj_volumePos = make_real3(0);
		}
			
	};

	template<
		typename pos_t,
		typename ret_t = decltype(cudAD::getX(pos_t()))>
		__host__ __device__ __inline__ ret_t fetchTrilinear(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const pos_t& volumePos)
	{
		//fetch all eight neighbors
		int3 ipos = make_int3(static_cast<real3>(volumePos));
		int3 iposL = clamp(ipos, make_int3(0), volumeSize - make_int3(1));
		int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize - make_int3(1));
		const real_t d000 = volume[batch][iposL.x][iposL.y][iposL.z];
		const real_t d001 = volume[batch][iposL.x][iposL.y][iposH.z];
		const real_t d010 = volume[batch][iposL.x][iposH.y][iposL.z];
		const real_t d011 = volume[batch][iposL.x][iposH.y][iposH.z];
		const real_t d100 = volume[batch][iposH.x][iposL.y][iposL.z];
		const real_t d101 = volume[batch][iposH.x][iposL.y][iposH.z];
		const real_t d110 = volume[batch][iposH.x][iposH.y][iposL.z];
		const real_t d111 = volume[batch][iposH.x][iposH.y][iposH.z];
		//lerp
		auto fpos = volumePos - make_real3(ipos);
		using namespace cudAD;
		return lerp(
			lerp(
				lerp(d000, d100, getX(fpos)),
				lerp(d010, d110, getX(fpos)),
				getY(fpos)),
			lerp(
				lerp(d001, d101, getX(fpos)),
				lerp(d011, d111, getX(fpos)),
				getY(fpos)),
			getZ(fpos));
	}

	template<>
	struct VolumeInterpolation<FilterTrilinear, false, false>
	{
		static constexpr real_t EPS = real_t(1e-7);
#ifndef TRILINEAR_INTERPOLATION_USE_CACHE
#define TRILINEAR_INTERPOLATION_USE_CACHE 1
#endif

#if TRILINEAR_INTERPOLATION_USE_CACHE==1
		
		int4 lastIndexAndBatch_ = make_int4(-1);
		real_t densities_[8];

	private:
		__host__ __device__ __inline__ void fetchValues(
			const int3& ipos, const int batch, const int3& volumeSize, const Tensor4Read& volume)
		{
			const int3 volumeSize2 = volumeSize - make_int3(1);
			int3 iposL = clamp(ipos, make_int3(0), volumeSize2);
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize2);
			densities_[0b000] = volume[batch][iposL.x][iposL.y][iposL.z];
			densities_[0b001] = volume[batch][iposL.x][iposL.y][iposH.z];
			densities_[0b010] = volume[batch][iposL.x][iposH.y][iposL.z];
			densities_[0b011] = volume[batch][iposL.x][iposH.y][iposH.z];
			densities_[0b100] = volume[batch][iposH.x][iposL.y][iposL.z];
			densities_[0b101] = volume[batch][iposH.x][iposL.y][iposH.z];
			densities_[0b110] = volume[batch][iposH.x][iposH.y][iposL.z];
			densities_[0b111] = volume[batch][iposH.x][iposH.y][iposH.z];
		}

	public:
		template<
			typename pos_t,
			typename ret_t = decltype(cudAD::getX(pos_t()))>
			__host__ __device__ __inline__ ret_t fetch(
				const Tensor4Read& volume, const int3& volumeSize,
				int batch, const pos_t& volumePos)
		{
			//fetch all eight neighbors
			int3 ipos = make_int3(static_cast<real3>(volumePos));

			int4 indexAndBatch = make_int4(ipos, batch);
			if (any(indexAndBatch != lastIndexAndBatch_))
			{
				fetchValues(ipos, batch, volumeSize, volume);
				lastIndexAndBatch_ = indexAndBatch;
			}
			
			//lerp
			auto fpos = volumePos - make_real3(ipos);
			using namespace cudAD;
			return lerp(
				lerp(
					lerp(densities_[0b000], densities_[0b100], getX(fpos)),
					lerp(densities_[0b010], densities_[0b110], getX(fpos)),
					getY(fpos)),
				lerp(
					lerp(densities_[0b001], densities_[0b101], getX(fpos)),
					lerp(densities_[0b011], densities_[0b111], getX(fpos)),
					getY(fpos)),
				getZ(fpos));
		}
		
#else
		template<
			typename pos_t,
			typename ret_t=decltype(cudAD::getX(pos_t()))>
		__host__ __device__ __inline__ ret_t fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const pos_t& volumePos) const
		{
			return fetchTrilinear(volume, volumeSize, batch, volumePos);
		}
#endif
	};

	template<bool hasVolumeAdjoint>
	struct VolumeInterpolation<FilterTrilinear, true, hasVolumeAdjoint>
	{
		const int batch_;
		const int3 volumeSize_;
		int3 lastIndex_;
		real_t densities_[8];
		real_t adjDensities_[8];
		BTensor4RW adjVolume_;
		real3 fpos_;

		static constexpr real_t EPS = real_t(1e-7);

		__host__ __device__ __inline__ explicit VolumeInterpolation(
			int batch, const int3& volumeSize, const BTensor4RW& adjVolume)
			: adjVolume_(adjVolume)
			, batch_(batch)
			, volumeSize_(volumeSize - make_int3(1))
			, lastIndex_(make_int3(-1))
			, densities_()
			, adjDensities_()
		{}

	private:
		__host__ __device__ __inline__ void fetchValues(
			const int3& ipos, const Tensor4Read& volume)
		{
			int3 iposL = clamp(ipos, make_int3(0), volumeSize_);
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize_);
			densities_[0b000] = volume[batch_][iposL.x][iposL.y][iposL.z];
			densities_[0b001] = volume[batch_][iposL.x][iposL.y][iposH.z];
			densities_[0b010] = volume[batch_][iposL.x][iposH.y][iposL.z];
			densities_[0b011] = volume[batch_][iposL.x][iposH.y][iposH.z];
			densities_[0b100] = volume[batch_][iposH.x][iposL.y][iposL.z];
			densities_[0b101] = volume[batch_][iposH.x][iposL.y][iposH.z];
			densities_[0b110] = volume[batch_][iposH.x][iposH.y][iposL.z];
			densities_[0b111] = volume[batch_][iposH.x][iposH.y][iposH.z];
		}
		/**
		 * \brief The adjoint values for the current voxel are cached
		 * until the next voxel is entered.
		 * Call this after the end of the tracing loop to explicitly
		 * finish the current voxel.
		 */
		__host__ __device__ __inline__ void finishAdjoint()
		{
			if constexpr (hasVolumeAdjoint) {
				if (lastIndex_.x >= 0) //we are not in uninitialized state
				{
					int3 iposL = clamp(lastIndex_, make_int3(0), volumeSize_);
					int3 iposH = clamp(lastIndex_ + make_int3(1), make_int3(0), volumeSize_);

					kernel::atomicAdd(&(adjVolume_[batch_][iposL.x][iposL.y][iposL.z]), adjDensities_[0b000]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposL.x][iposL.y][iposH.z]), adjDensities_[0b001]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposL.x][iposH.y][iposL.z]), adjDensities_[0b010]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposL.x][iposH.y][iposH.z]), adjDensities_[0b011]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposH.x][iposL.y][iposL.z]), adjDensities_[0b100]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposH.x][iposL.y][iposH.z]), adjDensities_[0b101]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposH.x][iposH.y][iposL.z]), adjDensities_[0b110]);
					kernel::atomicAdd(&(adjVolume_[batch_][iposH.x][iposH.y][iposH.z]), adjDensities_[0b111]);

					#pragma unroll
					for (int i = 0; i < 8; ++i) 
						adjDensities_[i] = 0;
				}
			}
		}

	public:
		__host__ __device__ __inline__ ~VolumeInterpolation() { finishAdjoint(); }

		VolumeInterpolation(const VolumeInterpolation& other) = delete;
		VolumeInterpolation(VolumeInterpolation&& other) noexcept = delete;
		VolumeInterpolation& operator=(const VolumeInterpolation& other) = delete;
		VolumeInterpolation& operator=(VolumeInterpolation&& other) noexcept = delete;
		
		__host__ __device__ __inline__ real_t fetch(
				const Tensor4Read& volume, const int3& volumeSize,
				int batch, const real3& volumePos)
		{
			//fetch all eight neighbors
			int3 ipos = make_int3(volumePos);
			if (any(ipos != lastIndex_))
			{
				finishAdjoint();
				lastIndex_ = ipos;
				fetchValues(ipos, volume);
			}
			
			//lerp
			fpos_ = volumePos - make_real3(ipos);
			using namespace cudAD;
			return lerp(
				lerp(
					lerp(densities_[0b000], densities_[0b100], fpos_.x),
					lerp(densities_[0b010], densities_[0b110], fpos_.x),
					fpos_.y),
				lerp(
					lerp(densities_[0b001], densities_[0b101], fpos_.x),
					lerp(densities_[0b011], densities_[0b111], fpos_.x),
					fpos_.y),
				fpos_.z);
		}
		
		//adjoint code, must be preceded by fetch() at the same position
		__host__ __device__ __inline__ void adjoint(
			const real_t& adj_density, real3& adj_volumePos)
		{
			const real3 cw = fpos_;
			const real3 fw = make_real3(1) - fpos_;
			real3 adj_cw = make_real3(0);
			real3 adj_fw = make_real3(0);

#define scatter(idx, X, Y, Z)	\
	do {	\
		adjDensities_[idx] += adj_density*(X.x*Y.y*Z.z);	\
		adj_ ## X .x += adj_density*densities_[idx]*Y.y*Z.z;	\
		adj_ ## Y .y += adj_density*densities_[idx]*X.x*Z.z;	\
		adj_ ## Z .z += adj_density*densities_[idx]*X.x*Y.y;	\
	} while(0)
	

			scatter(0b000, fw, fw, fw);
			scatter(0b100, cw, fw, fw);
			scatter(0b010, fw, cw, fw);
			scatter(0b110, cw, cw, fw);
			scatter(0b001, fw, fw, cw);
			scatter(0b101, cw, fw, cw);
			scatter(0b011, fw, cw, cw);
			scatter(0b111, cw, cw, cw);

			adj_volumePos = adj_cw - adj_fw;
		}
#undef scatter
	};



	template<>
	struct VolumeInterpolation<FilterPreshaded, false, false>
	{
		static constexpr real_t EPS = real_t(1e-7);

		__host__ __device__ __inline__ real4 fetch(
				const Tensor4Read& volume, const int3& volumeSize,
				int batch, const real3& volumePos) const
		{
			//fetch all eight neighbors
			int3 ipos = make_int3(static_cast<real3>(volumePos));
			int3 iposL = clamp(ipos, make_int3(0), volumeSize - make_int3(1));
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize - make_int3(1));
			real_t channels[4];
#pragma unroll
			for (int b = 0; b < 4; ++b) {
				const real_t d000 = volume[b][iposL.x][iposL.y][iposL.z];
				const real_t d001 = volume[b][iposL.x][iposL.y][iposH.z];
				const real_t d010 = volume[b][iposL.x][iposH.y][iposL.z];
				const real_t d011 = volume[b][iposL.x][iposH.y][iposH.z];
				const real_t d100 = volume[b][iposH.x][iposL.y][iposL.z];
				const real_t d101 = volume[b][iposH.x][iposL.y][iposH.z];
				const real_t d110 = volume[b][iposH.x][iposH.y][iposL.z];
				const real_t d111 = volume[b][iposH.x][iposH.y][iposH.z];
				//lerp
				auto fpos = volumePos - make_real3(ipos);
				using namespace cudAD;
				channels[b] = lerp(
					lerp(
						lerp(d000, d100, getX(fpos)),
						lerp(d010, d110, getX(fpos)),
						getY(fpos)),
					lerp(
						lerp(d001, d101, getX(fpos)),
						lerp(d011, d111, getX(fpos)),
						getY(fpos)),
					getZ(fpos));
			}
			return make_real4(channels[0], channels[1], channels[2], channels[3]);
		}
	};

	template<bool hasVolumeAdjoint>
	struct VolumeInterpolation<FilterPreshaded, true, hasVolumeAdjoint>
	{
		const int3 volumeSize_;
		int3 lastIndex_;
		real_t densities_[4][8];
		real_t adjDensities_[4][8];
		BTensor4RW adjVolume_;
		real3 fpos_;

		static constexpr real_t EPS = real_t(1e-7);

		__host__ __device__ __inline__ explicit VolumeInterpolation(
			int batch, const int3& volumeSize, const BTensor4RW& adjVolume)
			: adjVolume_(adjVolume)
			, volumeSize_(volumeSize - make_int3(1))
			, lastIndex_(make_int3(-1))
			, densities_()
			, adjDensities_()
		{}

	private:
		__host__ __device__ __inline__ void fetchValues(
			const int3& ipos, const Tensor4Read& volume)
		{
			int3 iposL = clamp(ipos, make_int3(0), volumeSize_);
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize_);
#pragma unroll
			for (int b = 0; b < 4; ++b) {
				densities_[b][0b000] = volume[b][iposL.x][iposL.y][iposL.z];
				densities_[b][0b001] = volume[b][iposL.x][iposL.y][iposH.z];
				densities_[b][0b010] = volume[b][iposL.x][iposH.y][iposL.z];
				densities_[b][0b011] = volume[b][iposL.x][iposH.y][iposH.z];
				densities_[b][0b100] = volume[b][iposH.x][iposL.y][iposL.z];
				densities_[b][0b101] = volume[b][iposH.x][iposL.y][iposH.z];
				densities_[b][0b110] = volume[b][iposH.x][iposH.y][iposL.z];
				densities_[b][0b111] = volume[b][iposH.x][iposH.y][iposH.z];
			}
		}
		/**
		 * \brief The adjoint values for the current voxel are cached
		 * until the next voxel is entered.
		 * Call this after the end of the tracing loop to explicitly
		 * finish the current voxel.
		 */
		__host__ __device__ __inline__ void finishAdjoint()
		{
			if constexpr (hasVolumeAdjoint) {
				if (lastIndex_.x >= 0) //we are not in uninitialized state
				{
					int3 iposL = clamp(lastIndex_, make_int3(0), volumeSize_);
					int3 iposH = clamp(lastIndex_ + make_int3(1), make_int3(0), volumeSize_);

#pragma unroll
					for (int b = 0; b < 4; ++b) {
						kernel::atomicAdd(&(adjVolume_[b][iposL.x][iposL.y][iposL.z]), adjDensities_[b][0b000]);
						kernel::atomicAdd(&(adjVolume_[b][iposL.x][iposL.y][iposH.z]), adjDensities_[b][0b001]);
						kernel::atomicAdd(&(adjVolume_[b][iposL.x][iposH.y][iposL.z]), adjDensities_[b][0b010]);
						kernel::atomicAdd(&(adjVolume_[b][iposL.x][iposH.y][iposH.z]), adjDensities_[b][0b011]);
						kernel::atomicAdd(&(adjVolume_[b][iposH.x][iposL.y][iposL.z]), adjDensities_[b][0b100]);
						kernel::atomicAdd(&(adjVolume_[b][iposH.x][iposL.y][iposH.z]), adjDensities_[b][0b101]);
						kernel::atomicAdd(&(adjVolume_[b][iposH.x][iposH.y][iposL.z]), adjDensities_[b][0b110]);
						kernel::atomicAdd(&(adjVolume_[b][iposH.x][iposH.y][iposH.z]), adjDensities_[b][0b111]);
					}

#pragma unroll
					for (int b=0; b<4; ++b) for (int i = 0; i < 8; ++i)
						adjDensities_[b][i] = 0;
				}
			}
		}

	public:
		__host__ __device__ __inline__ ~VolumeInterpolation() { finishAdjoint(); }

		VolumeInterpolation(const VolumeInterpolation& other) = delete;
		VolumeInterpolation(VolumeInterpolation&& other) noexcept = delete;
		VolumeInterpolation& operator=(const VolumeInterpolation& other) = delete;
		VolumeInterpolation& operator=(VolumeInterpolation&& other) noexcept = delete;
		
		__host__ __device__ __inline__ real4 fetch(
			const Tensor4Read& volume, const int3& volumeSize,
			int batch, const real3& volumePos)
		{
			//fetch all eight neighbors
			int3 ipos = make_int3(volumePos);
			if (any(ipos != lastIndex_))
			{
				finishAdjoint();
				lastIndex_ = ipos;
				fetchValues(ipos, volume);
			}

			//lerp
			fpos_ = volumePos - make_real3(ipos);
			using namespace cudAD;
			real_t channels[4];
			for (int b = 0; b < 4; ++b) {
				channels[b] = lerp(
					lerp(
						lerp(densities_[b][0b000], densities_[b][0b100], fpos_.x),
						lerp(densities_[b][0b010], densities_[b][0b110], fpos_.x),
						fpos_.y),
					lerp(
						lerp(densities_[b][0b001], densities_[b][0b101], fpos_.x),
						lerp(densities_[b][0b011], densities_[b][0b111], fpos_.x),
						fpos_.y),
					fpos_.z);
			}
			return make_real4(channels[0], channels[1], channels[2], channels[3]);
		}

		//adjoint code, must be preceded by fetch() at the same position
		__host__ __device__ __inline__ void adjoint(
			const real4& adj_density, real3& adj_volumePos)
		{
			const real3 cw = fpos_;
			const real3 fw = make_real3(1) - fpos_;
			real3 adj_cw = make_real3(0);
			real3 adj_fw = make_real3(0);

#define scatter(idx, X, Y, Z)	\
	do {	\
		adjDensities_[b][idx] += adj_colors[b]*(X.x*Y.y*Z.z);	\
		adj_ ## X .x += adj_colors[b]*densities_[b][idx]*Y.y*Z.z;	\
		adj_ ## Y .y += adj_colors[b]*densities_[b][idx]*X.x*Z.z;	\
		adj_ ## Y .z += adj_colors[b]*densities_[b][idx]*X.x*Y.y;	\
	} while(0)

			real_t adj_colors[4] = { adj_density.x, adj_density.y, adj_density.z, adj_density.w };
			for (int b = 0; b < 4; ++b) {
				scatter(0b000, fw, fw, fw);
				scatter(0b100, cw, fw, fw);
				scatter(0b010, fw, cw, fw);
				scatter(0b110, cw, cw, fw);
				scatter(0b001, fw, fw, cw);
				scatter(0b101, cw, fw, cw);
				scatter(0b011, fw, cw, cw);
				scatter(0b111, cw, cw, cw);
			}

			adj_volumePos = adj_cw - adj_fw;
#undef scatter
		}
	};
	}
