#pragma once

#include "commons.h"
#include "helper_math.cuh"
#include <torch/types.h>

#include <cuda_runtime.h>
#include <vector>

#include "renderer_settings.cuh"

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

BEGIN_RENDERER_NAMESPACE

//From https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp
//Assumes color channels are in [0,1]
__device__ __host__
inline real3 rgbToXyz(const real3& rgb)
{
	real_t r = ((rgb.x > 0.04045f) ? powf((rgb.x + 0.055f) / 1.055f, 2.4f) : (rgb.x / 12.92f)) * 100.0f;
	real_t g = ((rgb.y > 0.04045f) ? powf((rgb.y + 0.055f) / 1.055f, 2.4f) : (rgb.y / 12.92f)) * 100.0f;
	real_t b = ((rgb.z > 0.04045f) ? powf((rgb.z + 0.055f) / 1.055f, 2.4f) : (rgb.z / 12.92f)) * 100.0f;

	real_t x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
	real_t y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
	real_t z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

	return { x, y, z };
}

//Output color channels are in [0,1]
__device__ __host__
inline real3 xyzToRgb(const real3& xyz)
{
	auto x = xyz.x / 100.0f;
	auto y = xyz.y / 100.0f;
	auto z = xyz.z / 100.0f;

	auto r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
	auto g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
	auto b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

	r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
	g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
	b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

	return { r, g, b };
}

__device__ __host__
inline real3 rgbToLab(const real3& rgb)
{
	auto xyz = rgbToXyz(rgb);

	auto x = xyz.x / 95.047f;
	auto y = xyz.y / 100.00f;
	auto z = xyz.z / 108.883f;

	x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 16.0f / 116.0f);
	y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 16.0f / 116.0f);
	z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 16.0f / 116.0f);

	return { (116.0f * y) - 16.0f, 500.0f * (x - y), 200.0f * (y - z) };
}

__device__ __host__
inline real3 labToRgb(const real3& lab)
{
	auto y = (lab.x + 16.0f) / 116.0f;
	auto x = lab.y / 500.0f + y;
	auto z = y - lab.z / 200.0f;

	auto x3 = x * x * x;
	auto y3 = y * y * y;
	auto z3 = z * z * z;

	x = ((x3 > 0.008856f) ? x3 : ((x - 16.0f / 116.0f) / 7.787f)) * 95.047f;
	y = ((y3 > 0.008856f) ? y3 : ((y - 16.0f / 116.0f) / 7.787f)) * 100.0f;
	z = ((z3 > 0.008856f) ? z3 : ((z - 16.0f / 116.0f) / 7.787f)) * 108.883f;

	return xyzToRgb({ x, y, z });
}

struct TFPoint
{
	real_t pos;
	real4 val; //rgb, opacity
};

struct TFUtils
{
	TFUtils() = delete;

	/**
	 * \brief Assemble the merged list of TF control points from the settings.
	 * \return 
	 */
	static std::vector<TFPoint> assembleFromSettings(
		const std::vector<real3>& colorValuesLab,
		const std::vector<real_t>& colorPositions,
		const std::vector<real_t>& opacityValues,
		const std::vector<real_t>& opacityPositions,
		real_t minDensity, real_t maxDensity, real_t opacityScaling,
		bool purgeZeroOpacityRegions = true
	);

	/**
	 * \brief Converts the list of control points to a tensor of shape 1*R*C
	 * fit for kernel::TFMode::Linear.
	 * The resulting tensor will have dtype=float32 and device=cpu.
	 */
	static torch::Tensor getPiecewiseTensor(const std::vector<TFPoint>& points);

	/**
	 * \brief Converts the list of control points to a tensor of shape 1*R*C
	 * fit for kernel::TFMode::Texture.
	 * The resulting tensor will have dtype=float32 and device=cpu.
	 */
	static torch::Tensor getTextureTensor(const std::vector<TFPoint>& points, int resolution);

	/**
	 * \brief Pre-shades a density volume of shape 1*X*Y*Z using
	 * the specified transfer function into a color volume of shape
	 * 4*X*Y*Z.
	 *
	 * This function implements the PyTorch autograd protocoll,
	 * it is differentiable with respect to the density volume.
	 * 
	 * \param densityVolume the density volume of shape 1*X*Y*Z
	 * \param tf the transfer function tensor
	 * \param tfMode the type of transfer function
	 * \return the preshaded color volume of shape 4*X*Y*Z, channels are r,g,b,opacity
	 */
	static torch::Tensor preshadeVolume(
		const torch::Tensor& densityVolume,
		const torch::Tensor& tf, kernel::TFMode tfMode);

	/**
	 * \brief Finds the best matching density value for the given
	 * pre-shaded color volume and transfer function.
	 */
	static torch::Tensor findBestFit(
		const torch::Tensor& colorVolume,
		const torch::Tensor& tf, kernel::TFMode tfMode,
		int numSamples, real_t opacityWeighting,
		const torch::Tensor* previousDensities = nullptr,
		real_t neighborWeighting = 0.1f);
};

END_RENDERER_NAMESPACE

namespace kernel
{
	void PreshadeVolume(
		const Tensor4Read& volume, const Tensor3Read& tf, Tensor4RW& color,
		TFMode tfMode, bool cuda);

	void PreshadeVolumeAdj(
		const Tensor4Read& volume, const Tensor3Read& tf,
		const Tensor4Read& adjColor,
		Tensor4RW& adjVolume,
		TFMode tfMode, bool cuda); //TODO: adjoint for TF

	void FindBestFit(
		const Tensor4Read& colorVolume, const Tensor3Read& tf,
		Tensor4RW& densityOut, const LTensor1Read& seeds,
		kernel::TFMode tfMode, int numSamples, real_t opacityWeighting,
		bool cuda);

	void FindBestFitWithComparison(
		const Tensor4Read& colorVolume, const Tensor3Read& tf,
		const Tensor4RW& previousDensity,
		Tensor4RW& densityOut, const LTensor1Read& seeds,
		kernel::TFMode tfMode, int numSamples, real_t opacityWeighting,
		real_t neighborWeighting, bool cuda);
}

#ifdef _WIN32
#pragma warning( pop )
#endif
