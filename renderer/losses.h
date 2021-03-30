#pragma once

#include "commons.h"

#include <cuda_runtime.h>
#include <ATen/native/TensorIterator.h>
#include <torch/types.h>

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"
#include "renderer_tf.cuh"

BEGIN_RENDERER_NAMESPACE

/*
 * A collection of miscellaneous kernels
 */

/**
 * Computes $t * log_2(t)$ with proper handling of $t=0$.
 * This function is differentiable
 */
torch::Tensor mul_log(const torch::Tensor& t);

/**
 * \brief Computes the log-MSE loss, that is, for inputs x and y it holds:
 *
 * \f[
 *    (x-y)^2 = MSE(x,y) = \exp(logMSE(\log(x), \log(y)))
 * \f]
 *
 * This function is differentiable
 *
 * \param logX the log of the first argument
 * \param logY the log of the second argument
 * \return log((exp(logX)-exp(logY))^2)
 */
torch::Tensor logMSE(const torch::Tensor& logX, const torch::Tensor& logY);

/**
 * \brief Computes the log-L1 loss, that is, for inputs x and y it holds:
 *
 * \f[
 *    |x-y| = L1(x,y) = \exp(logL1(\log(x), \log(y)))
 * \f]
 *
 * This function is differentiable
 *
 * \param logX the log of the first argument
 * \param logY the log of the second argument
 * \return log(|exp(logX)-exp(logY)|)
 */
torch::Tensor logL1(const torch::Tensor& logX, const torch::Tensor& logY);

/**
 * \brief Performs importance sampling on the volume.
 * There are four sampling categories, weighted by the respective
 * 'weight*' parameters:
 * - uniform: accept all samples
 * - density gradient: rejection sampling based on the gradient norm of the density
 * - opacity: rejection sampling based on the opacity after TF mapping
 * - opacity gradient: rejection sampling based on the opacity gradient norm after TF mapping
 * The weights will be normalized.
 *
 * Supported device: GPU
 * 
 * \param sampleLocations the locations of the samples, float tensor on the GPU of
 *	shape (3,N) where N is the number of samples and each sample is in [0,1]^3.
 * \param densityVolume the density volume of shape 1*X*Y*Z
 * \param tf the transfer function tensor
 * \param tfMode the type of transfer function
 * \param weightUniform the weight for uniform sampling
 * \param weightDensityGradient the weight for density gradient norm sampling
 * \param weightOpacity  the weight for opacity sampling
 * \param weightOpacityGradient the weight for opacity gradient norm sampling
 * \param seed the seed for the random number generator
 * \return a boolean tensor of shape (N,) with true iff that sample should be taken.
 */
torch::Tensor sampleImportance(
	const torch::Tensor& sampleLocations,
	const torch::Tensor& densityVolume,
	const torch::Tensor& tf, kernel::TFMode tfMode,
	float weightUniform, float weightDensityGradient,
	float weightOpacity, float weightOpacityGradient,
	int seed);

END_RENDERER_NAMESPACE

namespace kernel
{
	void mulLogForward(
		at::TensorIterator& iter, // t -> output
		bool cuda);

	void mulLogBackward(
		at::TensorIterator& iter, //(t, grad_output) -> grad_t
		bool cuda);

	void logMSEForward(at::TensorIterator& iter, bool cuda);
	void logMSEBackward(at::TensorIterator& iter, bool cuda);
	void logL1Forward(at::TensorIterator& iter, bool cuda);
	void logL1Backward(at::TensorIterator& iter, bool cuda);
	
	typedef PackedTensorAccessor32<bool, 1, DefaultPtrTraits> BTensor1RW;
	void sampleImportanceCUDA(
		BTensor1RW& output,
		const Tensor2Read& sampleLocations,
		const Tensor4Read& densityVolume,
		const Tensor3Read& tf, TFMode tfMode,
		float weightUniform, float weightDensityGradient, //cummulative weights
		float weightOpacity, float weightOpacityGradient /*=1*/,
		int seed);
	
}
