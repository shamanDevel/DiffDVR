#pragma once

#include "commons.h"
#include <variant>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <glm/glm.hpp>
#include <filesystem>

#include "renderer_settings.cuh"

BEGIN_RENDERER_NAMESPACE

/**
 * \brief Host version of the renderer inputs.
 * Some settings which can be specified globally can be passed
 * as the basic type instead of a tensor.
 * They are copied to a tensor then under-the-hood.
 */
struct RendererInputsHost
{
	int2 screenSize; //W*H
					 //must match the sizes of per-pixel settings

	kernel::VolumeFilterMode volumeFilterMode;
	torch::Tensor volume; //B*X*Y*Z

	std::variant<torch::Tensor, real3> boxMin; // (B)*3
	std::variant<torch::Tensor, real3> boxSize;// (B)*3

	kernel::CameraMode cameraMode;
	struct CameraPerPixelRays
	{
		torch::Tensor cameraRayStart; //B*H*W*3
		torch::Tensor cameraRayDir; //B*H*W*3
	};
	typedef torch::Tensor CameraInverseViewMatrixTensor; //B*4*4
	typedef glm::mat4 CameraInverseViewMatrixScalar;
	struct CameraReferenceFrame
	{
		/**
		 * the viewport matrix of shape B*3*3
		 *   where viewport[:,0,:] is the camera/eye position,
		 *   viewport[:,1,:] the right vector and
		 *   viewport[:,2,:] the up vector.
		 */
		torch::Tensor viewport;
		// field-of-view for the y-axis in radians
		real_t fovYRadians;
	};
	std::variant<
		CameraPerPixelRays,
		CameraInverseViewMatrixTensor,
		CameraInverseViewMatrixScalar,
		CameraReferenceFrame> camera;
	
	std::variant<torch::Tensor, real_t> stepSize; //B*H*W

	kernel::TFMode tfMode;
	torch::Tensor tf; //(B)*R*C

	kernel::BlendMode blendMode;
	real_t blendingEarlyOut = real_t(1) - real_t(1e-5);
};

//The outputs of the forward pass
struct RendererOutputsHost
{
	torch::Tensor color; //B*H*W*4 (r,g,b,alpha)
	torch::Tensor terminationIndex; //B*H*W integer
};

//host-side version of ForwardDifferencesSettings
struct ForwardDifferencesSettingsHost
{
	int D; //number of derivatives, a template parameter in the kernel.

	static constexpr const int NOT_ENABLED = 
		kernel::ForwardDifferencesSettings::NOT_ENABLED; //the variable is not enabled

	//index into the derivative array for the step size
	int d_stepsize = NOT_ENABLED;

	//derivative indices for the start position of the ray
	int3 d_rayStart = make_int3(NOT_ENABLED, NOT_ENABLED, NOT_ENABLED);
	//derivative indices for the ray direction
	int3 d_rayDir = make_int3(NOT_ENABLED, NOT_ENABLED, NOT_ENABLED);

	//derivative index for the TF parameters, shape B*R*C
	torch::Tensor d_tf;
	//performance optimization, true iff at least one entry in d_tf is != NOT_ENABLED
	bool hasTfDerivatives = false;
};

struct AdjointOutputsHost
{
	bool hasVolumeDerivatives = false;
	torch::Tensor adj_volume; // (B)*X*Y*Z

	bool hasStepSizeDerivatives = false;
	torch::Tensor adj_stepSize; // (B)*(H)*(W)

	bool hasCameraDerivatives = false;
	torch::Tensor adj_cameraRayStart; // (B)*H*W*3
	torch::Tensor adj_cameraRayDir;   // (B)*H*W*3

	bool hasTFDerivatives = false;
	// delay final accumulation until ray finished tracing. Might be faster...
	bool tfDelayedAcummulation = false;
	torch::Tensor adj_tf; // (B)*R*C
};

class Renderer
{
public:
	typedef std::vector<torch::Tensor> TensorsToKeepAlive_t;
	static std::tuple<kernel::RendererInputs, TensorsToKeepAlive_t>
		checkInputs(const RendererInputsHost& inputsHost,
			int& B, int& W, int& H, int& X, int& Y, int& Z, bool& cuda,
			bool ignoreVolume = false);
	static kernel::RendererOutputs checkOutput(renderer::RendererOutputsHost& outputsHost,
		int B, int H, int W, bool cuda);
	
	/**
	 * Loads the dynamic cuda libraries of PyTorch.
	 * Call this in the main executable before any other calls
	 * to the renderer or PyTorch.
	 * \return true if cuda is available and was loaded successfully
	 */
	static bool initCuda();
	static void setCudaCacheDir(const std::filesystem::path& path);
	static void reloadCudaKernels();
	/*
	 * \brief Disables the kernel cache on disk.
	 * Can only be reactivated by specifying the cache directory with
	 * \ref setCudaCacheDir()
	 */
	static void disableCudaCache();
	
	/*
	 * \brief Cleans up all CUDA references,
	 * i.e. unloads all cached kernels.
	 * This function is safe to be called multiple times
	 */
	static void cleanupCuda();

	//in sync-mode, a device sync is issued after each kernel launch (for debugging)
	static void setCudaSyncMode(bool sync);
	static bool getCudaSyncMode();

	//enables debug information for the CUDA kernels
	static void setCudaDebugMode(bool debug);
	
	/**
	 * \brief Performs a simple forward rendering without gradients.
	 * \param inputsHost the input settings
	 * \param outputsHost the rendering outputs
	 */
	static void renderForward(
		const RendererInputsHost& inputsHost,
		RendererOutputsHost& outputsHost);

	/**
	 * \brief Performs forward rendering with forward derivative computations.
	 * 
	 * The input settings are specified in 'inputsHost' and the regular rendering
	 * output is written to 'outputsHost'. This is exactly equal to the
	 * result from \ref renderForward().
	 *
	 * Now this function adds functionality to compute forward derivatives
	 * with respect to a subset of the input settings.
	 * Let 'D' be the number of scalar parameters for which derivatives are traced.
	 * The mapping which parameter is traced with which index is done in
	 * 'differnecesSettingsHost'.
	 * The gradient output tensor 'gradientsOut' of shape B*H*W*D*C
	 * contains the derivatives of each of the 'D' parameters (in the 4th dimension)
	 * for each pixel x,y (dimension 3,2, of size W,H) and color channel
	 * C=4 (red, green, blue, alpha).
	 *
	 * To include this in the backward pass of an adjoint optimization framework,
	 * store or recompute this gradient tensor 'gradientsOut'.
	 * The backward pass provides as input the adjoint variables of the color
	 * output 'adjointColor', e.g. a tensor of shape B*H*W*C (C=4).
	 * Then the adjoint variable for the i-th parameter (0<=i<D) is computed
	 * by dot(adjointColor, gradientsOut[:,:,:,i,:]).
	 * 
	 * \param inputsHost the input settings
	 * \param differencesSettingsHost specification of which derivatives to compute
	 * \param outputsHost the rendered output
	 * \param gradientsOut the output gradients of shape B*H*W*D*C
	 */
	static void renderForwardGradients(
		const RendererInputsHost& inputsHost,
		ForwardDifferencesSettingsHost& differencesSettingsHost,
		RendererOutputsHost& outputsHost,
		torch::Tensor& gradientsOut);

	/**
	 * \brief Converts the forward variables from \ref renderForwardGradients()
	 *	to adjoint variables for the camera (rayStart, rayDir), stepsize and
	 *	transfer function, with respect to the adjoint variable of the
	 *	color output 'gradientOutputColor'.
	 *	This allows the forward differentiation to be embedded in an
	 *	adjoint optimization framework.
	 *	
	 * \param forwardVariables the forward variables of shape B*H*W*D*C
	 *		from Renderer::renderForwardGradients
	 * \param gradientOutputColor the adjoint variable of the output color
	 * \param differencesSettingsHost the mapping from derivative index to variable
	 * \param adj_outputs the adjoint variables for the individual parts.
	 */
	static void forwardVariablesToGradients(
		const torch::Tensor& forwardVariables,
		const torch::Tensor& gradientOutputColor,
		ForwardDifferencesSettingsHost& differencesSettingsHost,
		AdjointOutputsHost& adj_outputs);

	/**
	 * \brief Computes the L2-difference of the rendered color image
	 * to a reference color image, while propagating forward derivatives.
	 *
	 * If reduce==false, the differences are computed per-pixel
	 * and the gradients also store the per-pixel derivatives.
	 * If reduce==true, the differences are averaged over the image and batch.
	 *
	 * For the shapes:
	 * - B: the batch dimension
	 * - H: the height of the image
	 * - W: the width of the image
	 * - D: the number of derivatives to trace
	 * - C: the color channels (=4)
	 * 
	 * \param colorInput the color input of shape B*H*W*C
	 * \param gradientsInput the gradients for the color input of shape B*H*W*D*C
	 * \param colorReference the reference color image of shape B*H*W*C
	 * \param differenceOut output: the L2-norm values.
	 *    reduce==false -> per pixel, shape B*H*W*1
	 *    reduce==true  -> averaged, shape 1
	 * \param gradientsOut output: the forward gradients for the difference
	 *    reduce==false -> per pixel, shape B*H*W*D
	 *    reduce==true  -> averaged, shape D
	 * \param reduce true iff the L2-norm should be averaged over the whole image.
	 */
	static void compareToImage(
		const torch::Tensor& colorInput,
		const torch::Tensor& gradientsInput,
		const torch::Tensor& colorReference,
		torch::Tensor& differenceOut,
		torch::Tensor& gradientsOut,
		bool reduce = false);

	/**
	 * \brief Performs adjoint/backward rendering.
	 * It starts with the outputs from the forward pass and the
	 * adjoint variable for the output color as input.
	 * Then the adjoint variables are traced backward through the tracing
	 * and accumulated in adj_outputs.
	 * 
	 * \param inputsHost the input settings
	 * \param outputsFromForwardHost the outputs from \ref renderForward
	 * \param adj_color the adjoint of the output color (B*H*W*4)
	 * \param adj_outputs [Out] the adjoint variables of the input settings.
	 */
	static void renderAdjoint(
		const RendererInputsHost& inputsHost,
		const RendererOutputsHost& outputsFromForwardHost,
		const torch::Tensor& adj_color,
		AdjointOutputsHost& adj_outputs);
};

END_RENDERER_NAMESPACE
