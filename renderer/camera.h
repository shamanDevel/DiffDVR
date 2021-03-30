#pragma once

#include "commons.h"

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "glm/gtc/quaternion.hpp"
#include <torch/types.h>

#include "renderer_settings.cuh"
#include "renderer_utils.cuh"

BEGIN_RENDERER_NAMESPACE

class MY_API Camera
{
public:

	/**
	 * \brief Computes the perspective camera matrices
	 * \param cameraOrigin camera origin / eye pos
	 * \param cameraLookAt look at / target
	 * \param cameraUp up vector
	 * \param fovDegrees vertical field-of-views in degree
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param nearClip the near clipping plane
	 * \param farClip the far clipping plane
	 * \param viewMatrixOut view-projection matrix in Row Major order (viewMatrixOut[0] is the first row), [OUT]
	 * \param viewMatrixInverseOut inverse view-projection matrix in Row Major order (viewMatrixInverseOut[0] is the first row), [OUT]
	 * \param normalMatrixOut normal matrix in Row Major order (normalMatrixOut[0] is the first row), [OUT]
	 */
	static void computeMatrices(
		float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp, 
		float fovDegrees, int width, int height, float nearClip, float farClip,
		glm::mat4& viewMatrixOut, glm::mat4& viewMatrixInverseOut, glm::mat4& normalMatrixOut
	);
	/**
	 * \brief Computes the inverse view-projection matrix.
	 * Not differentiable!
	 * \param cameraOrigin camera origin / eye pos
	 * \param cameraLookAt look at / target
	 * \param cameraUp up vector
	 * \param fovDegrees vertical field-of-views in degree
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param nearClip the near clipping plane
	 * \param farClip the far clipping plane
	 */
	static glm::mat4 computeInverseViewProjectionMatrix(
		float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp,
		float fovDegrees, int width, int height, float nearClip, float farClip);

	enum Orientation
	{
		Xp, Xm, Yp, Ym, Zp, Zm
	};
	static const char* OrientationNames[6];
	static const float3 OrientationUp[6];
	static const int3 OrientationPermutation[6];
	static const bool OrientationInvertYaw[6];
	static const bool OrientationInvertPitch[6];
	
	// PyTorch part

	/**
	 * \brief Constructs the viewport matrix for \ref generateRays
	 * from the camera origin look-at target and up vector.
	 * \param origin the camera origin / eye pos, shape B*3
	 * \param lookAt look at / target, shape B*3
	 * \param cameraUp up vector (normalized), shape B*3
	 * \return the viewport matrix, shape B*3*3
	 */
	static torch::Tensor viewportFromLookAt(
		const torch::Tensor& origin,
		const torch::Tensor& lookAt,
		const torch::Tensor& cameraUp);

	/**
	 * \brief Constructs the viewport matrix for \ref generateRays
	 * from a position on the sphere around a center.
	 * \param center the center of the sphere, shape B*3
	 * \param yawRadians the yaw around the sphere, shape B*1
	 * \param pitchRadians the pitch around the sphere, shape B*1
	 * \param distance the distance to the center, shape B*1
	 * \param orientation the orientation for the up-vector
	 * \return the viewport matrix, shape B*3*3
	 */
	static torch::Tensor viewportFromSphere(
		const torch::Tensor& center,
		const torch::Tensor& yawRadians,
		const torch::Tensor& pitchRadians,
		const torch::Tensor& distance,
		Orientation orientation);
	
	/**
	 * \brief Generates per-pixel rays from the given camera matrix.
	 * This function is differentiable / implements the Autograd protocol.
	 * 
	 * \param viewport the viewport matrix of shape B*3*3
	 *   where viewport[:,0,:] is the camera/eye position,
	 *   viewport[:,1,:] the right vector and
	 *   viewport[:,2,:] the up vector.
	 * \param fovY the field of view for the y axis, in radians.
	 * \param screenWidth the screen width
	 * \param screenHeight the screen height
	 * \return a tuple with the entries:
	 *  - ray_start: the start positions of the rays of shape B*H*W*3
	 *  - ray_dir: the directions of the rays of shape B*H*W*3
	 *  The returned tuple can be directly cast to RendererInputsHost::CameraPerPixelRays
	 */
	static std::tuple<torch::Tensor, torch::Tensor> generateRays(
		const torch::Tensor& viewport,
		float fovY,
		int screenWidth, int screenHeight);
};

END_RENDERER_NAMESPACE


namespace kernel
{
	void generateRaysForward(
		const Tensor3Read& viewport,
		float fovY, int B, int H, int W,
		Tensor4RW& rayStart, Tensor4RW& rayDir,
		bool cuda);

	void generateRaysAdjoint(
		const Tensor3Read& viewport,
		float fovY, int B, int H, int W,
		const Tensor4Read& adj_rayStart, const Tensor4Read& adj_rayDir,
		Tensor3RW& adj_viewport,
		bool cuda);
	
}

