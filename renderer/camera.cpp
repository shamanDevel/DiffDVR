#include "camera.h"

#include <iostream>
#include <iomanip>
#include <torch/torch.h>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtc/matrix_access.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/string_cast.hpp"

#include "pytorch_utils.h"

namespace std
{
	std::ostream& operator<<(std::ostream& o, const float3& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::vec3 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::vec4 v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
	std::ostream& operator<<(std::ostream& o, const glm::mat4 m)
	{
		o << m[0] << "\n" << m[1] << "\n" << m[2] << "\n" << m[3];
		return o;
	}
}

BEGIN_RENDERER_NAMESPACE

namespace{
	// copy of glm::perspectiveFovLH_ZO, seems to not be defined in unix
	glm::mat4 perspectiveFovLH_ZO(float fov, float width, float height, float zNear, float zFar)
	{
		assert(width > static_cast<float>(0));
		assert(height > static_cast<float>(0));
		assert(fov > static_cast<float>(0));

		float const rad = fov;
		float const h = glm::cos(static_cast<float>(0.5) * rad) / glm::sin(static_cast<float>(0.5) * rad);
		float const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

		glm::mat4 Result(static_cast<float>(0));
		Result[0][0] = w;
		Result[1][1] = h;
		Result[2][2] = zFar / (zFar - zNear);
		Result[2][3] = static_cast<float>(1);
		Result[3][2] = -(zFar * zNear) / (zFar - zNear);
		return Result;
	}
}

const char* Camera::OrientationNames[6] = {
	"Xp", "Xm", "Yp", "Ym", "Zp", "Zm"
};
const float3 Camera::OrientationUp[6] = {
	float3{1,0,0}, float3{-1,0,0},
	float3{0,1,0}, float3{0,-1,0},
	float3{0,0,1}, float3{0,0,-1}
};
const int3 Camera::OrientationPermutation[6] = {
	int3{2,-1,-3}, int3{-2, 1, 3},
	int3{1,2,3}, int3{-1,-2,-3},
	int3{-3,-1,2}, int3{3,1,-2}
};
const bool Camera::OrientationInvertYaw[6] = {
	true, false, false, true, false, true
};
const bool Camera::OrientationInvertPitch[6] = {
	false, false, false, false, false, false
};

void Camera::computeMatrices(float3 cameraOrigin_, float3 cameraLookAt_, float3 cameraUp_, float fovDegrees,
	int width, int height, float nearClip, float farClip, glm::mat4& viewMatrixOut,
	glm::mat4& viewMatrixInverseOut, glm::mat4& normalMatrixOut)
{
	const glm::vec3 cameraOrigin = *reinterpret_cast<glm::vec3*>(&cameraOrigin_.x);
	const glm::vec3 cameraLookAt = *reinterpret_cast<glm::vec3*>(&cameraLookAt_.x);
	const glm::vec3 cameraUp = *reinterpret_cast<glm::vec3*>(&cameraUp_.x);

	float fovRadians = glm::radians(fovDegrees);

	glm::mat4 viewMatrix = glm::lookAtLH(cameraOrigin, cameraLookAt, normalize(cameraUp));
	glm::mat4 projMatrix = perspectiveFovLH_ZO(fovRadians, float(width), float(height), nearClip, farClip);

	glm::mat4 viewProjMatrix = projMatrix * viewMatrix;
	glm::mat4 invViewProjMatrix = glm::inverse(viewProjMatrix);
	glm::mat4 normalMatrix = glm::inverse(glm::transpose(glm::mat4(glm::mat3(viewMatrix))));

	viewProjMatrix = glm::transpose(viewProjMatrix);
	invViewProjMatrix = glm::transpose(invViewProjMatrix);
	normalMatrix = glm::transpose(normalMatrix);
	normalMatrix[0] = -normalMatrix[0]; //somehow, the networks were trained with normal-x inverted
	viewMatrixOut = viewProjMatrix;
	viewMatrixInverseOut = invViewProjMatrix;
	normalMatrixOut = normalMatrix;
}

glm::mat4 Camera::computeInverseViewProjectionMatrix(float3 cameraOrigin, float3 cameraLookAt, float3 cameraUp,
	float fovDegrees, int width, int height, float nearClip, float farClip)
{
	//std::cout << "origin: " << cameraOrigin << ", look-at: " << cameraLookAt
	//	<< ", up: " << cameraUp << ", fov: " << fovDegrees << ", width: " << width
	//	<< ", height: " << height << ", near-clip: " << nearClip << ", far-clip: " << farClip
	//	<< std::endl;
	glm::mat4 viewMatrix, viewMatrixInverse, normalMatrix;
	computeMatrices(cameraOrigin, cameraLookAt, cameraUp, fovDegrees, width, height,
		nearClip, farClip, viewMatrix, viewMatrixInverse, normalMatrix);
	//std::cout << "matrix:\n" << glm::to_string(viewMatrixInverse) << std::endl;
	return viewMatrixInverse;
}

torch::Tensor Camera::viewportFromLookAt(const torch::Tensor& origin, const torch::Tensor& lookAt,
	const torch::Tensor& cameraUp)
{
	//early error checking
	CHECK_DIM(origin, 2);
	CHECK_SIZE(origin, 1, 3);
	int B = origin.size(0);
	CHECK_DIM(lookAt, 2);
	CHECK_SIZE(lookAt, 1, 3);
	B = CHECK_BATCH(lookAt, B);
	CHECK_DIM(cameraUp, 2);
	CHECK_SIZE(cameraUp, 1, 3);
	CHECK_BATCH(cameraUp, B);
	
	const auto front = torch::nn::functional::normalize(lookAt - origin);
	const auto right = torch::nn::functional::normalize(
		torch::cross(front, cameraUp, 1));
	const auto up2 = torch::nn::functional::normalize(
		torch::cross(right, front, 1));
	return torch::stack({ origin, right, up2 }, 1);
}

torch::Tensor Camera::viewportFromSphere(const torch::Tensor& center, const torch::Tensor& yawRadians, 
	const torch::Tensor& pitchRadians, const torch::Tensor& distance, Orientation orientation)
{
	CHECK_DIM(center, 2);
	CHECK_SIZE(center, 1, 3);
	int B = center.size(0);
	
	CHECK_DIM(yawRadians, 2);
	CHECK_SIZE(yawRadians, 1, 1);
	B = CHECK_BATCH(yawRadians, B);

	CHECK_DIM(pitchRadians, 2);
	CHECK_SIZE(pitchRadians, 1, 1);
	B = CHECK_BATCH(pitchRadians, B);

	CHECK_DIM(distance, 2);
	CHECK_SIZE(distance, 1, 1);
	B = CHECK_BATCH(distance, B);
	
	const auto yaw = yawRadians * (OrientationInvertYaw[orientation] ? +1.0f : -1.0f);
	const auto pitch = pitchRadians * (OrientationInvertPitch[orientation] ? +1.0f : -1.0f);
	//std::cout << "yaw=" << yaw << ", pitch=" << pitch << std::endl;
	torch::Tensor pos[] =
	{
		torch::cos(pitch) * torch::cos(yaw) * distance,
		torch::sin(pitch)* distance,
		torch::cos(pitch) * torch::sin(yaw) * distance
	};
	//std::cout << "pos1: " << pos[0] << ", " << pos[1] << ", " << pos[2] << std::endl;
	torch::Tensor pos2[3];
	for (int i = 0; i < 3; ++i)
	{
		int p = (&renderer::Camera::OrientationPermutation[orientation].x)[i];
		pos2[i] = pos[std::abs(p) - 1] * (p > 0 ? 1 : -1);
	}
	auto origin = torch::cat({ pos2[0], pos2[1], pos2[2] }, 1);
	//std::cout << "origin: " << origin << std::endl;

	auto upVector = OrientationUp[orientation];
	auto upVectorTorch = torch::tensor({ {upVector.x, upVector.y, upVector.z} },
		center.options());
	if (B > 1) {
		std::vector<torch::Tensor> l(B);
		for (int b = 0; b < B; ++b) l[b] = upVectorTorch;
		upVectorTorch = torch::cat(l, 0);
	}

	//std::cout << "up: " << upVectorTorch << std::endl;
	return viewportFromLookAt(origin + center, center, upVectorTorch);
}

class GenerateRayFunction : public torch::autograd::Function<GenerateRayFunction> {
public:
	// Note that both forward and backward are static functions

	// bias is an optional argument
	static torch::autograd::variable_list forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& viewport,
		float fovY, int screenWidth, int screenHeight) {

		CHECK_DIM(viewport, 3);
		CHECK_SIZE(viewport, 1, 3);
		CHECK_SIZE(viewport, 2, 3);
		CHECK_DTYPE(viewport, real_dtype);
		int B = viewport.size(0);
		bool cuda = viewport.device().is_cuda();

		ctx->save_for_backward({ viewport });
		ctx->saved_data["fovY"] = fovY;
		ctx->saved_data["screenWidth"] = screenWidth;
		ctx->saved_data["screenHeight"] = screenHeight;

		torch::Tensor rayStart = torch::empty({ B, screenHeight, screenWidth, 3 },
			viewport.options());
		torch::Tensor rayDir = torch::empty({ B, screenHeight, screenWidth, 3 },
			viewport.options());
		auto rayStartAcc = accessor<kernel::Tensor4RW>(rayStart);
		auto rayDirAcc = accessor<kernel::Tensor4RW>(rayDir);
		
		kernel::generateRaysForward(
			accessor<kernel::Tensor3Read>(viewport),
			fovY, B, screenHeight, screenWidth,
			rayStartAcc, rayDirAcc,
			cuda);
		
		return {rayStart, rayDir};
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto viewport = saved[0];
		float fovY = ctx->saved_data["fovY"].toDouble();
		int screenWidth = ctx->saved_data["screenWidth"].toInt();
		int screenHeight = ctx->saved_data["screenHeight"].toInt();

		auto grad_rayStart = grad_outputs[0];
		auto grad_rayDir = grad_outputs[1];
		int B = viewport.size(0);
		bool cuda = viewport.device().is_cuda();

		torch::Tensor grad_viewport = torch::zeros({ B, 3, 3 }, grad_rayStart.options());
		auto grad_viewportAcc = accessor<kernel::Tensor3RW>(grad_viewport);

		kernel::generateRaysAdjoint(
			accessor<kernel::Tensor3Read>(viewport),
			fovY, B, screenHeight, screenWidth,
			accessor<kernel::Tensor4Read>(grad_rayStart), 
			accessor<kernel::Tensor4Read>(grad_rayDir),
			grad_viewportAcc,
			cuda);

		//return torch::Tensor() for non-optimized arguments
		return { grad_viewport, torch::Tensor(), torch::Tensor(), torch::Tensor() };
	}
};

std::tuple<torch::Tensor, torch::Tensor> Camera::generateRays(const torch::Tensor& viewport, float fovY, int screenWidth, int screenHeight)
{
	torch::autograd::tensor_list ret = GenerateRayFunction::apply(viewport, fovY, screenWidth, screenHeight);
	return std::make_tuple(ret[0], ret[1]);
}

END_RENDERER_NAMESPACE
