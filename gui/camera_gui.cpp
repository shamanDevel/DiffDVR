#include "camera_gui.h"

#include "imgui/imgui.h"
#include <cmath>
#include <glm/glm.hpp>
#include "utils.h"

#include <helper_math.cuh>
#include <camera.h>

bool CameraGui::specifyUI()
{
	bool changed = false;

	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	float fovMin = 0.1, fovMax = 90;
	if (ImGui::SliderScalar("FoV", ImGuiDataType_Float, &fov_, &fovMin, &fovMax, u8"%.5f\u00b0", 2)) changed = true;
	ImGui::InputFloat3("Camera Origin", &origin.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
	if (ImGui::InputFloat3("Camera Look At", &lookAt_.x)) changed = true;
	ImGui::InputFloat3("Camera Up", &up.x);

	for (int i = 0; i < 6; ++i) {
		if (ImGui::RadioButton(renderer::Camera::OrientationNames[i], orientation_ == renderer::Camera::Orientation(i))) {
			orientation_ = renderer::Camera::Orientation(i);
			changed = true;
		}
		if (i<5) ImGui::SameLine();
	}
	
	float minPitch = -80, maxPitch = +80;
	if (ImGui::SliderScalar("Pitch", ImGuiDataType_Float, &currentPitch_, &minPitch, &maxPitch, u8"%.5f\u00b0")) changed = true;
	if (ImGui::InputFloat("Yaw", &currentYaw_, 0, 0, u8"%.5f\u00b0")) changed = true;

	if (ImGui::InputFloat("Zoom", &zoomvalue_)) changed = true;
	ImGui::InputFloat("Distance", &distance, 0, 0, ".3f", ImGuiInputTextFlags_ReadOnly);

	return changed;
}

bool CameraGui::updateMouse()
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) return false;

	if (io.MouseDown[0])
	{
		//dragging
		currentPitch_ = std::max(-80.0f, std::min(80.0f, 
			currentPitch_ + rotateSpeed_ * io.MouseDelta.y));
		currentYaw_ += rotateSpeed_ * io.MouseDelta.x;
	}
	//zoom
	mouseWheel_ = ImGui::GetIO().MouseWheel;
	zoomvalue_ += mouseWheel_;

	bool changed = mouseWheel_ != 0 || (io.MouseDown[0] && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0));
	mouseWheel_ = 0;
	return changed;
}

void CameraGui::updateRenderArgs(renderer::RendererArgs& args) const
{
	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	args.cameraOrigin = origin;
	args.cameraFovDegrees = fov_;
	args.cameraLookAt = lookAt_;
	args.cameraUp = up;
}

glm::mat4 CameraGui::getInverseViewProjectionMatrix(int screenWidth, int screenHeight) const
{
	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	glm::mat4 viewMatrix, invViewMatrix, normalMatrix;
	const float nearClip = 0.1;
	const float farClip = 10.0;
	renderer::Camera::computeMatrices(origin, lookAt_, up, fov_,
		screenWidth, screenHeight, nearClip, farClip,
		viewMatrix, invViewMatrix, normalMatrix);

	return invViewMatrix;
}

float3 CameraGui::screenToWorld(const float3& screenDirection) const
{
	float3 origin, up;
	float distance;
	computeParameters(origin, up, distance);

	float3 viewDir = normalize(lookAt_ - origin);
	up = normalize(up - dot(viewDir, up)*viewDir);
	float3 right = cross(up, viewDir);

	return right * screenDirection.x +
		up * screenDirection.y +
		viewDir * screenDirection.z;
}

nlohmann::json CameraGui::toJson() const
{
	return {
		{"orientation", orientation_},
		{"lookAt", lookAt_},
		{"rotateSpeed", rotateSpeed_},
		{"zoomSpeed", zoomSpeed_},
		{"fov", fov_},
		{"currentPitch", currentPitch_},
		{"currentYaw", currentYaw_},
		{"zoomValue", zoomvalue_}
	};
}

void CameraGui::fromJson(const nlohmann::json& s)
{
	orientation_ = s.at("orientation").get<renderer::Camera::Orientation>();
	lookAt_ = s.at("lookAt").get<float3>();
	rotateSpeed_ = s.at("rotateSpeed").get<float>();
	zoomSpeed_ = s.at("zoomSpeed").get<float>();
	fov_ = s.at("fov").get<float>();
	currentPitch_ = s.at("currentPitch").get<float>();
	currentYaw_ = s.at("currentYaw").get<float>();
	zoomvalue_ = s.at("zoomValue").get<float>();
}

void CameraGui::computeParameters(float3& origin, float3& up, float& distance) const
{
	distance = baseDistance_ * std::pow(zoomSpeed_, zoomvalue_);
	up = renderer::Camera::OrientationUp[orientation_];

	float yaw = glm::radians(!renderer::Camera::OrientationInvertYaw[orientation_] ? -currentYaw_ : +currentYaw_);
	float pitch = glm::radians(!renderer::Camera::OrientationInvertPitch[orientation_] ? -currentPitch_ : +currentPitch_);
	float pos[3];
	pos[1] = std::sin(pitch) * distance;
	pos[0] = std::cos(pitch) * std::cos(yaw) * distance;
	pos[2] = std::cos(pitch) * std::sin(yaw) * distance;
	float pos2[3];
	for (int i=0; i<3; ++i)
	{
		int p = (&renderer::Camera::OrientationPermutation[orientation_].x)[i];
		pos2[i] = pos[std::abs(p) - 1] * (p > 0 ? 1 : -1);
	}
	origin = make_float3(pos2[0], pos2[1], pos2[2]) + lookAt_;
}
