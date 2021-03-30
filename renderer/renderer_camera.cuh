#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_settings.cuh"

#include "forward.h"
#include "forward_vector.h"

namespace kernel
{
	template<CameraMode cameraMode>
	struct CameraEval
	{
		static __host__ __device__ void computeRays(
			int x, int y, int b,
			const RendererInputs& inputs,
			real3& rayStartOut, real3& rayDirOut)
		{
			if constexpr (cameraMode == CameraRayStartDir)
			{
				rayStartOut = fetchReal3(inputs.cameraRayStart, b, y, x);
				rayDirOut = fetchReal3(inputs.cameraRayDir, b, y, x);
			}
			else if constexpr (cameraMode == CameraInverseViewMatrix)
			{
				const real4 inverseViewMatrix[4] = {
					fetchReal4(inputs.cameraMatrix, b, 0),
					fetchReal4(inputs.cameraMatrix, b, 1),
					fetchReal4(inputs.cameraMatrix, b, 2),
					fetchReal4(inputs.cameraMatrix, b, 3)
				};
				const real_t fx = 2 * (x + real_t(0.5)) / real_t(inputs.screenSize.x) - 1; //NDC in [-1,+1]
				const real_t fy = 2 * (y + real_t(0.5)) / real_t(inputs.screenSize.y) - 1;
				rayStartOut = matmul_proj(inverseViewMatrix, fx, fy, 0);
				const real3 rayEnd = matmul_proj(inverseViewMatrix, fx, fy, real_t(0.9));
				rayDirOut = normalize(rayEnd - rayStartOut);
			}
			else //cameraMode == CameraReferenceFrame
			{
				real_t fovXRadians = inputs.cameraFovYRadians *
					real_t(inputs.screenSize.x) / real_t(inputs.screenSize.y);
				real_t tanFovX = tan(fovXRadians / 2);
				real_t tanFovY = tan(inputs.cameraFovYRadians / 2);

				real3 eye = fetchReal3(inputs.cameraMatrix, b, 0);
				real3 right = fetchReal3(inputs.cameraMatrix, b, 1);
				real3 up = fetchReal3(inputs.cameraMatrix, b, 2);
				real3 front = cross(up, right);

				//to normalized coordinates
				const real_t fx = 2 * (x + 0.5f) / real_t(inputs.screenSize.x) - 1; //NDC in [-1,+1]
				const real_t fy = 2 * (y + 0.5f) / real_t(inputs.screenSize.y) - 1;

				real3 dir = front + fx * tanFovX * right + fy * tanFovY * up;
				dir = normalize(dir);
				rayStartOut = eye;
				rayDirOut = dir;
			}
		}
	};

	template<CameraMode cameraMode, int D, bool HasCameraDerivatives>
	struct CameraEvalForwardGradients;
	
	template<CameraMode cameraMode, int D>
	struct CameraEvalForwardGradients<cameraMode, D, false>
	{
		typedef real3 vector_t;
		static __host__ __device__ void computeRays(
			int x, int y, int b,
			const RendererInputs& inputs,
			const ForwardDifferencesSettings& settings,
			vector_t& rayStartOut, vector_t& rayDirOut)
		{
			CameraEval<cameraMode>::computeRays(x, y, b, inputs, rayStartOut, rayDirOut);
		}
	};
	template<CameraMode cameraMode, int D>
	struct CameraEvalForwardGradients<cameraMode, D, true>
	{
		typedef cudAD::fvar<real3, D> vector_t;
		static __host__ __device__ void computeRays(
			int x, int y, int b,
			const RendererInputs& inputs,
			const ForwardDifferencesSettings& settings,
			vector_t& rayStartOut, vector_t& rayDirOut)
		{
			real3 rayStart, rayDir;
			if constexpr (cameraMode == CameraRayStartDir)
			{
				rayStart = fetchReal3(inputs.cameraRayStart, b, y, x);
				rayDir = fetchReal3(inputs.cameraRayDir, b, y, x);
			}
			else if constexpr (cameraMode == CameraInverseViewMatrix)
			{
				const real4 inverseViewMatrix[4] = {
					fetchReal4(inputs.cameraMatrix, b, 0),
					fetchReal4(inputs.cameraMatrix, b, 1),
					fetchReal4(inputs.cameraMatrix, b, 2),
					fetchReal4(inputs.cameraMatrix, b, 3)
				};
				const real_t fx = 2 * (x + 0.5f) / real_t(inputs.screenSize.x) - 1; //NDC in [-1,+1]
				const real_t fy = 2 * (y + 0.5f) / real_t(inputs.screenSize.y) - 1;
				rayStart = matmul_proj(inverseViewMatrix, fx, fy, 0);
				const real3 rayEnd = matmul_proj(inverseViewMatrix, fx, fy, 0.9);
				rayDir = normalize(rayEnd - rayStart);
			}
			else
			{
				real_t fovXRadians = inputs.cameraFovYRadians *
					real_t(inputs.screenSize.x) / real_t(inputs.screenSize.y);
				real_t tanFovX = tan(fovXRadians / 2);
				real_t tanFovY = tan(inputs.cameraFovYRadians / 2);

				rayStart = fetchReal3(inputs.cameraMatrix, b, 0);
				real3 right = fetchReal3(inputs.cameraMatrix, b, 1);
				real3 up = fetchReal3(inputs.cameraMatrix, b, 2);
				real3 front = cross(up, right);

				//to normalized coordinates
				const real_t fx = 2 * (x + 0.5f) / real_t(inputs.screenSize.x) - 1; //NDC in [-1,+1]
				const real_t fy = 2 * (y + 0.5f) / real_t(inputs.screenSize.y) - 1;

				real3 dir = front + fx * tanFovX * right + fy * tanFovY * up;
				rayDir = normalize(dir);
			}
			rayStartOut = cudAD::make_real3in<D>(rayStart, settings.d_rayStart);
			rayDirOut = cudAD::make_real3in<D>(rayDir, settings.d_rayDir);
		}
	};

}
