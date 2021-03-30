#pragma once

#include "commons.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

BEGIN_RENDERER_NAMESPACE

struct MY_API ShadingSettings
{
	float3 ambientLightColor = make_float3(0.1f,0.1f,0.1f);
	float3 diffuseLightColor = make_float3(0.8f,0.8f,0.8f);
	float3 specularLightColor = make_float3(0.1f,0.1f,0.1f);
	float specularExponent = 16;
	float3 materialColor = make_float3(1.0f, 1.0f, 1.0f);
	float aoStrength = 0;

	///the light direction
	/// renderer: world space
	/// post-shading: screen space
	float3 lightDirection = make_float3(0,0,1);
};

struct MY_API [[deprecated("Replaced by RendererInputsHost")]] RendererArgs
{
	//The mipmap level, 0 means the original level
	int mipmapLevel = 0;
	enum RenderMode
	{
		ISO, //mask, normal, depth, flow
		DVR //mask(alpha), rgb
	};
	RenderMode renderMode = ISO;
	
	int cameraResolutionX = 512;
	int cameraResolutionY = 512;
	double cameraFovDegrees = 45;
	//Viewport (startX, startY, endX, endY)
	//special values endX=endY=-1 delegate to cameraResolutionX and cameraResolutionY
	int4 cameraViewport = make_int4(0, 0, -1, -1);
	float3 cameraOrigin = make_float3(0, 0, -1);
	float3 cameraLookAt = make_float3(0, 0, 0);
	float3 cameraUp = make_float3(0, 1, 0);

	float nearClip = 0.1f;
	float farClip = 10.0f;

	double isovalue = 0.5;
	int binarySearchSteps = 5;
	double stepsize = 0.5;
	enum VolumeFilterMode
	{
		TRILINEAR,
		TRICUBIC,
		_VOLUME_FILTER_MODE_COUNT_
	};
	VolumeFilterMode volumeFilterMode = TRILINEAR;

	int aoSamples = 0;
	double aoRadius = 0.1;
	double aoBias = 1e-4;

	//TF Editor
	std::vector<float> densityAxisOpacity;
	std::vector<float> opacityAxis;
	std::vector<float> densityAxisColor;
	std::vector<float3> colorAxis;
	float opacityScaling = 1.0f;
	float minDensity = 0.0f;
	float maxDensity = 1.0f;

	//shading
	ShadingSettings shading;
	bool dvrUseShading = false;
};

END_RENDERER_NAMESPACE

#ifdef _WIN32
#pragma warning( pop )
#endif
