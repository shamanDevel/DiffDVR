#pragma once

#include <cuda_runtime.h>
#include <lib.h>
#include <GL/glew.h>
#include <sstream>
#include <third-party/Eigen/Core> // in cuMat, for scene network

#include "quad_drawer.h"
#include "tf_editor.h"
#include "camera_gui.h"
#include "visualizer_kernels.h"
#include "background_worker.h"

#include <renderer_settings.cuh>

struct GLFWwindow;

class Visualizer
{
public:
	Visualizer(GLFWwindow* window);
	~Visualizer();

	void specifyUI();

	void render(int display_w, int display_h);

	ImVec4 clearColor_ = ImVec4(1, 1, 1, 1);
	
private:

	enum RedrawMode
	{
		RedrawNone,
		RedrawPost,
		RedrawRenderer,

		_RedrawModeCount_
	};
	static const char* RedrawModeNames[_RedrawModeCount_];
	RedrawMode redrawMode_ = RedrawNone;

	GLFWwindow* window_;

	//volume
	enum class VolumeInput
	{
		SYNTHETIC,
		FILE,
		__COUNT__
	};
	static const char* VolumeInputNames[int(VolumeInput::__COUNT__)];
	VolumeInput volumeInput_ = VolumeInput::SYNTHETIC;
	int syntheticDatasetIndex_ = 0;
	int syntheticDatasetResolutionPower_ = 6;
	std::string volumeDirectory_;
	std::string volumeFilename_;
	std::string volumeFullFilename_;
	std::unique_ptr<renderer::Volume> syntheticVolume_;
	std::unique_ptr<renderer::Volume> fileVolume_;
	static constexpr int MipmapLevels[] = { 0, 1, 2, 3, 7 };
	int volumeMipmapLevel_ = 0;
	
	//camera, not much to do here
	CameraGui cameraGui_;

	//background computation
	BackgroundWorker worker_;
	std::function<void()> backgroundGui_;

	//information string that is displayed together with the FPS
	//It is cleared every frame. Use it to append per-frame information
	std::stringstream extraFrameInformation_;

	//rendering
	bool runOnCUDA_ = true;
	kernel::TFMode tfMode_ = kernel::TFLinear;
	static const char* TFModeNames[kernel::__TFModeCount__];
	float stepSize_ = 0.5f;
	kernel::VolumeFilterMode volumeFilterMode_ = kernel::FilterTrilinear;
	static const char* VolumeFilterModeNames[kernel::__VolumeFilterModeCount__];
	kernel::BlendMode blendMode_ = kernel::BlendBeerLambert;
	static const char* BlendModeNames[kernel::__BlendModeCount__];

	//display
	int displayWidth_ = 0;
	int displayHeight_ = 0;
	unsigned int screenTextureGL_ = 0;
	cudaGraphicsResource_t screenTextureCuda_ = nullptr;
	GLubyte* screenTextureCudaBuffer_ = nullptr;
	QuadDrawer drawer_;

	//dvr
	TfPiecewiseLinearEditor editorLinear_;
	TfTextureEditor editorTexture_;
	TfGaussianEditor editorGaussian_;
	std::string tfDirectory_;
	float minDensity_{ 0.0f };
	float maxDensity_{ 1.0f };
	float opacityScaling_{ 50.0f };
	bool showColorControlPoints_{ true };
	bool dvrUseShading_ = false;
	torch::Tensor tfTensorIdentity_;
	torch::Tensor tfTensorLinear_;
	torch::Tensor tfTensorTexture_;
	torch::Tensor tfTensorGaussian_;
	RENDERER_NAMESPACE::Volume::Histogram volumeHistogram_;

	//forward differences - test
	bool fd_enabled = false;
	bool fd_stepsize = true;
	bool fd_camera = false;
	int fd_camera_index = 0; //0,1,2: rayStart xyz; 3,4,5: rayDir xyz
	bool fd_tf = false;
	int fd_tf_point = 0; //index of the point to inspect
	int fd_tf_channel = 0; //index of the channel to inspect
	float fd_minGradient = 0;
	float fd_maxGradient = 0;
	int fd_outputChannel = 0; //red, green, blue, alpha, difference
	bool fd_recordReference = false;
	float fd_avgLoss = 0;
	float fd_avgGradient = 0;
	torch::Tensor forwardDifferencesReference;
	torch::Tensor forwardDifferencesOutput; //B*H*W*D*C (B=1, D=1, C=4)

	//intermediate computation results
	renderer::RendererOutputsHost rendererOutputsCpu_;
	renderer::RendererOutputsHost rendererOutputsGpu_;
	GLubyte* postOutput_ = nullptr;

	//shading
	float3 ambientLightColor{ 0.1, 0.1, 0.1 };
	float3 diffuseLightColor{ 0.8, 0.8, 0.8 };
	float3 specularLightColor{ 0.1, 0.1, 0.1 };
	float specularExponent = 16;
	float3 materialColor{ 1.0, 1.0, 1.0 };
	float aoStrength = 0.5;
	float3 lightDirectionScreen{ 0,0,+1 };

	//screenshot
	std::string screenshotString_;
	float screenshotTimer_ = 0;

	//settings
	std::string settingsDirectory_;
	enum SettingsToLoad
	{
		CAMERA = 1 << 0,
		COMPUTATION_MODE = 1 << 1,
		TF_EDITOR = 1 << 2,
		RENDERER = 1 << 3,
		SHADING = 1 << 4,
		DATASET = 1 << 5,
		_ALL_SETTINGS_ = 0x0fffffff
	};
	int settingsToLoad_ = _ALL_SETTINGS_;

private:
	void releaseResources();
	
	void settingsSave();
	void settingsLoad();
	
	void loadVolumeDialog();
	void loadVolume(const std::string& filename, float* progress = nullptr);
	renderer::Volume* getCurrentVolume() const;
	void selectMipmapLevel(int level);
	
	void uiMenuBar();
	void uiVolume();
	void uiCamera();
	void uiRenderer();
	void uiTfEditor();
	void uiComputationMode();
	void uiShading();
	void uiScreenshotOverlay();
	void uiFPSOverlay();
	void uiForwardDifferences();

	void updateTFTensors(kernel::TFMode tfMode);
	renderer::RendererInputsHost setupRendererArgs();
	renderer::ForwardDifferencesSettingsHost setupForwardDifferencesArgs(
		const renderer::RendererInputsHost& inputs);
	void renderImpl();
	void copyBufferToOpenGL();
	void resize(int display_w, int display_h);
	void triggerRedraw(RedrawMode mode);

	void screenshot();

	static float3 computeCenterOfMass(
		const renderer::Volume* volume,
		const torch::Tensor& tfTensorTextureCpu);
};

