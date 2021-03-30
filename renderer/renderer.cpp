#include "renderer.h"

#include <ATen/cuda/CUDAContext.h>

#include "renderer_cpu.h"
#include "renderer_cuda.h"
#include "pytorch_utils.h"

bool renderer::Renderer::initCuda()
{
	return RendererCuda::Instance().initCuda();
}

void renderer::Renderer::setCudaCacheDir(const std::filesystem::path& path)
{
	RendererCuda::Instance().setCudaCacheDir(path);
}

void renderer::Renderer::reloadCudaKernels()
{
	RendererCuda::Instance().reloadCudaKernels();
}

void renderer::Renderer::disableCudaCache()
{
	RendererCuda::Instance().disableCudaCache();
}

void renderer::Renderer::cleanupCuda()
{
	RendererCuda::Instance().cleanup();
}

void renderer::Renderer::setCudaSyncMode(bool sync)
{
	RendererCuda::Instance().setSyncMode(sync);
}
bool renderer::Renderer::getCudaSyncMode()
{
	return RendererCuda::Instance().getSyncMode();
}

void renderer::Renderer::setCudaDebugMode(bool debug)
{
	RendererCuda::Instance().setDebugMode(debug);
}

typedef std::vector<torch::Tensor> TensorsToKeepAlive_t;
std::tuple<kernel::RendererInputs, TensorsToKeepAlive_t>
	renderer::Renderer::checkInputs(const renderer::RendererInputsHost& inputsHost,
                int& B, int& W, int& H, int& X, int& Y, int& Z, bool& cuda,
                bool ignoreVolume)
{
	TensorsToKeepAlive_t tensorsToKeepAlive;
	
	//check inputs
	const auto cpuTensorOptions = at::TensorOptions().device(c10::kCPU).dtype(real_dtype);
	const auto deviceTensorOptions = at::TensorOptions().device(inputsHost.volume.device()).dtype(real_dtype);
	W = inputsHost.screenSize.x;
	H = inputsHost.screenSize.y;

	const torch::Tensor& volume = inputsHost.volume;
	CHECK_DIM(volume, 4);
	cuda = volume.is_cuda();
	if (!ignoreVolume) {
		CHECK_DTYPE(volume, real_dtype);
		B = volume.size(0);
		X = volume.size(1);
		Y = volume.size(2);
		Z = volume.size(3);

		TORCH_CHECK((inputsHost.volumeFilterMode == kernel::FilterPreshaded) ==
			(inputsHost.tfMode == kernel::TFPreshaded),
			"if pre-shading is selected, both the filter mode and TF mode must be set to preshaded");
		if (inputsHost.volumeFilterMode == kernel::FilterPreshaded)
		{
			TORCH_CHECK(volume.size(0) == 4,
				"In pre-shaded mode, the volume colors are encoded in the batch dimension of the volume. It must be 4, but is ", volume.size(0));
			B = 1;
		}
	} else
	{
		B = 1;
		X = Y = Z = 0;
	}
	

	torch::Tensor boxMin, boxSize;
	if (inputsHost.boxMin.index() == 0)
	{
		boxMin = std::get<0>(inputsHost.boxMin);
		CHECK_DIM(boxMin, 2);
		CHECK_CUDA(boxMin, cuda);
		CHECK_SIZE(boxMin, 1, 3);
		B = CHECK_BATCH(boxMin, B);
		CHECK_DTYPE(boxMin, real_dtype);
	}
	else
	{
		real3 boxMin_v = std::get<1>(inputsHost.boxMin);
		boxMin = torch::tensor({ {boxMin_v.x, boxMin_v.y, boxMin_v.z} },
			cpuTensorOptions).to(volume.device());
		tensorsToKeepAlive.push_back(boxMin);
	}
	if (inputsHost.boxSize.index() == 0)
	{
		boxSize = std::get<0>(inputsHost.boxSize);
		CHECK_DIM(boxSize, 2);
		CHECK_CUDA(boxSize, cuda);
		CHECK_SIZE(boxSize, 1, 3);
		B = CHECK_BATCH(boxSize, B);
		CHECK_DTYPE(boxSize, real_dtype);
	}
	else
	{
		real3 boxSize_v = std::get<1>(inputsHost.boxSize);
		boxSize = torch::tensor({{boxSize_v.x, boxSize_v.y, boxSize_v.z}},
			cpuTensorOptions).to(volume.device());
		tensorsToKeepAlive.push_back(boxSize);
	}

	torch::Tensor cameraRayStart, cameraRayDir, cameraMatrix;
	real_t cameraFovYRadians = 0;
	switch (inputsHost.cameraMode)
	{
	case kernel::CameraRayStartDir:
	{
		TORCH_CHECK(inputsHost.camera.index() == 0, "'CameraRayStartDir' specified in 'cameraMode', but 'camera' does not hold the first variant 'CameraPerPixelRays'");
		cameraRayStart = std::get<0>(inputsHost.camera).cameraRayStart;
		cameraRayDir = std::get<0>(inputsHost.camera).cameraRayDir;

		CHECK_DIM(cameraRayStart, 4);
		CHECK_CUDA(cameraRayStart, cuda);
		CHECK_SIZE(cameraRayStart, 1, H);
		CHECK_SIZE(cameraRayStart, 2, W);
		CHECK_SIZE(cameraRayStart, 3, 3);
		B = CHECK_BATCH(cameraRayStart, B);
		CHECK_DTYPE(cameraRayStart, real_dtype);

		CHECK_DIM(cameraRayDir, 4);
		CHECK_CUDA(cameraRayDir, cuda);
		CHECK_SIZE(cameraRayDir, 1, H);
		CHECK_SIZE(cameraRayDir, 2, W);
		CHECK_SIZE(cameraRayDir, 3, 3);
		B = CHECK_BATCH(cameraRayDir, B);
		CHECK_DTYPE(cameraRayDir, real_dtype);

		cameraMatrix = torch::empty({ 0,0,0 }, deviceTensorOptions);
		tensorsToKeepAlive.push_back(cameraMatrix);
	} break;
	case kernel::CameraInverseViewMatrix:
	{
		TORCH_CHECK(inputsHost.camera.index() > 0, "'CameraInverseViewMatrix' specified in 'cameraMode', but 'camera' holds the first variant 'CameraPerPixelRays'");
		cameraRayStart = torch::empty({ 0,0,0,0 }, deviceTensorOptions);
		cameraRayDir = torch::empty({ 0,0,0,0 }, deviceTensorOptions);
		tensorsToKeepAlive.push_back(cameraRayStart);
		tensorsToKeepAlive.push_back(cameraRayDir);
		if (inputsHost.camera.index() == 1)
		{
			cameraMatrix = std::get<1>(inputsHost.camera);
			CHECK_DIM(cameraMatrix, 3);
			CHECK_CUDA(cameraMatrix, cuda);
			CHECK_SIZE(cameraMatrix, 1, 4);
			CHECK_SIZE(cameraMatrix, 2, 4);
			B = CHECK_BATCH(cameraMatrix, B);
			CHECK_DTYPE(cameraMatrix, real_dtype);
		}
		else //index()==2
		{
			glm::mat4 m4_1 = std::get<2>(inputsHost.camera);
#if USE_DOUBLE_PRECISION==1
			using rmat4 = glm::mat<4, 4, real_t, glm::defaultp>;
#else
			using rmat4 = glm::mat4;
#endif
			rmat4 m4_2 = m4_1;
			cameraMatrix = torch::from_blob(static_cast<void*>(&m4_2[0].x), { 1,4,4 }, cpuTensorOptions).clone().to(volume.device());
			tensorsToKeepAlive.push_back(cameraMatrix);
		}
	} break;
	case kernel::CameraReferenceFrame:
	{
		TORCH_CHECK(inputsHost.camera.index() == 3, "'CameraReferenceFrame' specified in 'cameraMode', but 'camera' does not hold the fourth variant 'CameraReferenceFrame'");
		cameraMatrix = std::get<3>(inputsHost.camera).viewport;
		cameraFovYRadians = std::get<3>(inputsHost.camera).fovYRadians;

		CHECK_DIM(cameraMatrix, 3);
		CHECK_CUDA(cameraMatrix, cuda);
		CHECK_SIZE(cameraMatrix, 1, 3);
		CHECK_SIZE(cameraMatrix, 2, 3);
		B = CHECK_BATCH(cameraMatrix, B);
		CHECK_DTYPE(cameraMatrix, real_dtype);

		cameraRayStart = torch::empty({ 0,0,0,0 }, deviceTensorOptions);
		cameraRayDir = torch::empty({ 0,0,0,0 }, deviceTensorOptions);
		tensorsToKeepAlive.push_back(cameraRayStart);
		tensorsToKeepAlive.push_back(cameraRayDir);
	} break;
	default:
		throw std::runtime_error("unknown camera enum value");
	}

	torch::Tensor stepSize;
	if (inputsHost.stepSize.index() == 0)
	{
		stepSize = std::get<0>(inputsHost.stepSize);
		CHECK_DIM(stepSize, 3);
		CHECK_CUDA(stepSize, cuda);
		B = CHECK_BATCH(stepSize, B);
		checkBatch(stepSize, "stepSize-Height", H, 1);
		checkBatch(stepSize, "stepSize-Width", W, 2);
		CHECK_DTYPE(stepSize, real_dtype);
	}
	else
	{
		real_t stepSize_v = std::get<1>(inputsHost.stepSize);
		stepSize = torch::tensor({{{stepSize_v}}},
			cpuTensorOptions).to(volume.device());
		tensorsToKeepAlive.push_back(stepSize);
	}

	const torch::Tensor& tf = inputsHost.tf;
	CHECK_DIM(tf, 3);
	CHECK_CUDA(tf, cuda);
	B = CHECK_BATCH(tf, B);
	switch (inputsHost.tfMode)
	{
	case kernel::TFIdentity:
		CHECK_SIZE(tf, 1, 1);
		CHECK_SIZE(tf, 2, 2);
		break;
	case kernel::TFTexture:
		CHECK_SIZE(tf, 2, 4);
		break;
	case kernel::TFLinear:
		CHECK_SIZE(tf, 2, 5);
		TORCH_CHECK(tf.size(1) > 1, "tensor 'tf' must have at least two control points in TFLinear-mode");
		break;
	case kernel::TFGaussian:
		CHECK_SIZE(tf, 2, 6);
		break;
	case kernel::TFPreshaded:
		//any size is allowed, TF is ignored
		break;
	default:
		throw std::runtime_error("unknown tf enum value");
	}

	return { {
		make_int2(W, H),
		accessor<kernel::Tensor4Read>(volume),
		make_int3(X, Y, Z),
		accessor<kernel::Tensor2Read>(boxMin),
		accessor<kernel::Tensor2Read>(boxSize),
		accessor<kernel::Tensor4Read>(cameraRayStart),
		accessor<kernel::Tensor4Read>(cameraRayDir),
		accessor<kernel::Tensor3Read>(cameraMatrix),
		cameraFovYRadians,
		accessor<kernel::Tensor3Read>(stepSize),
		accessor<kernel::Tensor3Read>(tf),
		inputsHost.blendingEarlyOut
	}, tensorsToKeepAlive };
}

static kernel::AdjointOutputs checkAdjointOutput(
	renderer::AdjointOutputsHost& adj_outputs,
	bool cuda, int B, int H, int W,int X, int Y, int Z,
	const torch::Tensor& inputsTf = {}, bool preshadedVolume = false)
{
	kernel::BTensor4RW adj_volumeDevice;
	const bool hasVolumeDerivative = adj_outputs.hasVolumeDerivatives;
	if (hasVolumeDerivative)
	{
		torch::Tensor& adj_volume = adj_outputs.adj_volume;
		CHECK_DIM(adj_volume, 4);
		CHECK_CUDA(adj_volume, cuda);
		if (preshadedVolume) {
			CHECK_SIZE(adj_volume, 0, 4);
		}
		else {
			CHECK_BATCH(adj_volume, B);
		}
		CHECK_SIZE(adj_volume, 1, X);
		CHECK_SIZE(adj_volume, 2, Y);
		CHECK_SIZE(adj_volume, 3, Z);
		CHECK_DTYPE(adj_volume, real_dtype);
		adj_volumeDevice = accessor<kernel::BTensor4RW>(adj_volume);
	}

	kernel::BTensor3RW adj_stepSizeDevice;
	const bool hasStepSizeDerivative = adj_outputs.hasStepSizeDerivatives;
	bool stepSizeHasBroadcasting = false;
	if (hasStepSizeDerivative)
	{
		torch::Tensor& adj_stepSize = adj_outputs.adj_stepSize;
		CHECK_DIM(adj_stepSize, 3);
		CHECK_CUDA(adj_stepSize, cuda);
		CHECK_BATCH(adj_stepSize, B);
		checkBatch(adj_stepSize, "adj_stepSize", H, 1);
		checkBatch(adj_stepSize, "adj_stepSize", W, 2);
		CHECK_DTYPE(adj_stepSize, real_dtype);
		adj_stepSizeDevice = accessor<kernel::BTensor3RW>(adj_stepSize);
		stepSizeHasBroadcasting =
			B != adj_stepSize.size(0) ||
			H != adj_stepSize.size(1) ||
			W != adj_stepSize.size(2);
	}

	kernel::BTensor4RW adj_cameraRayStartDevice;
	kernel::BTensor4RW adj_cameraRayDirDevice;
	const bool hasCameraDerivative = adj_outputs.hasCameraDerivatives;
	bool cameraHasBroadcasting = false;
	if (hasCameraDerivative)
	{
		torch::Tensor& adj_cameraRayStart = adj_outputs.adj_cameraRayStart;
		CHECK_DIM(adj_cameraRayStart, 4);
		CHECK_CUDA(adj_cameraRayStart, cuda);
		CHECK_BATCH(adj_cameraRayStart, B);
		CHECK_SIZE(adj_cameraRayStart, 1, H);
		CHECK_SIZE(adj_cameraRayStart, 2, W);
		CHECK_SIZE(adj_cameraRayStart, 3, 3);
		CHECK_DTYPE(adj_cameraRayStart, real_dtype);
		adj_cameraRayStartDevice = accessor<kernel::BTensor4RW>(adj_cameraRayStart);

		torch::Tensor& adj_cameraRayDir = adj_outputs.adj_cameraRayDir;
		CHECK_DIM(adj_cameraRayDir, 4);
		CHECK_CUDA(adj_cameraRayDir, cuda);
		CHECK_BATCH(adj_cameraRayDir, B);
		CHECK_SIZE(adj_cameraRayDir, 1, H);
		CHECK_SIZE(adj_cameraRayDir, 2, W);
		CHECK_SIZE(adj_cameraRayDir, 3, 3);
		CHECK_DTYPE(adj_cameraRayDir, real_dtype);
		adj_cameraRayDirDevice = accessor<kernel::BTensor4RW>(adj_cameraRayDir);

		cameraHasBroadcasting =
			B != adj_cameraRayStart.size(0) ||
			B != adj_cameraRayDir.size(0);
	}

	kernel::BTensor3RW adj_tfDevice;
	const bool hasTFDerivative = adj_outputs.hasTFDerivatives;
	if (hasTFDerivative)
	{
		torch::Tensor& adj_tf = adj_outputs.adj_tf;
		CHECK_DIM(adj_tf, 3);
		CHECK_CUDA(adj_tf, cuda);
		CHECK_BATCH(adj_tf, B);
		if (inputsTf.defined()) {
			CHECK_SIZE(adj_tf, 1, inputsTf.size(1));
			CHECK_SIZE(adj_tf, 2, inputsTf.size(2));
		}
		CHECK_DTYPE(adj_tf, real_dtype);
		adj_tfDevice = accessor<kernel::BTensor3RW>(adj_tf);
	}

	return kernel::AdjointOutputs{
		adj_volumeDevice,
		stepSizeHasBroadcasting,
		adj_stepSizeDevice,
		cameraHasBroadcasting,
		adj_cameraRayStartDevice,
		adj_cameraRayDirDevice,
		adj_tfDevice
	};
}

kernel::RendererOutputs renderer::Renderer::checkOutput(renderer::RendererOutputsHost& outputsHost,
	int B, int H, int W, bool cuda)
{
	torch::Tensor& color = outputsHost.color;
	CHECK_DIM(color, 4);
	CHECK_CUDA(color, cuda);
	CHECK_SIZE(color, 0, B);
	CHECK_SIZE(color, 1, H);
	CHECK_SIZE(color, 2, W);
	CHECK_SIZE(color, 3, 4);
	CHECK_DTYPE(color, real_dtype);

	torch::Tensor& terminationIndex = outputsHost.terminationIndex;
	CHECK_DIM(terminationIndex, 3);
	CHECK_CUDA(terminationIndex, cuda);
	CHECK_SIZE(terminationIndex, 0, B);
	CHECK_SIZE(terminationIndex, 1, H);
	CHECK_SIZE(terminationIndex, 2, W);
	CHECK_DTYPE(terminationIndex, c10::kInt);

	return {
		accessor<kernel::Tensor4RW>(color),
		accessor<kernel::ITensor3RW>(terminationIndex)
	};
}

void renderer::Renderer::renderForward(const RendererInputsHost& inputsHost, RendererOutputsHost& outputsHost)
{
	//check inputs
	int B, W, H, X, Y, Z;
	bool cuda;
	const auto [inputsDevice, tensorsToKeepAlive] =
		checkInputs(inputsHost, B, W, H, X, Y, Z, cuda);

	//check outputs
	kernel::RendererOutputs outputsDevice = checkOutput(outputsHost, B, H, W, cuda);

	//dispatch kernels
	if (cuda)
	{
		RendererCuda::Instance().renderForward(inputsDevice, outputsDevice, B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode);
	} else
	{
		RendererCpu::Instance().renderForward(inputsDevice, outputsDevice, B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode);
	}
	(void)tensorsToKeepAlive;
}

void renderer::Renderer::renderForwardGradients(const RendererInputsHost& inputsHost,
	ForwardDifferencesSettingsHost& differencesSettingsHost, RendererOutputsHost& outputsHost,
	torch::Tensor& gradientsOut)
{
	//check inputs
	int B, W, H, X, Y, Z;
	bool cuda;
	const auto [inputsDevice, tensorsToKeepAlive] =
		checkInputs(inputsHost, B, W, H, X, Y, Z, cuda);

	TORCH_CHECK(differencesSettingsHost.D > 0, "number of derivatives must be positive");
	torch::Tensor d_tf;
	if (differencesSettingsHost.d_tf.defined()) {
		d_tf = differencesSettingsHost.d_tf;
			CHECK_DIM(d_tf, 3);
			CHECK_CUDA(d_tf, cuda);
			CHECK_SIZE(d_tf, 0, inputsHost.tf.size(0));
			CHECK_SIZE(d_tf, 1, inputsHost.tf.size(1));
			CHECK_SIZE(d_tf, 2, inputsHost.tf.size(2));
			CHECK_DTYPE(d_tf, c10::kInt);
	} else
	{
		TORCH_CHECK(differencesSettingsHost.hasTfDerivatives == false,
			"d_tf not defined, hence hasTfDerivatives must be false but was set to true");
		d_tf = torch::empty(inputsHost.tf.sizes(),
			at::TensorOptions().dtype(c10::kInt).device(inputsHost.tf.device()));
	}
	bool hasStepsizeDerivative = differencesSettingsHost.d_stepsize >= 0;
	bool hasCameraDerivative = 
		any(differencesSettingsHost.d_rayStart >= make_int3(0)) ||
		any(differencesSettingsHost.d_rayDir >= make_int3(0));
	bool hasTFDerivative = differencesSettingsHost.hasTfDerivatives;
	kernel::ForwardDifferencesSettings settingsDevice = {
		differencesSettingsHost.d_stepsize,
		differencesSettingsHost.d_rayStart,
		differencesSettingsHost.d_rayDir,
		accessor<kernel::ITensor3Read>(d_tf),
		hasTFDerivative
	};
	TORCH_CHECK(hasStepsizeDerivative || hasCameraDerivative || hasTFDerivative,
		"at least one derivative must be activated");

	//check outputs
	kernel::RendererOutputs outputsDevice = checkOutput(outputsHost, B, H, W, cuda);

	CHECK_DIM(gradientsOut, 5);
	CHECK_CUDA(gradientsOut, cuda);
	CHECK_SIZE(gradientsOut, 0, B);
	CHECK_SIZE(gradientsOut, 1, H);
	CHECK_SIZE(gradientsOut, 2, W);
	CHECK_SIZE(gradientsOut, 3, differencesSettingsHost.D);
	CHECK_SIZE(gradientsOut, 4, 4);
	CHECK_DTYPE(gradientsOut, real_dtype);
	kernel::ForwardDifferencesOutput gradientsDevice = {
		accessor<kernel::Tensor5RW>(gradientsOut)
	};
	
	//dispatch kernels
	if (cuda)
	{
		RendererCuda::Instance().renderForwardGradients(
			inputsDevice, settingsDevice,
			outputsDevice, gradientsDevice,
			B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode,
			differencesSettingsHost.D,
			hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative);
	}
	else
	{
		RendererCpu::Instance().renderForwardGradients(
			inputsDevice, settingsDevice,
			outputsDevice, gradientsDevice,
			B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode,
			differencesSettingsHost.D,
			hasStepsizeDerivative, hasCameraDerivative, hasTFDerivative);
	}
	(void)tensorsToKeepAlive;
}

void renderer::Renderer::forwardVariablesToGradients(const torch::Tensor& forwardVariables,
	const torch::Tensor& gradientOutputColor, ForwardDifferencesSettingsHost& differencesSettingsHost,
	AdjointOutputsHost& adj_outputs)
{
	//forward variables
	CHECK_DIM(forwardVariables, 5);
	bool cuda = forwardVariables.is_cuda();
	int B = forwardVariables.size(0);
	int H = forwardVariables.size(1);
	int W = forwardVariables.size(2);
	int D = forwardVariables.size(3);
	CHECK_SIZE(forwardVariables, 4, 4);
	CHECK_DTYPE(forwardVariables, real_dtype);
	kernel::ForwardDifferencesOutput forwardVariablesDevice = {
		accessor<kernel::Tensor5RW>(forwardVariables)
	};

	//check adjoint color
	CHECK_DIM(gradientOutputColor, 4);
	CHECK_CUDA(gradientOutputColor, cuda);
	CHECK_SIZE(gradientOutputColor, 0, B);
	CHECK_SIZE(gradientOutputColor, 1, H);
	CHECK_SIZE(gradientOutputColor, 2, W);
	CHECK_SIZE(gradientOutputColor, 3, 4);
	CHECK_DTYPE(gradientOutputColor, real_dtype);
	auto gradientOutputColorDevice = accessor<kernel::AdjointColor_t>(gradientOutputColor);

	//differences settings
	kernel::ForwardDifferencesSettings settingsDevice = {
		differencesSettingsHost.d_stepsize,
		differencesSettingsHost.d_rayStart,
		differencesSettingsHost.d_rayDir,
		differencesSettingsHost.hasTfDerivatives
			? accessor<kernel::ITensor3Read>(differencesSettingsHost.d_tf)
			: kernel::ITensor3Read(),
		differencesSettingsHost.hasTfDerivatives
	};

	//check outputs
	TORCH_CHECK(!adj_outputs.hasVolumeDerivatives,
		"volume derivatives are not supported with forward variables");
	auto adj_outputsDevice = checkAdjointOutput(
		adj_outputs, cuda, B, H, W, -1, -1, -1);

	//dispatch kernels
	if (cuda)
	{
		RendererCuda::Instance().forwardVariablesToGradients(
			forwardVariablesDevice, gradientOutputColorDevice,
			settingsDevice, adj_outputsDevice,
			B, W, H);
	} else
	{
		RendererCpu::Instance().forwardVariablesToGradients(
			forwardVariablesDevice, gradientOutputColorDevice,
			settingsDevice, adj_outputsDevice,
			B, W, H);
	}
}

void renderer::Renderer::compareToImage(const torch::Tensor& colorInput, const torch::Tensor& gradientsInput,
                                        const torch::Tensor& colorReference, torch::Tensor& differenceOut, torch::Tensor& gradientsOut,
                                        bool reduce)
{
	int B, H, W, D, C;
	CHECK_DIM(colorInput, 4);
	CHECK_DTYPE(colorInput, real_dtype);
	bool cuda = colorInput.is_cuda();
	B = colorInput.size(0);
	H = colorInput.size(1);
	W = colorInput.size(2);
	CHECK_SIZE(colorInput, 3, 4);
	const auto colorInputAcc = accessor<kernel::Tensor4Read>(colorInput);

	CHECK_DIM(gradientsInput, 5);
	CHECK_CUDA(gradientsInput, cuda);
	CHECK_SIZE(gradientsInput, 0, B);
	CHECK_SIZE(gradientsInput, 1, H);
	CHECK_SIZE(gradientsInput, 2, W);
	D = gradientsInput.size(3);
	CHECK_SIZE(gradientsInput, 4, 4);
	CHECK_DTYPE(gradientsInput, real_dtype);
	const auto gradientsInputAcc = accessor<kernel::Tensor5Read>(gradientsInput);

	CHECK_DIM(colorReference, 4);
	CHECK_CUDA(colorReference, cuda);
	CHECK_SIZE(colorReference, 0, B);
	CHECK_SIZE(colorReference, 1, H);
	CHECK_SIZE(colorReference, 2, W);
	CHECK_SIZE(colorReference, 3, 4);
	CHECK_DTYPE(colorReference, real_dtype);
	const auto colorReferenceAcc = accessor<kernel::Tensor4Read>(colorReference);

	differenceOut.zero_();
	gradientsOut.zero_();
	
	if (reduce) {
		CHECK_DIM(differenceOut, 1);
		CHECK_CUDA(differenceOut, cuda);
		CHECK_SIZE(differenceOut, 0, 1);
		CHECK_DTYPE(differenceOut, real_dtype);
		auto differenceOutAcc = accessor<kernel::Tensor1RW>(differenceOut);

		CHECK_DIM(gradientsOut, 1);
		CHECK_CUDA(gradientsOut, cuda);
		CHECK_SIZE(gradientsOut, 0, D);
		CHECK_DTYPE(gradientsOut, real_dtype);
		auto gradientsOutAcc = accessor<kernel::Tensor1RW>(gradientsOut);

		if (cuda) {
			RendererCuda::Instance().compareToImage_WithReduce(
				colorInputAcc, gradientsInputAcc, colorReferenceAcc,
				differenceOutAcc, gradientsOutAcc,
				B, W, H, D);
		} else
		{
			RendererCpu::Instance().compareToImage_WithReduce(
				colorInputAcc, gradientsInputAcc, colorReferenceAcc,
				differenceOutAcc, gradientsOutAcc,
				B, W, H, D);
		}
	}
	else
	{
		CHECK_DIM(differenceOut, 4);
		CHECK_CUDA(differenceOut, cuda);
		CHECK_SIZE(differenceOut, 0, B);
		CHECK_SIZE(differenceOut, 1, H);
		CHECK_SIZE(differenceOut, 2, W);
		CHECK_SIZE(differenceOut, 3, 1);
		CHECK_DTYPE(differenceOut, real_dtype);
		auto differenceOutAcc = accessor<kernel::Tensor4RW>(differenceOut);

		CHECK_DIM(gradientsOut, 4);
		CHECK_CUDA(gradientsOut, cuda);
		CHECK_SIZE(gradientsOut, 0, B);
		CHECK_SIZE(gradientsOut, 1, H);
		CHECK_SIZE(gradientsOut, 2, W);
		CHECK_SIZE(gradientsOut, 3, D);
		CHECK_DTYPE(gradientsOut, real_dtype);
		auto gradientsOutAcc = accessor<kernel::Tensor4RW>(gradientsOut);

		if (cuda) {
			RendererCuda::Instance().compareToImage_NoReduce(
				colorInputAcc, gradientsInputAcc, colorReferenceAcc,
				differenceOutAcc, gradientsOutAcc,
				B, W, H, D);
		}
		else
		{
			RendererCpu::Instance().compareToImage_NoReduce(
				colorInputAcc, gradientsInputAcc, colorReferenceAcc,
				differenceOutAcc, gradientsOutAcc,
				B, W, H, D);
		}
	}
}

void renderer::Renderer::renderAdjoint(const RendererInputsHost& inputsHost,
	const RendererOutputsHost& outputsFromForwardHost, const torch::Tensor& adj_color, 
	AdjointOutputsHost& adj_outputs)
{
	//check inputs
	int B, W, H, X, Y, Z;
	bool cuda;
	const auto [inputsDevice, tensorsToKeepAlive] =
		checkInputs(inputsHost, B, W, H, X, Y, Z, cuda);

	//check outputsFromForwardHost
	const torch::Tensor& output_color = outputsFromForwardHost.color;
	CHECK_DIM(output_color, 4);
	CHECK_CUDA(output_color, cuda);
	CHECK_SIZE(output_color, 0, B);
	CHECK_SIZE(output_color, 1, H);
	CHECK_SIZE(output_color, 2, W);
	CHECK_SIZE(output_color, 3, 4);
	CHECK_DTYPE(output_color, real_dtype);

	const torch::Tensor& terminationIndex = outputsFromForwardHost.terminationIndex;
	CHECK_DIM(terminationIndex, 3);
	CHECK_CUDA(terminationIndex, cuda);
	CHECK_SIZE(terminationIndex, 0, B);
	CHECK_SIZE(terminationIndex, 1, H);
	CHECK_SIZE(terminationIndex, 2, W);
	CHECK_DTYPE(terminationIndex, c10::kInt);

	kernel::RendererOutputsAsInput outputsFromForwardDevice = {
		accessor<kernel::Tensor4Read>(output_color),
		accessor<kernel::ITensor3Read>(terminationIndex)
	};

	//check adjoint color
	CHECK_DIM(adj_color, 4);
	CHECK_CUDA(adj_color, cuda);
	CHECK_SIZE(adj_color, 0, B);
	CHECK_SIZE(adj_color, 1, H);
	CHECK_SIZE(adj_color, 2, W);
	CHECK_SIZE(adj_color, 3, 4);
	CHECK_DTYPE(adj_color, real_dtype);
	auto adj_colorDevice = accessor<kernel::AdjointColor_t>(adj_color);

	//check adjoint outputs
	auto adj_outputsDevice = checkAdjointOutput(
		adj_outputs, cuda, B, H, W, X, Y, Z, inputsHost.tf,
		inputsHost.volumeFilterMode == kernel::FilterPreshaded);
	const bool hasVolumeDerivative = adj_outputs.hasVolumeDerivatives;
	const bool hasStepSizeDerivative = adj_outputs.hasStepSizeDerivatives;
	const bool hasCameraDerivative = adj_outputs.hasCameraDerivatives;
	const bool hasTFDerivative = adj_outputs.hasTFDerivatives;

	if (!hasTFDerivative && !hasCameraDerivative && 
		!hasStepSizeDerivative && !hasVolumeDerivative)
	{
		std::cerr << "At least one adjoint variable must be activated!" << std::endl;
		throw std::runtime_error("At least one adjoint variable must be activated!");
	}
	
	//dispatch kernels
	if (cuda)
	{
		RendererCuda::Instance().renderAdjoint(
			inputsDevice, outputsFromForwardDevice,
			adj_colorDevice, adj_outputsDevice,
			B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode,
			hasStepSizeDerivative, hasCameraDerivative, 
			hasTFDerivative, adj_outputs.tfDelayedAcummulation, hasVolumeDerivative);
	}
	else
	{
		RendererCpu::Instance().renderAdjoint(
			inputsDevice, outputsFromForwardDevice,
			adj_colorDevice, adj_outputsDevice,
			B, W, H,
			inputsHost.volumeFilterMode, inputsHost.cameraMode,
			inputsHost.tfMode, inputsHost.blendMode,
			hasStepSizeDerivative, hasCameraDerivative, hasTFDerivative, hasVolumeDerivative);
	}
	(void)tensorsToKeepAlive;
}

