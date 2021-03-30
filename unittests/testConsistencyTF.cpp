/**
 * Tests that the differentiation algorithms
 * are consistent across device (CUDA, CPU) and
 * method (forward, adjoint)
 */

#include <catch.hpp>
#include <random>

#include <renderer.h>
#include <volume.h>
#include <camera.h>
#include <pytorch_utils.h>
#include <cuMat/src/Context.h>

static real_t difference(const torch::Tensor& t1, const torch::Tensor& t2)
{
	auto t1cpu = t1.to(c10::kCPU);
	auto t2cpu = t2.to(c10::kCPU);
	CHECK(t1cpu.sizes() == t2cpu.sizes());

	const real_t* acc1 = t1cpu.data_ptr<real_t>();
	const real_t* acc2 = t2cpu.data_ptr<real_t>();
	int size = cuMat::internal::narrow_cast<int>(t1cpu.numel());

	real_t maxSymmetricError = 0;
//#pragma omp parallel for reduction(max : maxSymmetricError)
	for (int i=0; i<size; ++i)
	{
		real_t x1 = acc1[i], x2 = acc2[i];
		real_t err = std::abs(x1 - x2) / std::max(real_t(1), std::abs(x1) + std::abs(x2));
		if (err > 0.5) {
			std::cout << "i=" << i << ", x1=" << x1 << ", x2=" << x2 << " -> err=" << err << std::endl;
			//__debugbreak();
		}
		if (err > maxSymmetricError)
			maxSymmetricError = err;
	}
	//std::cout << "max symmetric error: " << maxSymmetricError << std::endl;
	return maxSymmetricError;
}

TEST_CASE("Renderer-TfDiff", "[renderer]")
{
	bool oldSyncMode = renderer::Renderer::getCudaSyncMode();
	renderer::Renderer::setCudaSyncMode(true);
	// create test case
	int W = 128;
	int H = 128;
	static const float EPS = 1e-5f;
	const bool hasCpu = RENDERER_BUILD_CPU_KERNELS;
	
	auto volume = renderer::Volume::createImplicitDataset(
		32, renderer::Volume::ImplicitEquation::MARSCHNER_LOBB);
	volume->getLevel(0)->copyCpuToGpu();

	glm::mat4 cam = renderer::Camera::computeInverseViewProjectionMatrix(
		make_float3(-1, -0.5, 0), make_float3(0, 0, 0), make_float3(0, 0, 1),
		45, W, H, 0.1, 10.0);

	struct TfConfig
	{
		kernel::TFMode tfMode;
		int resolution;
	};
	std::vector<TfConfig> tfConfigs{
		//{kernel::TFTexture, 64},
		{kernel::TFIdentity, 1},
		{kernel::TFTexture, 2},
		{kernel::TFTexture, 4},
		{kernel::TFTexture, 8},
		{kernel::TFLinear, 4}
	};
	for (const auto tfConfig : tfConfigs)
	{
		std::cout << "use TF " << tfConfig.tfMode << " with resolution " << tfConfig.resolution << std::endl;
		INFO("use TF " << tfConfig.tfMode << " with resolution " << tfConfig.resolution);
		int C = -1;
		switch (tfConfig.tfMode)
		{
		case kernel::TFIdentity: C = 2; break;
		case kernel::TFLinear: C = 5; break;
		case kernel::TFTexture: C = 4; break;
		}
		torch::Tensor tf = torch::rand({ 1, tfConfig.resolution, C },
			at::TensorOptions().dtype(real_dtype).device(c10::kCPU));
		if (tfConfig.tfMode == kernel::TFLinear)
		{
			//overwrite positions
			auto acc = tf.accessor<real_t, 3>();
			acc[0][0][4] = 0;
			for (int i=1; i<tfConfig.resolution-1; ++i)
			{
				acc[0][i][4] = i/(tfConfig.resolution - 1);
			}
			acc[0][tfConfig.resolution-1][4] = 1;
		}
		torch::Tensor tfCuda = tf.to(c10::kCUDA);

		renderer::RendererInputsHost settingsCpu;
		settingsCpu.screenSize = make_int2(W, H);
		settingsCpu.volumeFilterMode = kernel::FilterTrilinear;
		settingsCpu.volume = volume->getLevel(0)->dataCpu();
		settingsCpu.boxMin = make_real3(-0.5, -0.5, -0.5);
		settingsCpu.boxSize = make_real3(1, 1, 1);
		settingsCpu.cameraMode = kernel::CameraInverseViewMatrix;
		settingsCpu.camera = cam;
		settingsCpu.stepSize = 0.5 / 64;
		settingsCpu.tfMode = tfConfig.tfMode;
		settingsCpu.tf = tf;
		settingsCpu.blendMode = kernel::BlendBeerLambert;
		renderer::RendererInputsHost settingsCuda = settingsCpu;
		settingsCuda.volume = volume->getLevel(0)->dataGpu();
		settingsCuda.tf = tfCuda;
		
		// render forward
		renderer::RendererOutputsHost outputsCpu{
			torch::empty({1,H,W,4}, at::TensorOptions().dtype(real_dtype).device(c10::kCPU)),
			torch::empty({1,H,W}, at::TensorOptions().dtype(c10::kInt).device(c10::kCPU))
		};
		if (hasCpu) renderer::Renderer::renderForward(settingsCpu, outputsCpu);

		renderer::RendererOutputsHost outputsCuda{
			torch::empty({1,H,W,4}, at::TensorOptions().dtype(real_dtype).device(c10::kCUDA)),
			torch::empty({1,H,W}, at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA))
		};
		renderer::Renderer::renderForward(settingsCuda, outputsCuda);

		// create adjoint variable
		const torch::Tensor adjointColorCpu = torch::randn_like(outputsCpu.color);
		const torch::Tensor adjointColorCuda = adjointColorCpu.to(c10::kCUDA);
		const auto createAdjointOutput = [&](bool cuda, bool tfDelayedAcummulation = false)
		{
			renderer::AdjointOutputsHost deriv;
			deriv.hasTFDerivatives = true;
			deriv.tfDelayedAcummulation = tfDelayedAcummulation;
			deriv.adj_tf = torch::zeros(
				{ 1,tfConfig.resolution,C },
				at::TensorOptions().dtype(real_dtype).device(cuda ? c10::kCUDA : c10::kCPU));
			return deriv;
		};

		// render forward
		renderer::ForwardDifferencesSettingsHost fdSettingsCpu;
		fdSettingsCpu.D = tfConfig.resolution * C;
		fdSettingsCpu.d_tf = torch::reshape(
			torch::arange(0, fdSettingsCpu.D, 1, at::TensorOptions().dtype(c10::kInt)),
			{ 1, tfConfig.resolution, C });
		fdSettingsCpu.hasTfDerivatives = true;
		renderer::ForwardDifferencesSettingsHost fdSettingsCuda = fdSettingsCpu;
		fdSettingsCuda.d_tf = fdSettingsCpu.d_tf.to(c10::kCUDA);
		//cpu
		torch::Tensor gradientsOut = torch::empty(
			{ 1, H, W, fdSettingsCpu.D, 4 }, 
			at::TensorOptions().dtype(real_dtype).device(c10::kCPU));
		renderer::RendererOutputsHost outputsCpuScrap{
			torch::empty({1,H,W,4}, at::TensorOptions().dtype(real_dtype).device(c10::kCPU)),
			torch::empty({1,H,W}, at::TensorOptions().dtype(c10::kInt).device(c10::kCPU))
		};
		if (hasCpu) renderer::Renderer::renderForwardGradients(settingsCpu, fdSettingsCpu, outputsCpuScrap, gradientsOut);
		renderer::AdjointOutputsHost derivForwardCpu = createAdjointOutput(false);
		if (hasCpu) renderer::Renderer::forwardVariablesToGradients(gradientsOut, adjointColorCpu, fdSettingsCpu, derivForwardCpu);
		//cuda
		gradientsOut = torch::empty(
			{ 1, H, W, fdSettingsCuda.D, 4 },
			at::TensorOptions().dtype(real_dtype).device(c10::kCUDA));
		renderer::RendererOutputsHost outputsCudaScrap{
			torch::empty({1,H,W,4}, at::TensorOptions().dtype(real_dtype).device(c10::kCUDA)),
			torch::empty({1,H,W}, at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA))
		};
		renderer::Renderer::renderForwardGradients(settingsCuda, fdSettingsCuda, outputsCudaScrap, gradientsOut);
		renderer::AdjointOutputsHost derivForwardCuda = createAdjointOutput(true);
		renderer::Renderer::forwardVariablesToGradients(gradientsOut, adjointColorCuda, fdSettingsCuda, derivForwardCuda);

		//adjoint
		renderer::AdjointOutputsHost derivAdjointImmediateCpu = createAdjointOutput(false, false);
		if (hasCpu) renderer::Renderer::renderAdjoint(settingsCpu, outputsCpu, adjointColorCpu, derivAdjointImmediateCpu);
		renderer::AdjointOutputsHost derivAdjointImmediateCuda = createAdjointOutput(true, false);
		renderer::Renderer::renderAdjoint(settingsCuda, outputsCuda, adjointColorCuda, derivAdjointImmediateCuda);
		renderer::AdjointOutputsHost derivAdjointDelayedCpu = createAdjointOutput(false, true);
		if (hasCpu) renderer::Renderer::renderAdjoint(settingsCpu, outputsCpu, adjointColorCpu, derivAdjointDelayedCpu);
		renderer::AdjointOutputsHost derivAdjointDelayedCuda = createAdjointOutput(true, true);
		renderer::Renderer::renderAdjoint(settingsCuda, outputsCuda, adjointColorCuda, derivAdjointDelayedCuda);

		std::cout << "Done. Now sync and compare" << std::endl;
		CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		
		
		//comparison
		bool success = true;
#define MY_CHECK(t1, t2)				\
	do {								\
		real_t d = difference(t1, t2);	\
		if (d > EPS) {					\
			std::cerr << "Difference of " << d << " between tensors " CUMAT_STR(t1) " and " CUMAT_STR(t2) " exceed the threshold of " << EPS << std::endl;	\
			success = false;			\
		} else {						\
			std::cout << "Difference between tensors " CUMAT_STR(t1) " and " CUMAT_STR(t2) " is " << d << std::endl;	\
		}	\
	} while (0);

		if (hasCpu) MY_CHECK(outputsCpu.color, outputsCuda.color);
		
		if (hasCpu) MY_CHECK(derivForwardCpu.adj_tf, derivForwardCuda.adj_tf);
		if (hasCpu) MY_CHECK(derivForwardCpu.adj_tf, derivAdjointImmediateCpu.adj_tf);
		if (hasCpu) MY_CHECK(derivForwardCpu.adj_tf, derivAdjointImmediateCuda.adj_tf);
		if (hasCpu) MY_CHECK(derivForwardCpu.adj_tf, derivAdjointDelayedCpu.adj_tf);
		if (hasCpu) MY_CHECK(derivForwardCpu.adj_tf, derivAdjointDelayedCuda.adj_tf);
		
		if (hasCpu) MY_CHECK(derivForwardCuda.adj_tf, derivAdjointImmediateCpu.adj_tf);
		MY_CHECK(derivForwardCuda.adj_tf, derivAdjointImmediateCuda.adj_tf);
		if (hasCpu) MY_CHECK(derivForwardCuda.adj_tf, derivAdjointDelayedCpu.adj_tf);
		MY_CHECK(derivForwardCuda.adj_tf, derivAdjointDelayedCuda.adj_tf);
		
		if (hasCpu) MY_CHECK(derivAdjointImmediateCpu.adj_tf, derivAdjointImmediateCuda.adj_tf);
		if (hasCpu) MY_CHECK(derivAdjointImmediateCpu.adj_tf,  derivAdjointDelayedCpu.adj_tf );
		if (hasCpu) MY_CHECK(derivAdjointImmediateCpu.adj_tf,  derivAdjointDelayedCuda.adj_tf);
		
		if (hasCpu) MY_CHECK(derivAdjointImmediateCuda.adj_tf, derivAdjointDelayedCpu.adj_tf );
		MY_CHECK(derivAdjointImmediateCuda.adj_tf, derivAdjointDelayedCuda.adj_tf);
		
		if (hasCpu) MY_CHECK(derivAdjointDelayedCpu.adj_tf,    derivAdjointDelayedCuda.adj_tf);

		if (!success) FAIL("At least one comparison failed");
	}
	renderer::Renderer::setCudaSyncMode(oldSyncMode);
}