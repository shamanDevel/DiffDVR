#include "catch.hpp"

/**
 * Test if the CUDA kernels can be compiled
 */

#include <vector>
#include <renderer_cuda.h>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include "kernel_loader.h"

namespace {

	struct KernelModes
	{
		kernel::VolumeFilterMode volumeFilterMode;
		kernel::CameraMode cameraMode;
		kernel::TFMode tfMode;
		kernel::BlendMode blendMode;
	};

	struct KernelModesForward
	{
		KernelModes m;
		int numDerivatives;
		bool hasStepsizeDerivative;
		bool hasCameraDerivative;
		bool hasTFDerivative;
	};

	struct KernelModesAdjoint
	{
		KernelModes m;
		bool hasStepsizeDerivative;
		bool hasCameraDerivative;
		bool hasTFDerivative;
		bool tfDelayedAcummulation;
		bool hasVolumeDerivative;
	};

	std::vector<KernelModes> enumerateKernelModes()
	{
		std::vector<KernelModes> m;
		for (int i1=0; i1<int(kernel::__VolumeFilterModeCount__); ++i1)
			for (int i2=0; i2<int(kernel::__CameraModeCount__); ++i2)
				for (int i3=0; i3<int(kernel::__TFModeCount__); ++i3)
					for (int i4=0; i4<int(kernel::__BlendModeCount__); ++i4)
					{
						m.push_back({
							kernel::VolumeFilterMode(i1),
							kernel::CameraMode(i2),
							kernel::TFMode(i3),
							kernel::BlendMode(i4)
						});
					}
		return m;
	}
	std::vector<KernelModesForward> enumerateKernelModesForward()
	{
		std::vector<KernelModesForward> f;
		static const int NUM_DERIVATIVES = 3;
		for (const auto& m : enumerateKernelModes())
		{
			for (int i1=1; i1<=NUM_DERIVATIVES; ++i1)
				for (int i2=0; i2<2; ++i2)
					for (int i3=0; i3<2; ++i3)
						for (int i4=0; i4<2; ++i4)
						{
							if (i2 == 0 && i3 == 0 && i4 == 0) break;
							f.push_back({
								m,
								i1,
								i2>0, i3>0, i4>0
							});
						}
		}
		return f;
	}
	std::vector<KernelModesAdjoint> enumerateKernelModesAdjoint()
	{
		std::vector<KernelModesAdjoint> a;
		static const int NUM_DERIVATIVES = 3;
		for (const auto& m : enumerateKernelModes())
		{
			for (int i1 = 0; i1 < 2; ++i1)
				for (int i2 = 0; i2 < 2; ++i2)
					for (int i3 = 0; i3 < 2; ++i3)
						for (int i4 = 0; i4 < 2; ++i4)
						{
							if (i1==0 && i2 == 0 && i3 == 0 && i4 == 0) break;
							a.push_back({
								m,
								i1>0, i2 > 0, i3 > 0, false, i4 > 0
								});
							if (i3)
								a.push_back({
									m,
									i1 > 0, i2 > 0, i3 > 0, true, i4 > 0
									});
						}
		}
		return a;
	}
	
}

TEST_CASE("Compile-Forward", "[cuda][.]")
{
	auto config = enumerateKernelModes();
	//extras
	for (int i1 = 0; i1<int(kernel::__VolumeFilterModeCount__); ++i1)
		for (int i2 = 0; i2<int(kernel::__CameraModeCount__); ++i2)
			for (int i4 = 0; i4<int(kernel::__BlendModeCount__); ++i4)
			{
				config.push_back({
					kernel::VolumeFilterMode(i1),
					kernel::CameraMode(i2),
					kernel::TFMode::TFGaussianLog,
					kernel::BlendMode(i4)
					});
			}
	
	renderer::RendererCuda::Instance().reloadCudaKernels();

	using namespace indicators;
	
	indicators::show_console_cursor(false);
	
	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		option::ShowElapsedTime{true},
		option::ShowRemainingTime{true},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
		option::MaxProgress{config.size()}
	};

	for (size_t i=0; i<config.size(); ++i)
	{
		//compile kernel
		const auto& c = config[i];
		std::string kernelName = renderer::RendererCuda::Instance().getForwardKernelName(
			c.volumeFilterMode, c.cameraMode, c.tfMode, c.blendMode);
		try
		{
			const auto& fun = renderer::KernelLoader::Instance().getKernelFunction(
				kernelName, false);
			REQUIRE(fun.has_value());
		} catch (std::exception& ex)
		{
			bar.set_option(option::ForegroundColor{ Color::red });
			INFO(kernelName);
			FAIL(ex.what());
		}
		
		// Show iteration as postfix text
		bar.set_option(option::PostfixText{
		  std::to_string(i+1) + "/" + std::to_string(config.size())
			});

		// update progress bar
		bar.tick();
	}

	//bar.mark_as_completed();
	indicators::show_console_cursor(true);
}

TEST_CASE("Compile-ForwardGradients", "[cuda][.]")
{
	const auto config = enumerateKernelModesForward();
	renderer::RendererCuda::Instance().reloadCudaKernels();

	using namespace indicators;

	indicators::show_console_cursor(false);

	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		option::ShowElapsedTime{true},
		option::ShowRemainingTime{true},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
		option::MaxProgress{config.size()}
	};

	for (size_t i = 0; i < config.size(); ++i)
	{
		//compile kernel
		const auto& c = config[i];
		std::string kernelName = renderer::RendererCuda::Instance().getForwardGradientsKernelName(
			c.m.volumeFilterMode, c.m.cameraMode, c.m.tfMode, c.m.blendMode,
			c.numDerivatives, c.hasStepsizeDerivative, c.hasCameraDerivative, c.hasTFDerivative);
		try
		{
			const auto& fun = renderer::KernelLoader::Instance().getKernelFunction(
				kernelName, false);
			REQUIRE(fun.has_value());
		}
		catch (std::exception& ex)
		{
			bar.set_option(option::ForegroundColor{ Color::red });
			INFO(kernelName);
			FAIL(ex.what());
		}

		// Show iteration as postfix text
		bar.set_option(option::PostfixText{
		  std::to_string(i + 1) + "/" + std::to_string(config.size())
			});

		// update progress bar
		bar.tick();
	}

	//bar.mark_as_completed();
	indicators::show_console_cursor(true);
}

TEST_CASE("Compile-Adjoint", "[cuda][.]")
{
	const auto config = enumerateKernelModesAdjoint();
	renderer::RendererCuda::Instance().reloadCudaKernels();

	using namespace indicators;

	indicators::show_console_cursor(false);

	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		option::ShowElapsedTime{true},
		option::ShowRemainingTime{true},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
		option::MaxProgress{config.size()}
	};

	for (size_t i = 0; i < config.size(); ++i)
	{
		//compile kernel
		const auto& c = config[i];
		std::string kernelName = renderer::RendererCuda::Instance().getAdjointKernelName(
			c.m.volumeFilterMode, c.m.cameraMode, c.m.tfMode, c.m.blendMode,
			c.hasStepsizeDerivative, c.hasCameraDerivative, 
			c.hasTFDerivative, c.tfDelayedAcummulation, c.hasVolumeDerivative);
		try
		{
			const auto& fun = renderer::KernelLoader::Instance().getKernelFunction(
				kernelName, false);
			REQUIRE(fun.has_value());
		}
		catch (std::exception& ex)
		{
			bar.set_option(option::ForegroundColor{ Color::red });
			INFO(kernelName);
			FAIL(ex.what());
		}

		// Show iteration as postfix text
		bar.set_option(option::PostfixText{
		  std::to_string(i + 1) + "/" + std::to_string(config.size())
			});

		// update progress bar
		bar.tick();
	}

	//bar.mark_as_completed();
	indicators::show_console_cursor(true);
}

