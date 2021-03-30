#define CATCH_CONFIG_RUNNER
#include <catch.hpp>
#include <iostream>

#include <renderer.h>
#include <cuMat/src/DevicePointer.h>

struct MyListener : Catch::TestEventListenerBase {

	using TestEventListenerBase::TestEventListenerBase; // inherit constructor

	virtual void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
		std::cout << "Execute " << testInfo.tagsAsString() << " " << testInfo.name << std::endl;
	}
};
CATCH_REGISTER_LISTENER(MyListener)


int main(int argc, char* argv[]) {
	// global setup

#if USE_DOUBLE_PRECISION==0
	std::cout << "WARNING: Single precision requested, unit tests will most likely fail!" << std::endl;
#endif

	size_t currentPrintfLimit;
	cudaDeviceGetLimit(&currentPrintfLimit, cudaLimitPrintfFifoSize);
	const size_t newPrintfLimit = 1024 * 1024 * 16;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, newPrintfLimit);
	std::cout << "Change printf limit from " << currentPrintfLimit << " to " << newPrintfLimit << std::endl;
	
	renderer::Renderer::initCuda();
	renderer::Renderer::disableCudaCache();

	int result;
	{
		cuMat::DevicePointer<int> ptr(1); //hold reference to CUDA
		result = Catch::Session().run(argc, argv);
		((void)ptr); //force it to be held
	}

	// global clean-up
	renderer::Renderer::cleanupCuda();

	return result;
}
