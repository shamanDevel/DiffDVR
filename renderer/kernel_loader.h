#pragma once

#include <cuda.h>
#include <filesystem>
#include <map>
#include <fstream>
#include <functional>
#include <optional>


#include "commons.h"

BEGIN_RENDERER_NAMESPACE

void throwOnError(CUresult err, const char* file, int line);
#define CU_SAFE_CALL( err ) RENDERER_NAMESPACE ::throwOnError( err, __FILE__, __LINE__ )

bool printError(CUresult result, const std::string& kernelName);

class KernelLoader : public NonAssignable
{
public:
	struct NameAndContent
	{
		std::string filename;
		std::string content;
	};
	typedef std::map<std::string, CUdeviceptr> constants_t;
	
private:
	struct KernelStorage
	{
		CUmodule module;
		std::vector<char> ptxData;
		std::string machineName;
		CUfunction function;
		int minGridSize;
		int bestBlockSize;
		constants_t constants;
		std::map<std::string, std::string> human2machine;

		//creates and compiles the kernel
		KernelStorage(
			const std::string& kernelName,
			const std::vector<NameAndContent>& includeFiles,
			const std::string& extraSource,
			const std::vector<std::string>& constantNames,
			const std::vector<const char*>& compileArgs,
			bool no_log = false, bool print_main = false);

		//loads the pre-compiled kernel from the cache file
		explicit KernelStorage(std::ifstream& i);

		void loadPTX(bool no_log = false);

		//unloads the kernel
		~KernelStorage();

		//saves the kernel to the cache file
		void save(std::ofstream& o) const;
	};
	
public:
	static KernelLoader& Instance();

	/**
	 * Loads the dynamic cuda libraries of PyTorch.
	 * Call this in the main executable before any other calls
	 * to the renderer or PyTorch.
	 * \return true if cuda is available and was loaded successfully
	 */
	bool initCuda();
	void setCudaCacheDir(const std::filesystem::path& path);
	void disableCudaCache();
	void reloadCudaKernels();
	void cleanup();
	void setDebugMode(bool debug);
	
	/**
	 * Function to load cuda sources.
	 * Populates the given vector with pairs of (file name, file content).
	 */
	typedef std::function<void(std::vector<NameAndContent>&)> customCudaSourcesLoader_t;
	/**
	 * Sets a custom cuda sources loader.
	 * If not set, loads the cuda sources from disk based on the absolute
	 * path specified by the CMake variable RENDERER_SHADER_DIRS
	 */
	void setCustomCudaSourcesLoader(const customCudaSourcesLoader_t& loader);
	/**
	 * Explicitly sets the kernel cache file.
	 * Only kernels from this cache file are available, runtime compilation
	 * is disabled. This is used for the server where the CUDA SDK is not available.
	 */
	void setKernelCacheFile(std::string file);

	class KernelFunction
	{
		std::shared_ptr<KernelStorage> storage_;
		friend class KernelLoader;
		explicit KernelFunction(const std::shared_ptr<KernelStorage>& storage) : storage_(storage) {}
	public:
		KernelFunction() = default;
		CUfunction fun() const;
		int minGridSize() const;
		int bestBlockSize() const;
		/**
		 * \brief Returns the device pointer for the constant variable with the given name
		 * or '0' if not found.
		 */
		CUdeviceptr constant(const std::string& name) const;
	};
	
	/**
	 * \brief Retrieves the CUfunction for the specified kernel.
	 * The kernel is compiled if needed.
	 * \param kernel the name of the kernel as you would declare it in code.
	 * \param no_throw true -> error messages are printed to std::cout/cerr and
	 *		nullptr is returned on error.
	 *		false -> an std::runtime_error is thrown on error
	 * \return the CUfunction or nullptr if not found / unable to compile
	 */
	std::optional<KernelFunction> getKernelFunction(
		const std::string& kernel, bool no_throw = true);

	/**
	 * \brief Retrieves the CUfunction for the specified kernel.
	 * The kernel is compiled if needed.
	 *
	 * Note: the kernel name must be unique as it is used to cache
	 * the compiled kernel in the kernel storage.
	 * 
	 * \param kernel the name of the kernel as you would declare it in code.
	 * \param extraSource extra sources that are appended to the include files
	 *    for custom generated kernels.
	 * \param constantNames list of constant variables that are converted to device names
	 *   and accessible via \ref KernelFunction::constant(std::string).
	 * \param noCache if true, the kernel is not cached and always compiled
	 * \param noThrow true -> error messages are printed to std::cout/cerr and
	 *		EMPTY is returned on error.
	 *		false -> an std::runtime_error is thrown on error,
	 *		  the returned optional is never empty
	 * \return the CUfunction or nullptr if not found / unable to compile
	 */
	std::optional<KernelFunction> getKernelFunction(
		const std::string& kernel,
		const std::string& extraSource,
		const std::vector<std::string>& constantNames,
		bool noCache = false,
		bool noThrow = true);

	/**
	 * Returns the kernel function for the specified kernel,
	 * if it is already compiled and cached.
	 * As opposed to \ref getKernelFunction(),
	 * this function does not compile the kernel if it is not found/cached.
	 */
	std::optional<KernelFunction> getCachedKernelFunction(const std::string& kernel);

private:

	//Loads the CUDA source file
	//Returns true if the source files have changed
	bool loadCUDASources(bool no_log = false);

	void saveKernelCache();
	void loadKernelCache(bool no_log = false);
	static constexpr unsigned int KERNEL_CACHE_MAGIC = 0x61437543u; //CuCa

	std::filesystem::path CACHE_DIR = "kernel_cache";
	bool debugMode = false;

	std::vector<NameAndContent> includeFiles;
	std::string includeFilesHash;
	customCudaSourcesLoader_t customCudaSourcesLoader;
	std::string customKernelCacheFile;

	std::map<std::string, std::shared_ptr<KernelStorage>> kernelStorage;
};

END_RENDERER_NAMESPACE
