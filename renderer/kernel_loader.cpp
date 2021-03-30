#include "kernel_loader.h"

#include <cuMat/src/Context.h>
#include <iostream>
#include <filesystem>
#include <nvrtc.h>
#include <sstream>
#include <fstream>
#include <magic_enum.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <torch/cuda.h>
#include <tinyformat.h>

#include "sha1.h"

namespace fs = std::filesystem;

void renderer::throwOnError(CUresult err, const char* file, int line)
{
	if (err != CUDA_SUCCESS)
	{
		const char* pStr;
		cuGetErrorString(err, &pStr);
		const char* pName;
		cuGetErrorName(err, &pName);
		std::stringstream ss;
		ss << "Cuda error " << pName << " at " << file << ":" << line << " : " << pStr;
		throw std::runtime_error(ss.str().c_str());
	}
}

static void throwOnNvrtcError(nvrtcResult result, const char* file, const int line)
{
	if (result != NVRTC_SUCCESS) {
		std::stringstream ss;
		ss << "NVRTC error at " << file << ":" << line << " : " << nvrtcGetErrorString(result);
		throw std::runtime_error(ss.str().c_str());
	}
}
#define NVRTC_SAFE_CALL( err ) throwOnNvrtcError( err, __FILE__, __LINE__ )

bool renderer::printError(CUresult result, const std::string& kernelName)
{
	const char* pStr;
	cuGetErrorString(result, &pStr);
	std::cerr << "Unable to launch kernel " << kernelName << ":\n " << pStr << std::endl;
	return false;
}

renderer::KernelLoader& renderer::KernelLoader::Instance()
{
	static renderer::KernelLoader INSTANCE;
	return INSTANCE;
}

bool renderer::KernelLoader::initCuda()
{
	//https://github.com/pytorch/pytorch/issues/31611
	int warpSize = at::cuda::warp_size();
	bool cudaAvailable = torch::cuda::is_available();
	return warpSize > 0 && cudaAvailable;
}

void renderer::KernelLoader::setCudaCacheDir(const std::filesystem::path& path)
{
	CACHE_DIR = path;
	reloadCudaKernels();
}

void renderer::KernelLoader::disableCudaCache()
{
	CACHE_DIR = "";
	assert(CACHE_DIR.empty());
}

void renderer::KernelLoader::reloadCudaKernels()
{
	includeFiles.clear();
	kernelStorage.clear();
}

void renderer::KernelLoader::cleanup()
{
	kernelStorage.clear();
}

void renderer::KernelLoader::setCustomCudaSourcesLoader(const customCudaSourcesLoader_t& loader)
{
	customCudaSourcesLoader = loader;
}

void renderer::KernelLoader::setKernelCacheFile(std::string file)
{
	customKernelCacheFile = file;
	reloadCudaKernels();
	loadKernelCache(false);
}

CUfunction renderer::KernelLoader::KernelFunction::fun() const
{
	return storage_->function;
}

int renderer::KernelLoader::KernelFunction::minGridSize() const
{
	return storage_->minGridSize;
}

int renderer::KernelLoader::KernelFunction::bestBlockSize() const
{
	return storage_->bestBlockSize;
}

CUdeviceptr renderer::KernelLoader::KernelFunction::constant(const std::string& name) const
{
	const auto it = storage_->constants.find(name);
	if (it == storage_->constants.end())
		return 0;
	else
		return it->second;
}

void renderer::KernelLoader::setDebugMode(bool debug)
{
	debugMode = debug;
	reloadCudaKernels();
}

bool renderer::KernelLoader::loadCUDASources(bool no_log)
{
	if (!includeFiles.empty()) return false;

	// load files
	if (customCudaSourcesLoader)
	{
		customCudaSourcesLoader(includeFiles);
	}
	else
	{
#ifdef RENDERER_SHADER_DIRS
		int index = 0;
		for (const char* rootStr : RENDERER_SHADER_DIRS) {
			fs::path root(rootStr);
			bool allowHfiles = index > 0;
			for (const auto& p : fs::directory_iterator(root))
			{
				if (p.path().extension() == ".cuh" ||
					(allowHfiles && p.path().extension() == ".h") ||
					p.path().extension() == ".inl")
				{
					try
					{
						std::ifstream t(p.path());
						std::ostringstream ss;
						ss << t.rdbuf();
						std::string buffer = ss.str();
						includeFiles.emplace_back(NameAndContent{ p.path().filename().string(), buffer });
						if (!no_log) std::cout << "Loaded file " << p.path() << std::endl;
					}
					catch (const std::exception& ex)
					{
						std::cerr << "Unable to read file " << p.path() << ": " << ex.what() << std::endl;
					}
				}
			}
			index++;
		}
#else
		throw std::runtime_error("RENDERER_SHADER_DIRS not specified as preprocessor macro. You must set a custom sources loader then.");
#endif
	}

	// compute hashes
	SHA1 sha1;
	for (const auto& e : includeFiles)
		sha1.update(e.content);

	sha1.update(CUMAT_STR(USE_DOUBLE_PRECISION));
	sha1.update(std::to_string(debugMode));

	std::string previousHash = includeFilesHash;
	includeFilesHash = sha1.final();
	return previousHash != includeFilesHash;
}

std::optional<renderer::KernelLoader::KernelFunction> renderer::KernelLoader::getKernelFunction(
	const std::string& kernel, bool no_throw)
{
	return getKernelFunction(kernel, "", {}, false, no_throw);
}

std::optional<renderer::KernelLoader::KernelFunction> renderer::KernelLoader::getKernelFunction(const std::string& kernel,
	const std::string& extraSource, const std::vector<std::string>& constantNames, 
	bool noCache, bool noThrow)
{
	loadKernelCache(!noThrow);

	const auto it = kernelStorage.find(kernel);
	if (noCache || it == kernelStorage.end())
	{
		if (!customKernelCacheFile.empty()) {
			throw std::runtime_error(
				"Kernel " + kernel +
				" not found, but recompilation is disabled because a custom kernel cache file was specified.");
		}

		//kernel not found in the cache, recompile it
#ifdef NVCC_ARGS
		std::string nvccArgsFromCMake(NVCC_ARGS);
		std::string nvccComputeVersion = nvccArgsFromCMake.substr(nvccArgsFromCMake.size() - 2, 2);
		std::string newNvccArgs = "--gpu-architecture=compute_" + nvccComputeVersion;
#else
		//fallback
		std::string newNvccArgs = "--gpu-architecture=compute_61";
#endif
		if (noThrow) std::cout << "NVCC args: " << newNvccArgs << "\n";
		std::vector<const char*> opts{
			"--std=c++17",
			"--use_fast_math",
			//"--device-debug",
			"--generate-line-info",
			newNvccArgs.c_str(),
#ifdef NVCC_INCLUDE_DIR
			"-I" CUMAT_STR(NVCC_INCLUDE_DIR), // NOLINT(bugprone-suspicious-missing-comma)
#endif
			"-D__NVCC__=1",
			"-DCUDA_NO_HOST=1",
			"-DUSE_DOUBLE_PRECISION=" CUMAT_STR(USE_DOUBLE_PRECISION)  // NOLINT(bugprone-suspicious-missing-comma)
		};
		if (debugMode)
		{
			opts.push_back("-G");
		}

		try {
			const auto storage = std::make_shared<KernelStorage>(
				kernel, includeFiles, extraSource, constantNames, opts, !noThrow, debugMode);
			if (!noCache) {
				kernelStorage.emplace(kernel, storage);
				saveKernelCache();
			}
			return KernelFunction(storage);
		}
		catch (std::exception& ex)
		{
			if (noThrow) {
				std::cerr << "Unable to compile kernel: " << ex.what() << std::endl;
				reloadCudaKernels(); //so that in the next iteration, we can compile again
				return {};
			}
			else
				throw std::runtime_error(ex.what());
		}
	}
	else
	{
		//kernel found, return immediately
		return KernelFunction(it->second);
	}
}

std::optional<renderer::KernelLoader::KernelFunction> renderer::KernelLoader::getCachedKernelFunction(
	const std::string& kernel)
{
	const auto it = kernelStorage.find(kernel);
	if (it == kernelStorage.end())
		return {}; //empty
	else
		return KernelFunction(it->second);
}

void renderer::KernelLoader::saveKernelCache()
{
	if (CACHE_DIR.empty()) return;

	if (!exists(CACHE_DIR)) {
		if (!create_directory(CACHE_DIR)) {
			std::cerr << "Unable to create cache directory " << absolute(CACHE_DIR) << std::endl;
			return;
		}
		else
			std::cout << "Cache directory created at " << absolute(CACHE_DIR) << std::endl;
	}

	fs::path cacheFile = CACHE_DIR / (includeFilesHash + ".kernel");
	std::ofstream o(cacheFile, std::ofstream::binary);
	if (!o.is_open())
	{
		std::cerr << "Unable to open cache file " << absolute(cacheFile) << " for writing" << std::endl;
		return;
	}
	o.write(reinterpret_cast<const char*>(&KERNEL_CACHE_MAGIC), sizeof(int));
	size_t entrySize = kernelStorage.size();
	o.write(reinterpret_cast<const char*>(&entrySize), sizeof(size_t));
	for (const auto e : kernelStorage)
	{
		size_t kernelNameSize = e.first.size();
		o.write(reinterpret_cast<const char*>(&kernelNameSize), sizeof(size_t));
		o.write(e.first.data(), kernelNameSize);
		e.second->save(o);
	}
	std::cout << entrySize << " kernels written to the cache file " << cacheFile << std::endl;
}

void renderer::KernelLoader::loadKernelCache(bool no_log)
{
	if (!kernelStorage.empty()) return; //already loaded

	//load cuda source files and updates the SHA1 hash
	loadCUDASources(no_log);

	fs::path cacheFile;
	if (customKernelCacheFile.empty()) {
		if (CACHE_DIR.empty()) return;
		cacheFile = CACHE_DIR / (includeFilesHash + ".kernel");
	}
	else
	{
		std::cout << "Use custom cache file " << customKernelCacheFile << std::endl;
		cacheFile = customKernelCacheFile;
	}
	if (exists(cacheFile))
	{
		std::cout << "Read from cache " << cacheFile << std::endl;
		std::ifstream i(cacheFile, std::ifstream::binary);
		if (!i.is_open())
		{
			std::cerr << "Unable to open file" << std::endl;
			return;
		}
		unsigned int magic;
		i.read(reinterpret_cast<char*>(&magic), sizeof(int));
		if (magic != KERNEL_CACHE_MAGIC)
		{
			std::cerr << "Invalid magic number, wrong file type or file is corrupted" << std::endl;
			return;
		}
		size_t entrySize;
		i.read(reinterpret_cast<char*>(&entrySize), sizeof(size_t));
		for (size_t j = 0; j < entrySize; ++j)
		{
			size_t kernelNameSize;
			std::string kernelName;
			i.read(reinterpret_cast<char*>(&kernelNameSize), sizeof(size_t));
			kernelName.resize(kernelNameSize);
			i.read(kernelName.data(), kernelNameSize);

			const auto storage = std::make_shared<KernelStorage>(i);
			kernelStorage.emplace(kernelName, storage);
		}
		std::cout << entrySize << " kernels loaded from cache" << std::endl;
	}
}

renderer::KernelLoader::KernelStorage::KernelStorage(const std::string& kernelName,
	const std::vector<NameAndContent>& includeFiles,
	const std::string& extraSource,
	const std::vector<std::string>& constantNames,
	const std::vector<const char*>& compileArgs,
	bool no_log, bool print_main)
{
	if (!no_log) std::cout << "Compile kernel \"" << kernelName << "\"" << std::endl;

	//create program
	nvrtcProgram prog;
	std::stringstream source;
	for (const auto& e : includeFiles)
	{
		source << "#include \"" << e.filename << "\"\n";
	}
	source << extraSource << "\n";

	const auto printSourceCode = [](const std::string& s)
	{
		std::istringstream iss(s);
		int lineIndex = 1;
		for (std::string line; std::getline(iss, line); lineIndex++)
		{
			std::cout << tinyformat::format("[%05d] %s\n", lineIndex, line);
		}
		std::cout << std::flush;
	};
	if (print_main)
	{
		printSourceCode(source.str());
	}
	
	std::vector<const char*> headerContents(includeFiles.size());
	std::vector<const char*> headerNames(includeFiles.size());
	for (size_t i = 0; i < includeFiles.size(); ++i)
	{
		headerContents[i] = includeFiles[i].content.c_str();
		headerNames[i] = includeFiles[i].filename.c_str();
	}
	const char* const* headers = includeFiles.empty() ? nullptr : headerContents.data();
	const char* const* includeNames = includeFiles.empty() ? nullptr : headerNames.data();
	NVRTC_SAFE_CALL(
		nvrtcCreateProgram(&prog,
			source.str().c_str(),
			"main.cu",
			includeFiles.size(),
			headers,
			includeNames));

	//add kernel name for resolving the native name
	NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernelName.c_str()));
	//add constant names
	for (const auto& var : constantNames)
	{
		NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, ("&" + var).c_str()));
	}

	//compile
	nvrtcResult compileResult = nvrtcCompileProgram(prog, compileArgs.size(), compileArgs.data());
	// obtain log
	size_t logSize;
	NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
	std::vector<char> log(logSize);
	NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
	if (!no_log) std::cout << log.data();
	if (compileResult != NVRTC_SUCCESS)
	{
		nvrtcDestroyProgram(&prog); //ignore possible errors
		if (!print_main)
			printSourceCode(source.str());
		std::string msg = std::string("Failed to compile kernel:\n") + log.data();
		throw std::runtime_error(msg.c_str());
	}

	//optain PTX
	size_t ptxSize;
	NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
	this->ptxData.resize(ptxSize);
	NVRTC_SAFE_CALL(nvrtcGetPTX(prog, this->ptxData.data()));

	////test
	//std::string ptxStr(this->ptxData.begin(), this->ptxData.end());
	//std::cout << "\nPTX:\n" << ptxStr << "\n" << std::endl;

	//get machine name
	const char* machineName;
	NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernelName.c_str(), &machineName));
	this->machineName = machineName;
	for (const auto& var : constantNames)
	{
		std::string humanName = "&" + var;
		const char* machineName;
		NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, humanName.c_str(), &machineName));
		human2machine.emplace(var, std::string(machineName));
	}

	//delete program
	NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

	loadPTX(no_log);
}

renderer::KernelLoader::KernelStorage::KernelStorage(std::ifstream& i)
{
	size_t machineNameSize, ptxSize;
	i.read(reinterpret_cast<char*>(&machineNameSize), sizeof(size_t));
	machineName.resize(machineNameSize);
	i.read(machineName.data(), machineNameSize);

	i.read(reinterpret_cast<char*>(&ptxSize), sizeof(size_t));
	ptxData.resize(ptxSize);
	i.read(ptxData.data(), ptxSize);

	size_t human2machineSize, strSize;
	std::string key, value;
	human2machine.clear();
	i.read(reinterpret_cast<char*>(&human2machineSize), sizeof(size_t));
	for (size_t j=0; j<human2machineSize; ++j)
	{
		i.read(reinterpret_cast<char*>(&strSize), sizeof(size_t));
		key.resize(strSize);
		i.read(key.data(), strSize);

		i.read(reinterpret_cast<char*>(&strSize), sizeof(size_t));
		value.resize(strSize);
		i.read(value.data(), strSize);

		human2machine[key] = value;
	}

	loadPTX(false);
}

void renderer::KernelLoader::KernelStorage::loadPTX(bool no_log)
{
	if (!no_log) {
		std::cout << "Load module \"" << this->machineName << "\"" << std::endl;
	}
	//load PTX
	CU_SAFE_CALL(cuModuleLoadDataEx(&this->module, this->ptxData.data(), 0, 0, 0));

	//get cuda function and constants
	CU_SAFE_CALL(cuModuleGetFunction(&this->function, this->module, this->machineName.data()));
	for (const auto& e : human2machine)
	{
		if (!no_log) {
			std::cout << "Fetch address for constant variable \"" << e.first
				<< "\", machine name \"" << e.second << "\"" << std::endl;
		}
		CUdeviceptr addr;
		CU_SAFE_CALL(cuModuleGetGlobal(&addr, nullptr, module, e.second.data()));
		constants[e.first] = addr;
		if (!no_log)
			std::cout << "constant variable " << e.first << " has device pointer 0x"
				<< std::hex << addr << std::dec << std::endl;
	}
	
	CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(
		&minGridSize, &bestBlockSize, function, NULL, 0, 0));

	if (!no_log) {
		std::cout << "Module \"" << this->machineName << "\" loaded successfully"
			<< ", block size: " << bestBlockSize << std::endl;
	}
}

renderer::KernelLoader::KernelStorage::~KernelStorage()
{
	CUresult err = cuModuleUnload(this->module);
	if (err != CUDA_SUCCESS) {
		const char* pStr;
		cuGetErrorString(err, &pStr);
		const char* pName;
		cuGetErrorName(err, &pName);
		std::stringstream ss;
		std::cerr << "Cuda error " << pName << " when unloading module for kernel " << machineName << ":" << pStr << std::endl;
	}
}

void renderer::KernelLoader::KernelStorage::save(std::ofstream& o) const
{
	size_t machineNameSize = machineName.size();
	o.write(reinterpret_cast<const char*>(&machineNameSize), sizeof(size_t));
	o.write(machineName.c_str(), machineNameSize);

	size_t ptxSize = ptxData.size();
	o.write(reinterpret_cast<const char*>(&ptxSize), sizeof(size_t));
	o.write(ptxData.data(), ptxSize);

	size_t human2machineSize = human2machine.size();
	o.write(reinterpret_cast<const char*>(&human2machineSize), sizeof(size_t));
	for (const auto& e : human2machine)
	{
		size_t strSize = e.first.size();
		o.write(reinterpret_cast<const char*>(&strSize), sizeof(size_t));
		o.write(e.first.data(), strSize);

		strSize = e.second.size();
		o.write(reinterpret_cast<const char*>(&strSize), sizeof(size_t));
		o.write(e.second.data(), strSize);
	}
}
