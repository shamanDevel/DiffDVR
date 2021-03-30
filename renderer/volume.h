#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <functional>
#include <cassert>
#include <vector>
#include <torch/types.h>
#include <cfloat>

#include "commons.h"
#include "renderer_settings.cuh"

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

BEGIN_RENDERER_NAMESPACE

typedef std::function<void(const std::string&)> VolumeLoggingCallback_t;
typedef std::function<void(float)> VolumeProgressCallback_t;
typedef std::function<void(const std::string&, int)> VolumeErrorCallback_t;

class MY_API Volume
{
public:
	template<int numOfBins>
	struct VolumeHistogram
	{
		float bins[numOfBins]{ 0.0f };
		float minDensity{ FLT_MAX };
		float maxDensity{ 0.0f };
		float maxFractionVal{ 1.0f };
		unsigned int numOfNonzeroVoxels{ 0 };

		static constexpr int getNumOfBins() { return numOfBins; }
	};
	using Histogram = VolumeHistogram<512>;
	
public:

	class MY_API MipmapLevel
	{
	private:
		int sizeX_, sizeY_, sizeZ_;
		std::vector<real_t> dataCpuBlob_;
		torch::Tensor dataCpu_;
		torch::Tensor dataGpu_;
		int cpuDataCounter_;
		int gpuDataCounter_;

		Volume* parent_;
		friend class Volume;

		MipmapLevel(Volume* parent, int sizeX, int sizeY, int sizeZ);
	public:
		~MipmapLevel();

		MipmapLevel(const MipmapLevel& other) = delete;
		MipmapLevel(MipmapLevel&& other) noexcept = delete;
		MipmapLevel& operator=(const MipmapLevel& other) = delete;
		MipmapLevel& operator=(MipmapLevel&& other) noexcept = delete;

		int sizeX() const { return sizeX_; }
		int sizeY() const { return sizeY_; }
		int sizeZ() const { return sizeZ_; }

		const torch::Tensor& dataGpu() const { return dataGpu_; } //shape: 1*X*Y*Z
		const torch::Tensor& dataCpu() const { return dataCpu_; } //shape: 1*X*Y*Z

		void copyCpuToGpu();
	};
	
private:
	
	double worldSizeX_, worldSizeY_, worldSizeZ_;
	std::vector<std::unique_ptr<MipmapLevel>> levels_;
	
public:
	Volume();
	Volume(int sizeX, int sizeY, int sizeZ);
	~Volume();

	Volume(const Volume& other) = delete;
	Volume(Volume&& other) noexcept = delete;
	Volume& operator=(const Volume& other) = delete;
	Volume& operator=(Volume&& other) noexcept = delete;

	static constexpr int NO_COMPRESSION = 0;
	static constexpr int MAX_COMPRESSION = 9;
	
	/**
	 * Saves the volume to the file
	 */
	void save(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error,
		int compression = NO_COMPRESSION) const;
	/**
	 * Saves the volume to the file using default progress callbacks
	 */
	void save(const std::string& filename,
		int compression = NO_COMPRESSION) const;
	/**
	 * Loads and construct the volume
	 */
	Volume(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error);
	/**
	 * Loads and construct the volume using default progress callbacks
	 */
	explicit Volume(const std::string& filename);

	/**
	 * Creates the histogram of the volume.
	 */
	Volume::Histogram extractHistogram() const;

	double worldSizeX() const { return worldSizeX_; }
	double worldSizeY() const { return worldSizeY_; }
	double worldSizeZ() const { return worldSizeZ_; }
	float3 worldSize() const { return make_float3(worldSizeX_, worldSizeY_, worldSizeZ_); }
	void setWorldSizeX(double s) { worldSizeX_ = s; }
	void setWorldSizeY(double s) { worldSizeY_ = s; }
	void setWorldSizeZ(double s) { worldSizeZ_ = s; }

	int3 baseResolution() const {
		return make_int3(
			levels_[0]->sizeX(), levels_[0]->sizeY(), levels_[0]->sizeZ());
	}
	
	/**
	 * \brief Creates the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling.
	 * This function does nothing if that level is already created.
	 * \param level the mipmap level
	 */
	void createMipmapLevel(int level);

	typedef std::function<float(float3)> ImplicitFunction_t;
	
	/**
	 * \brief Creates a synthetic dataset using the
	 * implicit function 'f'.
	 * The function is called with positions in the range [boxMin, boxMax]
	 * (inclusive bounds), equal-spaced with a resolution of 'resolution'
	 * \param resolution the volume resolution
	 * \param boxMin the minimal coordinate
	 * \param boxMax the maximal coordinate
	 * \param f the generative function
	 * \return the volume
	 */
	static std::unique_ptr<Volume> createSyntheticDataset(
		int resolution, float boxMin, float boxMax,
		const ImplicitFunction_t& f);

	enum class ImplicitEquation
	{
		MARSCHNER_LOBB, //params "fM", "alpha"
		CUBE, //param "scale"
		SPHERE,
		INVERSE_SPHERE,
		DING_DONG,
		ENDRASS,
		BARTH,
		HEART,
		KLEINE,
		CASSINI,
		STEINER,
		CROSS_CAP,
		KUMMER,
		BLOBBY,
		TUBE,
		_NUM_IMPLICIT_EQUATION_
	};

	static std::unique_ptr<Volume> createImplicitDataset(
		int resolution, ImplicitEquation equation,
		const std::unordered_map<std::string, float>& params = {});

	/**
	 * Creates a volume from a raw buffer (e.g. from numpy).
	 *
	 * Indexing:
	 *     for(int x=0; x<sizes[0]; ++x) for (y...) for (z...)
	 *     float density = buffer[x*strides[0], y*strides[1], z*...];
	 * 
	 * \param buffer the float buffer with the contents
	 * \param sizes the size of the volume in X, Y, Z
	 * \param strides the strides of the buffer in X, Y, Z
	 */
	static std::unique_ptr<Volume> createFromBuffer(
		const real_t* buffer, long long int sizes[3], long long int strides[3]);
	static std::unique_ptr<Volume> createFromBuffer(
		const real_t* buffer, long int sizes[3], long int strides[3]);

private:
	bool mipmapCheckOrCreate(int level);

public:
	/**
	 * \brief Deletes all mipmap levels.
	 */
	void deleteAllMipmapLevels();

	/**
	 * \brief Returns the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * If the level is not created yet, <code>nullptr</code> is returned.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
	 */
	const MipmapLevel* getLevel(int level) const;
	/**
	 * \brief Returns the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * If the level is not created yet, <code>nullptr</code> is returned.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
	 */
	MipmapLevel* getLevel(int level);

	const torch::Tensor& dataGpu(int level) const;

	const torch::Tensor& dataCpu(int level) const;
};

END_RENDERER_NAMESPACE

#ifdef _WIN32
#pragma warning( pop )
#endif
