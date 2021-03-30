#include "volume.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cuMat/src/Errors.h>
#include <filesystem>
#include <lz4cpp.hpp>

#include "renderer_settings.cuh"
#include "errors.h"

BEGIN_RENDERER_NAMESPACE

#if USE_DOUBLE_PRECISION==0
#define real_dtype c10::ScalarType::Float
#else
#define real_dtype c10::ScalarType::Double
#endif

Volume::MipmapLevel::MipmapLevel(Volume* parent, int sizeX, int sizeY, int sizeZ)
	: dataCpuBlob_(sizeX*sizeY*sizeZ)
	, dataCpu_(torch::from_blob(
		dataCpuBlob_.data(),
		{1, sizeX, sizeY, sizeZ},
		{sizeX*sizeY*sizeZ, 1, sizeX, sizeX*sizeY},
		at::TensorOptions().dtype(real_dtype).device(c10::kCPU)))
	, sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ)
	, cpuDataCounter_(0), gpuDataCounter_(0)
	, parent_(parent)
{

}

Volume::MipmapLevel::~MipmapLevel()
{
}

void Volume::MipmapLevel::copyCpuToGpu()
{
	if (gpuDataCounter_ == cpuDataCounter_ && dataGpu_.defined())
		return; //nothing changed
	gpuDataCounter_ = cpuDataCounter_;
	dataGpu_ = dataCpu_.to(c10::kCUDA);
}

Volume::Volume()
	: worldSizeX_(1), worldSizeY_(1), worldSizeZ_(1)
{
}

Volume::Volume(int sizeX, int sizeY, int sizeZ)
	: worldSizeX_(1), worldSizeY_(1), worldSizeZ_(1)
{
	levels_.push_back(std::unique_ptr<MipmapLevel>(new MipmapLevel(this, sizeX, sizeY, sizeZ)));
}

Volume::~Volume()
{
}

static const char MAGIC[] = "cvol";
enum DataTypeForIO
{
	TypeUChar,
	TypeUShort,
	TypeFloat,
	_TypeCount_
};
static int BytesPerType[DataTypeForIO::_TypeCount_] = {
	1, 2, 4
};

/*
 * FORMAT:
 * magic number "cvol", 4Bytes
 * sizeXYZ, 3*8 Bytes
 * voxelSizeXYZ, 3*8 Bytes,
 * datatype, 4Bytes
 * 8 bytes padding
 * ==> 64 Bytes header
 * Then follows the raw data
 */

void Volume::save(const std::string& filename,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error,
	int compression) const
{
#if USE_DOUBLE_PRECISION==1
	std::cerr << "Can't save volume in double precision, not implemeneted in the .cvol standard" << std::endl;
	return;
#endif
	assert(sizeof(size_t) == 8);
	assert(sizeof(double) == 8);
	if (compression < 0 || compression > MAX_COMPRESSION)
		throw std::runtime_error("Illegal compression factor");
	std::ofstream s(filename, std::fstream::binary);

	const MipmapLevel* data = getLevel(0);

	//header
	size_t sizeX = data->sizeX_;
	size_t sizeY = data->sizeY_;
	size_t sizeZ = data->sizeZ_;
	double voxelSizeX = worldSizeX_ / data->sizeX_;
	double voxelSizeY = worldSizeY_ / data->sizeY_;
	double voxelSizeZ = worldSizeZ_ / data->sizeZ_;
	s.write(MAGIC, 4);
	s.write(reinterpret_cast<const char*>(&sizeX), 8);
	s.write(reinterpret_cast<const char*>(&sizeY), 8);
	s.write(reinterpret_cast<const char*>(&sizeZ), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeX), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeY), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeZ), 8);
	int type = static_cast<int>(TypeFloat);
	s.write(reinterpret_cast<const char*>(&type), 4);
	char useCompression = compression > 0 ? 1 : 0;
	s.write(&useCompression, 1);
	char padding[8] = { 0 };
	s.write(padding, 7);

	//body
	progress(0.0f);
	const char* dataBlob = reinterpret_cast<const char*>(data->dataCpuBlob_.data());
	if (useCompression)
	{
		LZ4Compressor c(compression >= LZ4Compressor::MIN_COMPRESSION ? compression : LZ4Compressor::FAST_COMPRESSION);
		size_t dataToWrite = sizeof(float) * data->sizeX_ * data->sizeY_ * data->sizeZ_;
		int chunkSize = LZ4Compressor::MAX_CHUNK_SIZE;
		for (size_t offset = 0; offset < dataToWrite; offset += chunkSize)
		{
			const char* mem = dataBlob + offset;
			const int len = std::min(static_cast<int>(dataToWrite - offset), chunkSize);
			c.compress(s, mem, len);
			progress(offset / float(dataToWrite));
		}
	}
	else
	{
		size_t dataToWrite = sizeof(float) * data->sizeX_ * data->sizeY_;
		for (int z = 0; z < data->sizeZ_; ++z)
		{
			s.write(dataBlob + z * dataToWrite, dataToWrite);
			if (z % 10 == 0)
				progress(z / float(data->sizeZ_));
		}
	}
	progress(1.0f);
}

void Volume::save(const std::string& filename, int compression) const
{
	save(filename,
		[](float v) {},
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code) {throw std::runtime_error(msg.c_str()); },
		compression);
}

Volume::Volume(const std::string& filename,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error)
	: Volume()
{
	assert(sizeof(size_t) == 8);
	assert(sizeof(double) == 8);
	std::ifstream s(filename, std::fstream::binary);

	//header
	char magic[4];
	s.read(magic, 4);
	if (memcmp(MAGIC, magic, 4) != 0)
	{
		error("Illegal magic number", -1);
	}
	size_t sizeX, sizeY, sizeZ;
	double voxelSizeX, voxelSizeY, voxelSizeZ;
	char useCompression;
	s.read(reinterpret_cast<char*>(&sizeX), 8);
	s.read(reinterpret_cast<char*>(&sizeY), 8);
	s.read(reinterpret_cast<char*>(&sizeZ), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeX), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeY), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeZ), 8);
	int type;
	s.read(reinterpret_cast<char*>(&type), 4);
	s.read(&useCompression, 1);
	s.ignore(7);
	DataTypeForIO type_ = static_cast<DataTypeForIO>(type);
	if (sizeX * sizeY * sizeZ > std::numeric_limits<int>::max())
		throw std::runtime_error("Dataset is too big, must fit into an 32-bit signed int");

	//create level
	levels_.push_back(std::unique_ptr<MipmapLevel>(new MipmapLevel(this, sizeX, sizeY, sizeZ)));
	MipmapLevel* data = levels_[0].get();
	worldSizeX_ = voxelSizeX * sizeX;
	worldSizeY_ = voxelSizeY * sizeY;
	worldSizeZ_ = voxelSizeZ * sizeZ;

	//load first
	std::vector<char> rawData(BytesPerType[type] * sizeX * sizeY * sizeZ);
	progress(0.0f);
	if (useCompression) {
		LZ4Decompressor d;
		size_t dataToRead = BytesPerType[type_] * data->sizeX_ * data->sizeY_ * data->sizeZ_;
		for (size_t offset = 0; offset < dataToRead;)
		{
			char* mem = rawData.data() + offset;
			const int len = std::min(
				static_cast<int>(dataToRead - offset),
				std::numeric_limits<int>::max());
			int chunkSize = d.decompress(mem, len, s);
			progress(offset / float(dataToRead));
			offset += chunkSize;
		}
	}
	else
	{
		size_t dataToRead = BytesPerType[type_] * data->sizeX_ * data->sizeY_;
		for (int z = 0; z < data->sizeZ_; ++z)
		{
			s.read(rawData.data() + z * dataToRead, dataToRead);
			if (z % 10 == 0)
				progress(z / float(data->sizeZ_));
		}
	}

	//convert
	
	int sizeXYZ = static_cast<int>(sizeX * sizeY * sizeZ);
#if USE_DOUBLE_PRECISION==0
	float* data_ptr = data->dataCpuBlob_.data();
	if (type_ == TypeFloat) {
		memcpy(data_ptr, rawData.data(), sizeof(float) * sizeXYZ);
	}
	else if (type_ == TypeUChar)
	{
		const unsigned char* src = reinterpret_cast<const unsigned char*>(rawData.data());
#pragma omp parallel for
		for (int i = 0; i < sizeXYZ; ++i)
			data_ptr[i] = src[i] / 255.0f;
	}
	else if (type_ == TypeUShort)
	{
		const unsigned short* src = reinterpret_cast<const unsigned short*>(rawData.data());
#pragma omp parallel for
		for (int i = 0; i < sizeXYZ; ++i)
			data_ptr[i] = src[i] / 65535.0f;
	}
#else
	double* data_ptr = data->dataCpuBlob_.data();
	int sliceSize = data->sizeX_ * data->sizeY_;
	if (type_ == TypeFloat)
	{
		const float* src = reinterpret_cast<const float*>(rawData.data());
#pragma omp parallel for
		for (int i = 0; i < sizeXYZ; ++i)
			data_ptr[i] = src[i];
	}
	else if (type_ == TypeUChar)
	{
		const unsigned char* src = reinterpret_cast<const unsigned char*>(rawData.data());
#pragma omp parallel for
		for (int i = 0; i < sizeXYZ; ++i)
			data_ptr[i] = src[i] / 255.0f;
	}
	else if (type_ == TypeUShort)
	{
		const unsigned short* src = reinterpret_cast<const unsigned short*>(rawData.data());
#pragma omp parallel for
		for (int i = 0; i < sizeXYZ; ++i)
			data_ptr[i] = src[i] / 65535.0f;
	}
#endif
	progress(1.0f);
}

Volume::Volume(const std::string& filename)
	: Volume(filename,
		[](float v) {},
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code) {throw std::runtime_error(msg.c_str()); })
{}

Volume::Histogram Volume::extractHistogram() const
{
	auto data = getLevel(0);
	torch::PackedTensorAccessor32<real_t, 4> volumeCPU = data->dataCpu().packed_accessor32<real_t, 4>();
	int sizeX = volumeCPU.size(1);
	int sizeY = volumeCPU.size(2);
	int sizeZ = volumeCPU.size(3);

	Volume::Histogram histogram;

	//1. min/max
	real_t minDensity = FLT_MAX;
	real_t maxDensity = 0;
	int numNonZeros = 0;
#pragma omp parallel
	{
		real_t localMinDensity = FLT_MAX;
		real_t localMaxDensity = 0;
		int localNumNonZeros = 0;
#pragma omp for
		for (int x = 0; x < sizeX; ++x) {
			for (int y = 0; y < sizeY; ++y) for (int z = 0; z < sizeZ; ++z)
			{
				real_t density = volumeCPU[0][x][y][z];
				localMinDensity = fminf(localMinDensity, density);
				localMaxDensity = fmaxf(localMaxDensity, density);
				if (density > 0) localNumNonZeros++;
			}
		}
#pragma omp critical
		{
			if (localMinDensity < minDensity) minDensity = localMinDensity;
			if (localMaxDensity > maxDensity) maxDensity = localMaxDensity;
			numNonZeros += localNumNonZeros;
		}
	}

	histogram.maxDensity = maxDensity;
	histogram.minDensity = minDensity;
	histogram.numOfNonzeroVoxels = numNonZeros;
	const real_t binFrac = 1.0f / numNonZeros;
	
	//2. extract histogram
#pragma omp parallel
	{
		float localBins[Histogram::getNumOfBins()] {0};
#pragma omp for
		for (int x = 0; x < sizeX; ++x) {
			for (int y = 0; y < sizeY; ++y) for (int z = 0; z < sizeZ; ++z)
			{
				float density = volumeCPU[0][x][y][z];
				if (density > 0)
				{
					auto densityWidthResolution = (maxDensity - minDensity) / Histogram::getNumOfBins();
					auto binIdx = static_cast<int>((density - minDensity) / densityWidthResolution);
					//Precaution against floating-point errors
					binIdx = binIdx >= Histogram::getNumOfBins() ? (Histogram::getNumOfBins() - 1) : binIdx;
					localBins[binIdx] += binFrac;
				}
			}
		}
#pragma omp critical
		{
			for (int i = 0; i < Histogram::getNumOfBins(); ++i)
				histogram.bins[i] += localBins[i];
		}
	}

	return histogram;
}

void Volume::createMipmapLevel(int level)
{
	if (!mipmapCheckOrCreate(level)) return; //already available
	const auto& src = levels_[0];
	const auto& dst = levels_[level];
	at::adaptive_avg_pool3d_out(dst->dataCpu_, src->dataCpu_, dst->dataCpu_.sizes());
}

static inline float sqr(float s) { return s * s; }

std::unique_ptr<Volume> Volume::createSyntheticDataset(int resolution, float boxMin, float boxMax,
	const ImplicitFunction_t& f)
{
	auto vol = std::make_unique<Volume>(resolution, resolution, resolution);
	auto level = vol->getLevel(0);
	auto data = level->dataCpu().accessor<real_t, 4>();
	float scale = (boxMax - boxMin) / (resolution - 1);
#pragma omp parallel for
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
			{
				float3 xyz = make_float3(
					boxMin + x * scale, boxMin + y * scale, boxMin + z * scale);
				real_t v = f(xyz);
				data[0][x][y][z] = v;
			}
	return vol;
}

std::unique_ptr<Volume> Volume::createFromBuffer(const real_t* buffer, long long int sizes[3], long long int strides[3])
{
	//std::cout << "From Buffer: sizes=(" << sizes[0] << "," << sizes[1] << "," << sizes[2]
	//	<< "), strides=(" << strides[0] << "," << strides[1] << "," << strides[2] << ")" << std::endl;
	auto vol = std::make_unique<Volume>(
		static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), static_cast<int>(sizes[2]));
	auto level = vol->getLevel(0);
	auto data = level->dataCpu().accessor<real_t, 4>();
#pragma omp parallel for
	for (int x = 0; x < sizes[0]; ++x)
		for (int y = 0; y < sizes[1]; ++y)
			for (int z = 0; z < sizes[2]; ++z)
				data[0][x][y][z] = buffer[x * strides[0] + y * strides[1] + z * strides[2]];
	return vol;
}
std::unique_ptr<Volume> Volume::createFromBuffer(const real_t* buffer, long int sizes[3], long int strides[3])
{
	//std::cout << "From Buffer: sizes=(" << sizes[0] << "," << sizes[1] << "," << sizes[2]
	//	<< "), strides=(" << strides[0] << "," << strides[1] << "," << strides[2] << ")" << std::endl;
	auto vol = std::make_unique<Volume>(
		static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), static_cast<int>(sizes[2]));
	auto level = vol->getLevel(0);
	auto data = level->dataCpu().accessor<real_t, 4>();
#pragma omp parallel for
	for (int x = 0; x < sizes[0]; ++x)
		for (int y = 0; y < sizes[1]; ++y)
			for (int z = 0; z < sizes[2]; ++z)
				data[0][x][y][z] = buffer[x * strides[0] + y * strides[1] + z * strides[2]];
	return vol;
}

bool Volume::mipmapCheckOrCreate(int level)
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level < levels_.size() && levels_[level]) return false; //already available

	//create storage
	if (level >= levels_.size()) levels_.resize(level + 1);
	size_t newSizeX = std::max(1, levels_[0]->sizeX_ / (level + 1));
	size_t newSizeY = std::max(1, levels_[0]->sizeY_ / (level + 1));
	size_t newSizeZ = std::max(1, levels_[0]->sizeZ_ / (level + 1));
	levels_[level] = std::unique_ptr<MipmapLevel>(new MipmapLevel(this, newSizeX, newSizeY, newSizeZ));
	return true;
}

void Volume::deleteAllMipmapLevels()
{
	levels_.resize(1); //just keep the first level = original data
}

const Volume::MipmapLevel* Volume::getLevel(int level) const
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level].get();
}

Volume::MipmapLevel* Volume::getLevel(int level)
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level].get();
}

const torch::Tensor& Volume::dataGpu(int level) const
{
	const auto* lvl = getLevel(level);
	TORCH_CHECK(lvl, "Mipmap Level ", level, " was not created");
	return lvl->dataGpu();
}

const torch::Tensor& Volume::dataCpu(int level) const
{
	const auto* lvl = getLevel(level);
	TORCH_CHECK(lvl, "Mipmap Level ", level, " was not created");
	return lvl->dataCpu();
}

static void printProgress(const std::string& prefix, float progress)
{
	int barWidth = 50;
	std::cout << prefix << " [";
	int pos = static_cast<int>(barWidth * progress);
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();
	if (progress >= 1) std::cout << std::endl;
}

END_RENDERER_NAMESPACE
