#include "visualizer_kernels.h"
#include <helper_math.cuh>
#include <cuMat/src/Context.h>

#include <ATen/cuda/CUDAContext.h>
#include <renderer_settings.cuh>
#include <renderer_utils.cuh>

__global__ void CopyOutputToTextureKernel(
	dim3 virtual_size,
	kernel::Tensor4Read input,
	unsigned int* output)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
	{
		float r = input[0][y][x][0];
		float g = input[0][y][x][1];
		float b = input[0][y][x][2];
		float a = input[0][y][x][3];
		//printf("%d %d -> %f\n", int(x), int(y), float(a));
		output[y * input.size(2) + x] = kernel::rgbaToInt(r, g, b, a);
	}
	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::copyOutputToTexture(
	const kernel::Tensor4Read& inputTensor, GLubyte* outputBuffer)
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	unsigned width = inputTensor.size(2);
	unsigned height = inputTensor.size(1);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, CopyOutputToTextureKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	CopyOutputToTextureKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, inputTensor,
		 reinterpret_cast<unsigned int*>(outputBuffer));
	CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

//From https://github.com/berendeanicolae/ColorSpace/blob/master/src/Conversion.cpp
	//Assumes color channels are in [0,1]
__device__ __host__
inline float3 rgbToXyz(const float3& rgb)
{
	auto r = ((rgb.x > 0.04045f) ? powf((rgb.x + 0.055f) / 1.055f, 2.4f) : (rgb.x / 12.92f)) * 100.0f;
	auto g = ((rgb.y > 0.04045f) ? powf((rgb.y + 0.055f) / 1.055f, 2.4f) : (rgb.y / 12.92f)) * 100.0f;
	auto b = ((rgb.z > 0.04045f) ? powf((rgb.z + 0.055f) / 1.055f, 2.4f) : (rgb.z / 12.92f)) * 100.0f;

	auto x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
	auto y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
	auto z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

	return { x, y, z };
}

//Output color channels are in [0,1]
__device__ __host__
inline float3 xyzToRgb(const float3& xyz)
{
	auto x = xyz.x / 100.0f;
	auto y = xyz.y / 100.0f;
	auto z = xyz.z / 100.0f;

	auto r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
	auto g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
	auto b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

	r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
	g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
	b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

	return { r, g, b };
}

__device__ __host__
inline float3 xyzToLab(const float3& xyz)
{
	auto x = xyz.x / 95.047f;
	auto y = xyz.y / 100.00f;
	auto z = xyz.z / 108.883f;

	x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + 16.0f / 116.0f);
	y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + 16.0f / 116.0f);
	z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + 16.0f / 116.0f);

	return { (116.0f * y) - 16.0f, 500.0f * (x - y), 200.0f * (y - z) };
}

__device__ __host__
inline float3 rgbToLab(const float3& rgb)
{
	return xyzToLab(rgbToXyz(rgb));
}

__device__ __host__
inline float3 labToXyz(const float3& lab)
{
	auto y = (lab.x + 16.0f) / 116.0f;
	auto x = lab.y / 500.0f + y;
	auto z = y - lab.z / 200.0f;

	auto x3 = x * x * x;
	auto y3 = y * y * y;
	auto z3 = z * z * z;

	x = ((x3 > 0.008856f) ? x3 : ((x - 16.0f / 116.0f) / 7.787f)) * 95.047f;
	y = ((y3 > 0.008856f) ? y3 : ((y - 16.0f / 116.0f) / 7.787f)) * 100.0f;
	z = ((z3 > 0.008856f) ? z3 : ((z - 16.0f / 116.0f) / 7.787f)) * 108.883f;

	return { x, y, z };
}

__device__ __host__
inline float3 labToRgb(const float3& lab)
{
	return xyzToRgb(labToXyz(lab));
}

//https://www.kennethmoreland.com/color-maps/ColorMapsExpanded.pdf

__inline__ __device__ float3 rgb2msh(const float3& rgb)
{	
	float3 lab = rgbToLab(rgb);
	float M = length(lab);
	float s = acosf(lab.x / M);
	float h = atanf(lab.z / lab.y);
	return make_float3(M, s, h);
}

__inline__ __device__ float3 msh2rgb(const float3& msh)
{
	float L = msh.x * cosf(msh.y);
	float a = msh.x * sinf(msh.y) * cosf(msh.z);
	float b = msh.x * sinf(msh.y) * sinf(msh.z);
	return labToRgb(make_float3(L, a, b));
}

__inline__ __device__ float adjustHue(const float3& sat, float Munsat)
{
	if (sat.x >= Munsat)
		return sat.z;
	else
	{
		float hSpin = (sat.y * sqrtf(Munsat * Munsat - sat.x * sat.x)) / (sat.x * sinf(sat.y));
		if (sat.z > -M_PI / 3)
			return sat.z + hSpin;
		else
			return sat.z - hSpin;
	}
}

__inline__ __device__ float3 interpolateColor(
	const float3& a, const float3& b, float interp)
{
	float3 msh1 = rgb2msh(a);
	float3 msh2 = rgb2msh(b);
	//place white in the middle
	float Mmid = fmaxf(msh1.x, fmaxf(msh2.x, 88));
	if (interp<0.5)
	{
		msh2 = make_float3(Mmid, 0, 0);
		interp = interp * 2;
	} else
	{
		msh1 = make_float3(Mmid, 0, 0);
		interp = interp * 2 - 1;
	}
	//adjust hue
	if (msh1.y < 0.05 && msh2.y>0.05)
		msh1.z = adjustHue(msh2, msh1.x);
	if (msh2.y < 0.05 && msh1.y > 0.05)
		msh2.z = adjustHue(msh1, msh2.x);
	//linear interpolation
	float3 msh = lerp(msh1, msh2, interp);
	return msh2rgb(msh);
}

__inline__ __device__ float smootherstep(float edge0, float edge1, float x) {
	// Scale, and clamp x to 0..1 range
	x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	// Evaluate polynomial
	return x * x * x * (x * (x * 6 - 15) + 10);
}

__global__ void DivergentColorMapKernel(
	dim3 virtual_size,
	kernel::Tensor4Read input,
	float minValue, float maxValue,
	unsigned int* output)
{
	const float3 rgb1 = make_float3(0, 0, 1); //negative values
	const float3 rgb2 = make_float3(1, 0, 0); //positive values
	
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
	{
		float val = input[0][y][x][0];

		if (val < 0)
		{
			//map [minValue, 0] to [0, 0.5]
			val = 0.5f - 0.5f * val / fminf(minValue, -1e-5f);
		}
		else
		{
			//map [0, maxValue] to [0.5, 1.0]
			val = 0.5f + 0.5f * val / fmaxf(maxValue, 1e-5f);
		}
		val = smootherstep(0.0f, 1.0f, val);
		float3 rgb = interpolateColor(rgb1, rgb2, val)/1.66f;
		rgb = clamp(rgb, 0.0f, 1.0f);
		
		output[y * input.size(2) + x] = kernel::rgbaToInt(rgb.x, rgb.y, rgb.z, 1.0f);
	}
	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::divergentColorMap(const kernel::Tensor4Read& inputTensor, float minValue, float maxValue,
	GLubyte* outputBuffer)
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	unsigned width = inputTensor.size(2);
	unsigned height = inputTensor.size(1);
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, DivergentColorMapKernel);
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	DivergentColorMapKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, inputTensor, minValue, maxValue,
		 reinterpret_cast<unsigned int*>(outputBuffer));
	CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void FillColorMapKernel(
	dim3 virtualSize,
	cudaSurfaceObject_t surface,
	cudaTextureObject_t tfTexture)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtualSize)

		auto density = x / static_cast<float>(virtualSize.x);
		auto rgbo = tex1D<float4>(tfTexture, density);

		surf2Dwrite(kernel::rgbaToInt(rgbo.x, rgbo.y, rgbo.z, 1.0f), surface, x * 4, y);

	CUMAT_KERNEL_2D_LOOP_END
}

void kernel::fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, FillColorMapKernel);
	cudaStream_t stream = ctx.stream();
	FillColorMapKernel
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, colorMap, tfTexture);
	CUMAT_CHECK_ERROR();
}


