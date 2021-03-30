#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <renderer_settings.cuh>

namespace renderer {
	struct ShadingSettings;
}

namespace kernel
{
	/**
	 * \brief selects the output channel.
	 * \param inputTensor the input tensor of shape 1 * Height * Width * Channel
	 *  of type real_t (float or double), residing on the GPU
	 * \param outputBuffer the output texture, RGBA of uint8
	 */
	void copyOutputToTexture(
		const kernel::Tensor4Read& inputTensor,
		GLubyte* outputBuffer);

	/**
	 * \brief Converts the scalar field with positive and negative values
	 * into a color image based on a divergent color map.
	 * \param inputTensor the input tensor of shape 1 * Height * Width * 1
	 *  of type float, residing on the GPU
	 * \param minValue minimal value in the input tensor (<=0)
	 * \parma maxValue maximal value in the input tensor (>=0)
	 * \param outputBuffer the output texture, RGBA of uint8
	 */
	void divergentColorMap(
		const kernel::Tensor4Read& inputTensor,
		float minValue, float maxValue,
		GLubyte* outputBuffer);

	/**
	 * \brief  Fills the color map using tfTexture which is created
		according to control points. Filled color map is then displayed
		in TF Editor menu.
	 * \param colorMap surface object which makes it possible to modify color map via surface writes.
	 * \param tfTexture
	 * \param width width of color map
	 * \param height height of color map
	 */
	void fillColorMap(cudaSurfaceObject_t colorMap, cudaTextureObject_t tfTexture, int width, int height);

}
