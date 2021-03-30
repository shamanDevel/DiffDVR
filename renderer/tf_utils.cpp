#include "tf_utils.h"

#include <cuMat/src/Context.h>
#include <algorithm>

#include "pytorch_utils.h"
#include <torch/torch.h>
#include <magic_enum.hpp>

BEGIN_RENDERER_NAMESPACE

std::vector<TFPoint> TFUtils::assembleFromSettings(const std::vector<real3>& colorValuesLab_,
	const std::vector<real_t>& colorPositions_, const std::vector<real_t>& opacityValues_,
	const std::vector<real_t>& opacityPositions_, real_t minDensity, real_t maxDensity, real_t opacityScaling,
	bool purgeZeroOpacityRegions)
{
	//copy vectors, they will be modified
	auto colorValues = colorValuesLab_;
	auto colorPositions = colorPositions_;
	auto opacityValues = opacityValues_;
	auto opacityPositions = opacityPositions_;
	
	std::vector<TFPoint> points;

	//transform by minDensity/maxDensity
	for (real_t& d : colorPositions)
		d = minDensity + d * (maxDensity - minDensity);
	for (real_t& d : opacityPositions)
		d = minDensity + d * (maxDensity - minDensity);

	//add control points at t=0 if not existing
	//but not directly 0 and 1, better -1 and 2 as this is better for the iso-intersections
	if (colorPositions.front() > 0)
	{
		colorPositions.insert(colorPositions.begin(), -1.0f);
		colorValues.insert(colorValues.begin(), colorValues.front());
	}
	if (opacityPositions.front() > 0)
	{
		opacityPositions.insert(opacityPositions.begin(), -1.0f);
		opacityValues.insert(opacityValues.begin(), opacityValues.front());
	}
	//same with t=1
	if (colorPositions.back() < 1)
	{
		colorPositions.push_back(2);
		colorValues.push_back(colorValues.back());
	}
	if (opacityPositions.back() < 1)
	{
		opacityPositions.push_back(2);
		opacityValues.push_back(opacityValues.back());
	}

	//first point at pos=0
	points.push_back({ colorPositions[0],
		make_real4(renderer::labToRgb(colorValues[0]), opacityValues[0]) });

	//assemble the points
	int64_t iOpacity = 0; //next indices
	int64_t iColor = 0;
	while (iOpacity < opacityPositions.size() - 1 && iColor < colorPositions.size() - 1)
	{
		if (opacityPositions[iOpacity + 1] < colorPositions[iColor + 1])
		{
			//next point is an opacity point
			points.push_back({
				opacityPositions[iOpacity + 1] ,
				make_real4(
					renderer::labToRgb(lerp(
						colorValues[iColor],
						colorValues[iColor + 1],
						(opacityPositions[iOpacity + 1] - colorPositions[iColor]) / (colorPositions[iColor + 1] - colorPositions[iColor]))),
					opacityValues[iOpacity + 1]) });
			iOpacity++;
		}
		else
		{
			points.push_back({
				colorPositions[iColor + 1],
				make_real4(
					renderer::labToRgb(colorValues[iColor + 1]),
					lerp(
						opacityValues[iOpacity],
						opacityValues[iOpacity + 1],
						(colorPositions[iColor + 1] - opacityPositions[iOpacity]) / (opacityPositions[iOpacity + 1] - opacityPositions[iOpacity])))
				});
			iColor++;
		}

	}

	//filter the points
	//removes all color control points in areas of zero density
	//and also remove points that are close together
	if (purgeZeroOpacityRegions) {
		int numPointsRemoved = 0;
		constexpr real_t EPS = 1e-7;
		for (int64_t i = 0; i < static_cast<int64_t>(points.size()) - 2; )
		{
			if ((points[i].val.w < EPS && points[i + 1].val.w < EPS && points[i + 2].val.w < EPS) ||
				(points[i+1].pos-points[i].pos < EPS)) {
				points.erase(points.begin() + (i + 1));
				numPointsRemoved++;
			}
			else
				i++;
		}
		//std::cout << numPointsRemoved << " points removed with zero density" << std::endl;
	}

	//clamp color and scale opacity
	for (TFPoint& p : points)
	{
		p.val = clamp(p.val, 0.0f, 1.0f-FLT_EPSILON);
		p.val.w *= opacityScaling;
	}
	
	return points;
}

torch::Tensor TFUtils::getPiecewiseTensor(const std::vector<TFPoint>& points)
{
	int numPoints = static_cast<int>(points.size());
	torch::Tensor t = torch::empty(
		{ 1, numPoints, 5 }, 
		at::TensorOptions().dtype(real_dtype));
	auto acc = t.packed_accessor32<real_t, 3>();
	for (int i = 0; i < numPoints; ++i)
	{
		acc[0][i][0] = points[i].val.x;
		acc[0][i][1] = points[i].val.y;
		acc[0][i][2] = points[i].val.z;
		acc[0][i][3] = points[i].val.w;
		acc[0][i][4] = points[i].pos;
	}
	return t;
}

torch::Tensor TFUtils::getTextureTensor(const std::vector<TFPoint>& points, int resolution)
{
	torch::Tensor t = torch::empty(
		{ 1, resolution, 4 }, 
		at::TensorOptions().dtype(real_dtype));
	auto acc = t.packed_accessor32<real_t, 3>();
	int numPoints = static_cast<int>(points.size());

	for (int i = 0; i < resolution; ++i)
	{
		real_t density = (i + 0.5f) / resolution;
		//find control point
		int idx;
		for (idx = 0; idx < numPoints - 2; ++idx)
			if (points[idx + 1].pos > density) break;
		//interpolate
		const real_t pLow = points[idx].pos;
		const real_t pHigh = points[idx + 1].pos;
		assert(pLow <= density);
		assert(pHigh >= density);
		const real4 vLow = points[idx].val;
		const real4 vHigh = points[idx + 1].val;

		const real_t frac = (density - pLow) / (pHigh - pLow);
		real4 v = (1 - frac) * vLow + frac * vHigh;
		acc[0][i][0] = v.x;
		acc[0][i][1] = v.y;
		acc[0][i][2] = v.z;
		acc[0][i][3] = v.w;
	}

	return t;
}

class PreshadeVolumeFunction : public torch::autograd::Function<PreshadeVolumeFunction>
{
public:
	static torch::autograd::variable_list forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& densityVolume, const torch::Tensor& tf,
		kernel::TFMode tfMode)
	{
		CHECK_DIM(densityVolume, 4);
		CHECK_DTYPE(densityVolume, real_dtype);
		bool cuda = densityVolume.is_cuda();
		CHECK_SIZE(densityVolume, 0, 1);
		int64_t X = densityVolume.size(1);
		int64_t Y = densityVolume.size(2);
		int64_t Z = densityVolume.size(3);

		CHECK_DIM(tf, 3);
		CHECK_CUDA(tf, cuda);
		CHECK_SIZE(tf, 0, 1);
		switch (tfMode)
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
		case kernel::TFGaussianLog:
			CHECK_SIZE(tf, 2, 6);
			break;
		default:
			throw std::runtime_error(std::string("unknown tf enum value: ")+
				std::string(magic_enum::enum_name(tfMode)));
		}

		torch::Tensor out = torch::zeros({ 4, X, Y, Z }, densityVolume.options());

		const auto volumeAcc = accessor<kernel::Tensor4Read>(densityVolume);
		const auto tfAcc = accessor<kernel::Tensor3Read>(tf);
		auto colorAcc = accessor<kernel::Tensor4RW>(out);
		kernel::PreshadeVolume(volumeAcc, tfAcc, colorAcc, tfMode, cuda);
		ctx->save_for_backward({ densityVolume, tf });
		ctx->saved_data["tfMode"] = static_cast<int>(tfMode);
		
		return { out };
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs)
	{
		auto saved = ctx->get_saved_variables();
		auto densityVolume = saved[0];
		auto tf = saved[1];
		kernel::TFMode tfMode = static_cast<kernel::TFMode>(ctx->saved_data["tfMode"].toInt());

		auto grad_output = grad_outputs[0];
		bool cuda = grad_output.is_cuda();

		auto adj_volume = torch::empty_like(densityVolume);

		const auto volumeAcc = accessor<kernel::Tensor4Read>(densityVolume);
		const auto tfAcc = accessor<kernel::Tensor3Read>(tf);
		auto adjColorAcc = accessor<kernel::Tensor4Read>(grad_output);
		auto adjVolume = accessor<kernel::Tensor4RW>(adj_volume);
		kernel::PreshadeVolumeAdj(volumeAcc, tfAcc, adjColorAcc, adjVolume, tfMode, cuda);
		
		return { adj_volume, torch::Tensor(), torch::Tensor() };
	}
};

torch::Tensor TFUtils::preshadeVolume(const torch::Tensor& densityVolume, const torch::Tensor& tf,
	kernel::TFMode tfMode)
{
	return PreshadeVolumeFunction::apply(densityVolume, tf, tfMode)[0];
}

torch::Tensor TFUtils::findBestFit(
	const torch::Tensor& colorVolume, const torch::Tensor& tf,
	kernel::TFMode tfMode, int numSamples, real_t opacityWeighting,
	const torch::Tensor* previousDensities, real_t neighborWeighting)
{
	TORCH_CHECK(opacityWeighting >= 0, "opacity weighting must be >= 0, but is ", opacityWeighting);
	TORCH_CHECK(numSamples >= 1, "numSamples must be >= 1, but is ", numSamples);
	
	CHECK_DIM(colorVolume, 4);
	CHECK_DTYPE(colorVolume, real_dtype);
	bool cuda = colorVolume.is_cuda();
	CHECK_SIZE(colorVolume, 0, 4);
	int64_t X = colorVolume.size(1);
	int64_t Y = colorVolume.size(2);
	int64_t Z = colorVolume.size(3);

	CHECK_DIM(tf, 3);
	CHECK_CUDA(tf, cuda);
	CHECK_SIZE(tf, 0, 1);
	switch (tfMode)
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
	default:
		throw std::runtime_error(std::string("unknown tf enum value: ") +
			std::string(magic_enum::enum_name(tfMode)));
	}

	torch::Tensor out = torch::zeros({ 1, X, Y, Z }, colorVolume.options());
	at::Tensor rnd = at::randint(
		std::numeric_limits<int64_t>::max(), 1024,
		c10::TensorOptions().device(c10::kCPU).dtype(c10::kLong));
	if (cuda)
		rnd = rnd.to(c10::kCUDA);

	const auto seedAcc = accessor<kernel::LTensor1Read>(rnd);
	const auto colorVolumeAcc = accessor<kernel::Tensor4Read>(colorVolume);
	const auto tfAcc = accessor<kernel::Tensor3Read>(tf);
	auto densityVolumeAcc = accessor<kernel::Tensor4RW>(out);

	if (previousDensities == nullptr) {
		kernel::FindBestFit(
			colorVolumeAcc, tfAcc, densityVolumeAcc, seedAcc,
			tfMode, numSamples, opacityWeighting, cuda);
	}
	else
	{
		const auto pd = *previousDensities;
		CHECK_DIM(pd, 4);
		CHECK_CUDA(pd, cuda);
		CHECK_SIZE(pd, 0, 1);
		CHECK_SIZE(pd, 1, X);
		CHECK_SIZE(pd, 2, Y);
		CHECK_SIZE(pd, 3, Z);
		const auto previousDensityAcc = accessor<kernel::Tensor4RW>(pd);
		//std::cout << "sizes: " << previousDensityAcc.size(1) << ", " << previousDensityAcc.size(2)
		//	<< ", " << previousDensityAcc.size(3) << std::endl;
		kernel::FindBestFitWithComparison(
			colorVolumeAcc, tfAcc, previousDensityAcc, densityVolumeAcc, seedAcc,
			tfMode, numSamples, opacityWeighting, neighborWeighting, cuda);
	}

	return out;
}

END_RENDERER_NAMESPACE
