#include "losses.h"

#include <torch/torch.h>
#include "pytorch_utils.h"

BEGIN_RENDERER_NAMESPACE

class MulLogFunction : public torch::autograd::Function<MulLogFunction> {
public:
	// Note that both forward and backward are static functions

	// bias is an optional argument
	static torch::autograd::variable_list forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& t) {

		torch::Tensor output = torch::empty_like(t);
		auto iter = at::TensorIterator::unary_op(output, t);
		kernel::mulLogForward(iter, t.device().is_cuda());

		ctx->save_for_backward({ t });
		
		return { output };
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto t = saved[0];

		auto grad_output = grad_outputs[0];
		bool cuda = t.device().is_cuda();

		torch::Tensor grad_t = torch::empty_like(t);

		auto iter = at::TensorIterator::binary_float_op(grad_t, t, grad_output);
		kernel::mulLogBackward(iter, cuda);

		//return torch::Tensor() for non-optimized arguments
		return { grad_t };
	}
};

torch::Tensor mul_log(const torch::Tensor& t)
{
	return MulLogFunction::apply(t)[0];
}

class LogMSEFunction : public torch::autograd::Function<LogMSEFunction> {
public:
	// Note that both forward and backward are static functions

	// bias is an optional argument
	static torch::autograd::variable_list forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& logX, const torch::Tensor& logY)
	{
		TORCH_CHECK(logX.sizes() == logY.sizes(), "both args must have the same shape");
		TORCH_CHECK(logX.device() == logY.device(), "both args must be on the same device");
		TORCH_CHECK(logX.dtype() == logY.dtype(), "both args must have the same dtype")

		torch::Tensor output = torch::empty_like(logX);
		auto iter = at::TensorIterator::binary_op(output, logX, logY);
		kernel::logMSEForward(iter, logX.device().is_cuda());

		ctx->save_for_backward({ logX, logY });

		return { output };
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto logX = saved[0];
		auto logY = saved[1];

		auto grad_output = grad_outputs[0];
		bool cuda = logX.device().is_cuda();

		torch::Tensor grad_logX = torch::empty_like(logX);
		torch::Tensor grad_logY = torch::empty_like(logY);

		auto iter = at::TensorIteratorConfig()
			.add_output(grad_logX).add_output(grad_logY) //you need to add outputs before inputs. Why?!?
			.add_input(logX).add_input(logY).add_input(grad_output)
			.build();
		kernel::logMSEBackward(iter, cuda);

		return { grad_logX, grad_logY };
	}
};

torch::Tensor logMSE(const torch::Tensor& logX, const torch::Tensor& logY)
{
	return LogMSEFunction::apply(logX, logY)[0];
}

class LogL1Function : public torch::autograd::Function<LogL1Function> {
public:
	// Note that both forward and backward are static functions

	// bias is an optional argument
	static torch::autograd::variable_list forward(
		torch::autograd::AutogradContext* ctx,
		const torch::Tensor& logX, const torch::Tensor& logY)
	{
		TORCH_CHECK(logX.sizes() == logY.sizes(), "both args must have the same shape");
		TORCH_CHECK(logX.device() == logY.device(), "both args must be on the same device");
		TORCH_CHECK(logX.dtype() == logY.dtype(), "both args must have the same dtype")

			torch::Tensor output = torch::empty_like(logX);
		auto iter = at::TensorIterator::binary_op(output, logX, logY);
		kernel::logL1Forward(iter, logX.device().is_cuda());

		ctx->save_for_backward({ logX, logY });

		return { output };
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::tensor_list grad_outputs) {
		auto saved = ctx->get_saved_variables();
		auto logX = saved[0];
		auto logY = saved[1];

		auto grad_output = grad_outputs[0];
		bool cuda = logX.device().is_cuda();

		torch::Tensor grad_logX = torch::empty_like(logX);
		torch::Tensor grad_logY = torch::empty_like(logY);

		auto iter = at::TensorIteratorConfig()
			.add_output(grad_logX).add_output(grad_logY) //you need to add outputs before inputs. Why?!?
			.add_input(logX).add_input(logY).add_input(grad_output)
			.build();
		kernel::logL1Backward(iter, cuda);

		return { grad_logX, grad_logY };
	}
};

torch::Tensor logL1(const torch::Tensor& logX, const torch::Tensor& logY)
{
	return LogL1Function::apply(logX, logY)[0];
}

torch::Tensor sampleImportance(const torch::Tensor& sampleLocations, const torch::Tensor& densityVolume,
	const torch::Tensor& tf, kernel::TFMode tfMode, float weightUniform, float weightDensityGradient,
	float weightOpacity, float weightOpacityGradient, int seed)
{
	CHECK_DIM(densityVolume, 4);
	CHECK_DTYPE(densityVolume, real_dtype);
	bool cuda = densityVolume.is_cuda();
	TORCH_CHECK(cuda, "sampleImportance is only supported on CUDA");
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
		CHECK_SIZE(tf, 2, 6);
		break;
	default:
		throw std::runtime_error("unknown tf enum value");
	}

	CHECK_DIM(sampleLocations, 2);
	CHECK_CUDA(sampleLocations, cuda);
	CHECK_SIZE(sampleLocations, 1, 3);
	int64_t N = sampleLocations.size(0);

	TORCH_CHECK(weightUniform >= 0, "weights must be non-negative");
	TORCH_CHECK(weightDensityGradient >= 0, "weights must be non-negative");
	TORCH_CHECK(weightOpacity >= 0, "weights must be non-negative");
	TORCH_CHECK(weightOpacityGradient >= 0, "weights must be non-negative");
	//make cummulative
	weightDensityGradient += weightUniform;
	weightOpacity += weightDensityGradient;
	weightOpacityGradient += weightOpacity;
	float weightSum = weightOpacityGradient;
	TORCH_CHECK(weightSum > 0, "at least one weight parameter must be positive");
	//normalize weights
	weightUniform /= weightSum;
	weightDensityGradient /= weightSum;
	weightOpacity /= weightSum;
	weightOpacityGradient /= weightSum; //=1

	//std::cout << "weights: " << weightUniform << ", " << weightDensityGradient <<
	//	", " << weightOpacity << ", " << weightOpacityGradient << std::endl;
	
	torch::Tensor out = torch::zeros({ N }, sampleLocations.options().dtype(c10::kBool));

	auto outputAcc = accessor<kernel::BTensor1RW>(out);
	const auto sampleAcc = accessor<kernel::Tensor2Read>(sampleLocations);
	const auto volumeAcc = accessor<kernel::Tensor4Read>(densityVolume);
	const auto tfAcc = accessor<kernel::Tensor3Read>(tf);
	kernel::sampleImportanceCUDA(
		outputAcc, sampleAcc, volumeAcc, tfAcc, tfMode,
		weightUniform, weightDensityGradient, weightOpacity, weightOpacityGradient,
		seed);
	return out;
}

END_RENDERER_NAMESPACE
