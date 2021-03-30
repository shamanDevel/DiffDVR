#include <catch.hpp>
#include <torch/types.h>

#include <renderer_tf.cuh>
#include <renderer_blending.cuh>
#include <pytorch_utils.h>

TEST_CASE("Non-Convex Density Reconstruction", "[reconstruction]")
{
	//normal-distributed gaussian as TF
	static const real_t red = 1, green = 0, blue = 0, alpha = 1, mean = 0, variance = 0.5;
	const torch::Tensor tf = torch::tensor(
		{{{red, green, blue, alpha, mean, variance}}},
		at::TensorOptions().dtype(real_dtype));
	const kernel::Tensor3Read tf_acc = accessor<kernel::Tensor3Read>(tf);
	const kernel::ITensor3Read tf_d_acc;

	static const real_t stepsize = 1/32.f;
	
	static const real_t d0 = -1;
	//d1 is variable

	typedef cudAD::fvar<real4, 1> v4;
	typedef cudAD::fvar<real_t, 1> v1;
	const auto eval = [&](real_t d1_)
	{
		v4 color = v4::constant(make_real4(0));
		v1 d1 = v1::input<0>(d1_);
		
		int num_steps = std::max(32, static_cast<int>(std::abs((d1_ - d0) * stepsize)));
		real_t local_stepsize = 1.0f / num_steps;
		kernel::TransferFunctionEval<kernel::TFGaussian> tf;
		for (int i = 0; i <= num_steps; ++i)
		{
			real_t f = i / real_t(num_steps);
			v1 d = (1 - f) * d0 + f * d1;
			v4 c = tf.evalForwardGradients<1>(tf_acc, 0, d, tf_d_acc,
				kernel::integral_constant<bool, false>());
			color = kernel::Blending<kernel::BlendBeerLambert>::blend(color, c, local_stepsize);
		}
		return color;
	};
	const real4 ground_truth = eval(d0).value();
	
	const auto loss = [ground_truth](const v4& color)
	{
		return length(ground_truth - color);
	};

	FILE* pFile;
	pFile = fopen("NonConvexDensity.tsv", "w");
	
	printf("Ground Truth: d1=%+.3f -> (red, alpha)=(%.3f, %.3f)\n",
		d0, ground_truth.x, ground_truth.w);
	fprintf(pFile, "Ground Truth: d1=%+.3f -> (red, alpha)=(%.3f, %.3f)\n",
		d0, ground_truth.x, ground_truth.w);
	fprintf(pFile, "d1\tred\talpha\tloss\tgrad\n");
	real_t min_d = -2;
	real_t max_d = +2;
	int steps = 4 * 32;
	real_t d_steps = (max_d - min_d) / steps;
	for (int i=0; i<=steps; ++i)
	{
		real_t d1 = min_d + i * d_steps;
		v4 c = eval(d1);
		v1 l = loss(c);
		printf("d1=%+.3f -> (red, alpha)=(%.3f, %.3f), loss=%.3f, grad=%+3f\n",
			d1, c.value().x, c.value().w, l.value(), l.derivative<0>());
		fprintf(pFile, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n",
			d1, c.value().x, c.value().w, l.value(), l.derivative<0>());
	}
	real_t d_limit = 1000;
	v4 c = eval(d_limit);
	v1 l = loss(c);
	printf("Limit (d1=1000) -> (red, alpha)=(%.3f, %.3f), loss=%.3f, grad=%+3f\n",
		d_limit, c.value().x, c.value().w, l.value(), l.derivative<0>());
	fprintf(pFile, "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n",
		d_limit, c.value().x, c.value().w, l.value(), l.derivative<0>());

	fclose(pFile);
}