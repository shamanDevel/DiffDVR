#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tinyformat.h>
#include <cuMat/src/Context.h>
#include <glm/gtx/string_cast.hpp>
#include <third-party/Eigen/Core> // in cuMat

#include <kernel_loader.h>
#include <renderer.h>
#include <renderer_cuda.h>
#include <volume.h>
#include <camera.h>
#include <tf_utils.h>
#include <losses.h>
#include <pytorch_utils.h>

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#endif

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME pyrenderer
#endif

namespace py = pybind11;
using namespace renderer;

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(kernels);

static void staticCudaSourcesLoaderRec(
	std::vector<renderer::KernelLoader::NameAndContent>& fileList,
	const cmrc::directory_entry& e, const cmrc::embedded_filesystem& fs,
	const std::string& currentPath)
{
	if (e.is_file())
	{
		std::cout << "Load file " << e.filename() << std::endl;
		auto f = fs.open(currentPath + e.filename());
		std::string content(f.size(), '\0');
		memcpy(content.data(), f.begin(), f.size());
		fileList.push_back({ e.filename(), content });
	} else
	{
		std::cout << "Walk directory " << currentPath << std::endl;
		for (const auto& e2 : fs.iterate_directory(currentPath + e.filename()))
			staticCudaSourcesLoaderRec(fileList, e2, fs, currentPath + e.filename() + "/");
	}
}
static void staticCudaSourcesLoader(
	std::vector<renderer::KernelLoader::NameAndContent>& fileList)
{
	cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();
	for (const auto& e : fs.iterate_directory(""))
		staticCudaSourcesLoaderRec(fileList, e, fs, "");
}

std::filesystem::path getCacheDir()
{
	//suffix and default (if default, it is a relative path)
	static const std::filesystem::path SUFFIX{ "kernel_cache" };
#ifdef WIN32
	//get the path to this dll as base path
	char path[MAX_PATH];
	HMODULE hm = NULL;

	if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		(LPCSTR)&getCacheDir, &hm) == 0)
	{
		int ret = GetLastError();
		fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
		return SUFFIX;
	}
	if (GetModuleFileName(hm, path, sizeof(path)) == 0)
	{
		int ret = GetLastError();
		fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
		return SUFFIX;
	}

	std::filesystem::path out = path;
	out = out.parent_path();
	const auto out_str = out.string();
	fprintf(stdout, "This DLL is located at %s, use that as cache dir\n", out_str.c_str());
	out /= SUFFIX;
	return out;
	
#else
	return SUFFIX; //default
#endif
}

class GPUTimer
{
	cudaEvent_t start_, stop_;
public:
	GPUTimer()
		: start_(0), stop_(0)
	{
		CUMAT_SAFE_CALL(cudaEventCreate(&start_));
		CUMAT_SAFE_CALL(cudaEventCreate(&stop_));
	}
	~GPUTimer()
	{
		CUMAT_SAFE_CALL(cudaEventDestroy(start_));
		CUMAT_SAFE_CALL(cudaEventDestroy(stop_));
	}
	void start()
	{
		CUMAT_SAFE_CALL(cudaEventRecord(start_));
	}
	void stop()
	{
		CUMAT_SAFE_CALL(cudaEventRecord(stop_));
	}
	float getElapsedMilliseconds()
	{
		CUMAT_SAFE_CALL(cudaEventSynchronize(stop_));
		float ms;
		CUMAT_SAFE_CALL(cudaEventElapsedTime(&ms, start_, stop_));
		return ms;
	}
};

//BINDINGS
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	// initialize context
	Renderer::initCuda();
	Renderer::setCudaCacheDir(getCacheDir());
	KernelLoader::Instance().setCustomCudaSourcesLoader(staticCudaSourcesLoader);
	cuMat::Context& ctx = cuMat::Context::current();
	((void)ctx);
	
	m.doc() = "python bindings for the differentiable volume renderer";
	
	m.def("use_double_precision", []()
	{
		return USE_DOUBLE_PRECISION;
	}, py::doc("Checks if the library was compiled with double-precision (returns 1) or single-precision (returns 0)."));
	m.def("reload_kernels", &Renderer::reloadCudaKernels);
	m.def("set_cuda_cache_dir", [](const std::string& path)
	{
		Renderer::setCudaCacheDir(path);
	});
	m.def("disable_cuda_cache", &Renderer::disableCudaCache);
	m.def("set_cuda_sync_mode", &Renderer::setCudaSyncMode);
	m.def("set_cuda_debug_mode", &Renderer::setCudaDebugMode);
	m.def("set_kernel_cache_file", [](const std::string& filename)
		{
			KernelLoader::Instance().setKernelCacheFile(filename);
		}, py::doc("Explicitly sets the kernel cache file."));
	
	m.def("cleanup", &Renderer::cleanupCuda,
		py::doc("Explicit cleanup of all CUDA references"));
	auto cleanup_callback = []() {
		Renderer::cleanupCuda();
	};
	m.add_object("_cleanup", py::capsule(cleanup_callback));
	

	py::class_<real3>(m, "real3")
		.def(py::init<>())
		.def(py::init([](real_t x, real_t y, real_t z) {return make_real3(x, y, z); }))
		.def_readwrite("x", &real3::x)
		.def_readwrite("y", &real3::y)
		.def_readwrite("z", &real3::z)
		.def("__str__", [](const real3& v)
	{
		return tinyformat::format("(%f, %f, %f)", v.x, v.y, v.z);
	});
	
	py::class_<real4>(m, "real4")
		.def(py::init<>())
		.def(py::init([](real_t x, real_t y, real_t z, real_t w)
	{
		return make_real4(x, y, z, w);
	}))
		.def_readwrite("x", &real4::x)
		.def_readwrite("y", &real4::y)
		.def_readwrite("z", &real4::z)
		.def_readwrite("w", &real4::w)
		.def("__str__", [](const real4& v)
	{
		return tinyformat::format("(%f, %f, %f, %f)", v.x, v.y, v.z, v.w);
	});

	py::class_<int2>(m, "int2")
		.def(py::init<>())
		.def(py::init([](int x, int y) {return make_int2(x, y); }))
		.def_readwrite("x", &int2::x)
		.def_readwrite("y", &int2::y)
		.def("__str__", [](const int2& v)
	{
		return tinyformat::format("(%d, %d)", v.x, v.y);
	});
	
	py::class_<int3>(m, "int3")
		.def(py::init<>())
		.def(py::init([](int x, int y, int z) {return make_int3(x, y, z); }))
		.def_readwrite("x", &int3::x)
		.def_readwrite("y", &int3::y)
		.def_readwrite("z", &int3::z)
		.def("__str__", [](const int3& v)
	{
		return tinyformat::format("(%d, %d, %d)", v.x, v.y, v.z);
	});
	
	py::enum_<kernel::VolumeFilterMode>(m, "VolumeFilterMode")
		.value("Nearest", kernel::VolumeFilterMode::FilterNearest)
		.value("Trilinear", kernel::VolumeFilterMode::FilterTrilinear)
		.value("Preshaded", kernel::VolumeFilterMode::FilterPreshaded);
	py::enum_<kernel::CameraMode>(m, "CameraMode")
		.value("RayStartDir", kernel::CameraMode::CameraRayStartDir)
		.value("InverseViewMatrix", kernel::CameraMode::CameraInverseViewMatrix)
		.value("ReferenceFrame", kernel::CameraMode::CameraReferenceFrame);
	py::enum_<kernel::TFMode>(m, "TFMode")
		.value("Identity", kernel::TFMode::TFIdentity)
		.value("Texture", kernel::TFMode::TFTexture)
		.value("Linear", kernel::TFMode::TFLinear)
		.value("Gaussian", kernel::TFMode::TFGaussian)
		.value("GaussianLog", kernel::TFMode::TFGaussianLog)
		.value("Preshaded", kernel::TFMode::TFPreshaded);
	py::enum_<kernel::BlendMode>(m, "BlendMode")
		.value("BeerLambert", kernel::BlendMode::BlendBeerLambert)
		.value("Alpha", kernel::BlendMode::BlendAlpha);
	
	py::enum_<Volume::ImplicitEquation>(m, "ImplicitEquation")
		.value("MarschnerLobb", Volume::ImplicitEquation::MARSCHNER_LOBB)
		.value("Cube", Volume::ImplicitEquation::CUBE)
		.value("Sphere", Volume::ImplicitEquation::SPHERE)
		.value("InverseSphere", Volume::ImplicitEquation::INVERSE_SPHERE)
		.value("DingDong", Volume::ImplicitEquation::DING_DONG)
		.value("Endrass", Volume::ImplicitEquation::ENDRASS)
		.value("Barth", Volume::ImplicitEquation::BARTH)
		.value("Heart", Volume::ImplicitEquation::HEART)
		.value("Kleine", Volume::ImplicitEquation::KLEINE)
		.value("Cassini", Volume::ImplicitEquation::CASSINI)
		.value("Steiner", Volume::ImplicitEquation::STEINER)
		.value("CrossCap", Volume::ImplicitEquation::CROSS_CAP)
		.value("Kummer", Volume::ImplicitEquation::KUMMER)
		.value("Blobbly", Volume::ImplicitEquation::BLOBBY)
		.value("Tube", Volume::ImplicitEquation::TUBE);
	py::class_<Volume, std::shared_ptr<Volume>>(m, "Volume")
		.def(py::init<std::string>())
		.def("create_mipmap_level", &Volume::createMipmapLevel)
		.def_property_readonly("world_size", [](std::shared_ptr<Volume> v) {
			return make_real3(v->worldSizeX(), v->worldSizeY(), v->worldSizeZ());
		})
		.def("set_world_size", [](std::shared_ptr<Volume> v, float x, float y, float z)
		{
				v->setWorldSizeX(x);
				v->setWorldSizeY(y);
				v->setWorldSizeZ(z);
		})
		.def_property_readonly("resolution", &Volume::baseResolution)
		.def_static("create_implicit", [](Volume::ImplicitEquation name, int resolution, py::kwargs kwargs) {
			std::unordered_map<std::string, float> args;
			for (const auto& e : kwargs)
				args.insert({ e.first.cast<std::string>(), e.second.cast<float>() });
			return std::shared_ptr<Volume>(Volume::createImplicitDataset(resolution, name, args));
		})
		.def_static("from_numpy", [](py::buffer b)
			{
				const py::buffer_info info = b.request();

				//sanity checks
				if (info.format != py::format_descriptor<real_t>::format())
					throw std::runtime_error("Incompatible format: expected a real array!");

				if (info.ndim != 3)
					throw std::runtime_error("Incompatible buffer dimension, expected 3!");

				ssize_t sizes[] = { info.shape[0], info.shape[1], info.shape[2] };
				ssize_t strides[] = {
					info.strides[0] / sizeof(real_t),
					info.strides[1] / sizeof(real_t),
					info.strides[2] / sizeof(real_t) };
				return std::shared_ptr<Volume>(Volume::createFromBuffer(
					static_cast<const real_t*>(info.ptr),
					sizes, strides
				));
			}, py::doc("Create a volume from a 3D numpy array of type 'real'"))
		.def("copy_to_gpu", [](std::shared_ptr<Volume> v)
		{
			for (int i = 0; v->getLevel(i); ++i)
				v->getLevel(i)->copyCpuToGpu();
		})
		.def("save", [](std::shared_ptr<Volume> v, const std::string& filename)
		{
			v->save(filename);
		})
		.def("getDataCpu", &Volume::dataCpu, 
			py::doc("Returns the density tensor of shape 1*X*Y*Z for the specified layer with the data located on the CPU"),
			py::arg("level"))
		.def("getDataGpu", &Volume::dataGpu, 
			py::doc("Returns the density tensor of shape 1*X*Y*Z for the specified layer with the data located on the GPU"),
			py::arg("level"));
		
	m.def("sync", []() {CUMAT_SAFE_CALL(cudaDeviceSynchronize()); });


	py::class_<RendererInputsHost::CameraPerPixelRays>(m, "CameraPerPixelRays")
		.def(py::init<>())
		.def(py::init<>([](const torch::Tensor& s, const torch::Tensor& d) {
			return RendererInputsHost::CameraPerPixelRays{ s, d };
		}))
		.def_readwrite("ray_start", &RendererInputsHost::CameraPerPixelRays::cameraRayStart,
			py::doc("Tensor of ray starting positions of shape B*H*W*3"))
		.def_readwrite("ray_dir", &RendererInputsHost::CameraPerPixelRays::cameraRayDir,
			py::doc("Tensor of ray direction of shape B*H*W*3"));
		py::class_<RendererInputsHost::CameraReferenceFrame>(m, "CameraReferenceFrame")
			.def(py::init<>())
			.def(py::init<>([](const torch::Tensor& s, real_t fovY) {
			return RendererInputsHost::CameraReferenceFrame{ s, fovY };
		}))
			.def_readwrite("viewport", &RendererInputsHost::CameraReferenceFrame::viewport,
				py::doc("Tensor of the viewport matrix of shape B*3*3\n"
				"where viewport[:,0,:] is the camera/eye position, viewport[:,1,:] the right vector "
				"and viewport[:,2,:] the up vector (normalized)."))
			.def_readwrite("fov_y_radians", &RendererInputsHost::CameraReferenceFrame::fovYRadians,
				py::doc("field-of-view for the y-axis in radians"));

	py::class_<glm::mat4>(m, "mat4", py::buffer_protocol())
		.def(py::init([](py::buffer b)
		{
		/* Request a buffer descriptor from Python */
		py::buffer_info info = b.request();

		/* Some sanity checks ... */
		if (info.format != py::format_descriptor<float>::format())
			throw std::runtime_error("Incompatible format: expected a float array!");

		if (info.ndim != 2)
			throw std::runtime_error("Incompatible buffer dimension, expected 2D!");
		if (info.shape[0] != 4 || info.shape[1] != 4)
			throw std::runtime_error("Incompatible buffer sizes! Expected a shape of 4x4");

		glm::mat4 m;
		for (int i=0; i<4; ++i) for (int j=0; j<4; ++j)
		{
			int idx = (info.strides[0] * i + info.strides[1] * j) / sizeof(float);
			m[i][j] = static_cast<float*>(info.ptr)[idx];
		}
		return m;
		}))
		.def("__str__", [](const glm::mat4& m)
		{
			return tinyformat::format("%s", glm::to_string(m));
		})
		.def_buffer([](glm::mat4& m) -> py::buffer_info
		{
			return py::buffer_info(
				&m[0].x,
				sizeof(float),
				py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
				2,
				{ 4, 4 },
				{ sizeof(float) * 4, sizeof(float) }
			);
		})
		.def("to_torch", [](const glm::mat4& m)
		{
			const auto cpuTensorOptions = at::TensorOptions().device(c10::kCPU).dtype(real_dtype);
#if USE_DOUBLE_PRECISION==1
			glm::mat<4, 4, real_t> m4_2 = m;
#else
			glm::mat4 m4_2 = m;
#endif
			torch::Tensor t = torch::from_blob(static_cast<void*>(
				&m4_2[0].x), { 1,4,4 }, cpuTensorOptions).clone();
			return t;
		});
	
	
		py::class_<RendererInputsHost>(m, "RendererInputs")
			.def(py::init<>())
			.def("clone",
				[](const RendererInputsHost& args)->RendererInputsHost {return args; },
				py::doc("performs a shallow clone"))
			.def_readwrite("screen_size", &RendererInputsHost::screenSize,
				py::doc("the screen size (width, height) of type int2"))
			.def_readwrite("volume", &RendererInputsHost::volume,
				py::doc("the volume tensor of shape B*X*Y*Z"))
			.def_readwrite("volume_filter_mode", &RendererInputsHost::volumeFilterMode,
				py::doc("the filter mode used to sample the density volume"))
			.def_readwrite("box_min", &RendererInputsHost::boxMin,
				py::doc("The minimal position of the bounding box.\nSpecified either as single float3 or float tensor of shape B*3."))
			.def_readwrite("box_size", &RendererInputsHost::boxSize,
				py::doc("The size of the bounding box.\nSpecified either as single float3 or float tensor of shape B*3."))
			.def_readwrite("camera_mode", &RendererInputsHost::cameraMode,
				py::doc("Specifies how the camera is defined, i.e. how self.camera is interpreted"))
			.def_readwrite("camera", &RendererInputsHost::camera,
				py::doc("The camera\n"
					" - if self.camera_mode==RayStartDir, an instance of CameraPerPixelRays\n"
					" - if self.camera_mode==ViewMatrix, a tensor of shape B*4*4 or an instance of mat4"))
			.def_readwrite("step_size", &RendererInputsHost::stepSize,
				py::doc("The step size of the renderer.\nSpecified either as single float or float tensor of shape B*H*W."))
			.def_readwrite("tf_mode", &RendererInputsHost::tfMode,
				py::doc("The type of transfer function to use. Specifies, how self.tf is interpreted"))
			.def_readwrite("tf", &RendererInputsHost::tf,
				py::doc("The transfer function, a float tensor of shape B*R*C:\n"
					" - if self.tf_mode==Identity, R=1, C=2 (opacity scaling, color scaling)\n"
					" - if self.tf_mode==Texture, C=4 (r,g,b,absorption) and R is the resolution of the texture\n"
					" - if self.tf_mode==Linear, C=5 (r,g,b,absorption,position) and R is the number of control points of the piecewise linear function."))
			.def_readwrite("blend_mode", &RendererInputsHost::blendMode,
				py::doc("The blend mode to use"))
			.def_readwrite("blending_early_out", &RendererInputsHost::blendingEarlyOut,
				py::doc("Opacity threshold after which the color is deemed opaque and the renderer terminates.\nDefault = 1 - 1e-5"));
	
	py::class_<RendererOutputsHost>(m, "RendererOutputs")
		.def(py::init<>())
		.def(py::init<torch::Tensor, torch::Tensor>())
		.def_readwrite("color", &RendererOutputsHost::color)
		.def_readwrite("terminationIndex", &RendererOutputsHost::terminationIndex);
	
	py::class_<ForwardDifferencesSettingsHost>(m, "ForwardDifferencesSettings")
		.def(py::init<>())
		.def_readwrite("D", &ForwardDifferencesSettingsHost::D,
			py::doc("The number of derivatives to trace"))
		.def_readwrite("d_stepsize", &ForwardDifferencesSettingsHost::d_stepsize,
			py::doc("index into the derivative array for the step size"))
		.def_readwrite("d_rayStart", &ForwardDifferencesSettingsHost::d_rayStart,
			py::doc("derivative indices for the start position of the ray"))
		.def_readwrite("d_rayDir", &ForwardDifferencesSettingsHost::d_rayDir,
			py::doc("derivative indices for the ray direction"))
		.def_readwrite("d_tf", &ForwardDifferencesSettingsHost::d_tf,
			py::doc("derivative index for the TF parameters, shape B*R*C, dtype=int32"))
		.def_readwrite("has_tf_derivatives", &ForwardDifferencesSettingsHost::hasTfDerivatives,
			py::doc("does the d_tf tensor holds at least one entry >= 0?"));

	py::class_<AdjointOutputsHost>(m, "AdjointOutputs")
		.def(py::init<>())
		.def_readwrite("has_volume_derivatives", &AdjointOutputsHost::hasVolumeDerivatives)
		.def_readwrite("adj_volume", &AdjointOutputsHost::adj_volume)
		.def_readwrite("has_stepsize_derivatives", &AdjointOutputsHost::hasStepSizeDerivatives)
		.def_readwrite("adj_stepsize", &AdjointOutputsHost::adj_stepSize)
		.def_readwrite("has_camera_derivatives", &AdjointOutputsHost::hasCameraDerivatives)
		.def_readwrite("adj_camera_ray_start", &AdjointOutputsHost::adj_cameraRayStart)
		.def_readwrite("adj_camera_ray_dir", &AdjointOutputsHost::adj_cameraRayDir)
		.def_readwrite("has_tf_derivatives", &AdjointOutputsHost::hasTFDerivatives)
		.def_readwrite("tf_delayed_accumulation", &AdjointOutputsHost::tfDelayedAcummulation)
		.def_readwrite("adj_tf", &AdjointOutputsHost::adj_tf);
	
	py::class_<Renderer>(m, "Renderer")
		.def_static("render_forward", &Renderer::renderForward,
			py::doc("Performs a simple forward rendering without gradients."),
			py::arg("inputs"), py::arg("outputs"))
		.def_static("render_forward_gradients", &Renderer::renderForwardGradients,
			py::doc(R"doc(
		Performs forward rendering with forward derivative computations.
		
		The input settings are specified in 'inputsHost' and the regular rendering
		output is written to 'outputsHost'. This is exactly equal to the
		result from \ref renderForward().
		
		Now this function adds functionality to compute forward derivatives
		with respect to a subset of the input settings.
		Let 'D' be the number of scalar parameters for which derivatives are traced.
		The mapping which parameter is traced with which index is done in
		'differencesSettingsHost'.
		The gradient output tensor 'gradientsOut' of shape B*H*W*D*C
		contains the derivatives of each of the 'D' parameters (in the 4th dimension)
		for each pixel x,y (dimension 3,2, of size W,H) and color channel
		C=4 (red, green, blue, alpha).
		
		To include this in the backward pass of an adjoint optimization framework,
		store or recompute this gradient tensor 'gradientsOut'.
		The backward pass provides as input the adjoint variables of the color
		output 'adjointColor', e.g. a tensor of shape B*H*W*C (C=4).
		Then the adjoint variable for the i-th parameter (0<=i<D) is computed
		by dot(adjointColor, gradientsOut[:,:,:,i,:]).
			)doc"), py::arg("inputs"), py::arg("differences_settings"),
			py::arg("outputs"), py::arg("gradients_out"))
		.def_static("render_adjoint", &Renderer::renderAdjoint,
			py::doc(R"doc(
		Performs adjoint/backward rendering.

		It starts with the outputs from the forward pass and the
		Adjoint variable for the output color as input.
		Then the adjoint variables are traced backward through the tracing
		and accumulated in adj_outputs.
		
		:param inputsHost the input settings
		:param outputsFromForwardHost the outputs from \ref renderForward
		:param adj_color the adjoint of the output color (B*H*W*4)
		:param adj_outputs [Out] the adjoint variables of the input settings.
			)doc"), py::arg("inputs"), py::arg("outputs_from_forward"),
			py::arg("adj_color"), py::arg("adj_outputs"))
		.def_static("forward_variables_to_gradients", &Renderer::forwardVariablesToGradients,
			py::doc(R"doc(
		Converts the forward variables from \ref renderForwardGradients()
		to adjoint variables for the camera (rayStart, rayDir), stepsize and
		transfer function, with respect to the adjoint variable of the
		color output 'gradientOutputColor'.
		This allows the forward differentiation to be embedded in an
		adjoint optimization framework.
		
		:param forward_variables the forward variables of shape B*H*W*D*C
			from Renderer::render_forward_gradients
		:param adj_color the adjoint variable of the output color
		:param differences_settings the mapping from derivative index to variable
		:param adj_outputs the adjoint variables for the individual parts.
			)doc"), py::arg("forward_variables"), py::arg("adj_color"),
			py::arg("differences_settings"), py::arg("adj_outputs"));


	py::enum_<Camera::Orientation>(m, "Orientation")
		.value("Xp", Camera::Xp)
		.value("Xm", Camera::Xm)
		.value("Yp", Camera::Yp)
		.value("Ym", Camera::Ym)
		.value("Zp", Camera::Zp)
		.value("Zm", Camera::Zm);

	py::class_<Camera>(m, "Camera")
		.def_static("compute_matrix", &Camera::computeInverseViewProjectionMatrix,
			py::doc("Computes the inverse view-projection matrix. Not differentiable!"),
			py::arg("origin"), py::arg("look_at"), py::arg("up"),
			py::arg("fov_degrees"), py::arg("width"), py::arg("height"),
			py::arg("near_clip")=0.1f, py::arg("far_clip")=10.0f)
		.def_static("generate_rays", &Camera::generateRays,
			py::doc(R"doc(
		Generates per-pixel rays from the given camera matrix.
		This function is differentiable / implements the Autograd protocol.

		:param viewport the viewport matrix of shape B*3*3
		  where viewport[:,0,:] is the camera/eye position,
		  viewport[:,1,:] the right vector and
		  viewport[:,2,:] the up vector.
		:param fovY the field of view for the y axis, in radians.
		:param screenWidth the screen width
		:param screenHeight the screen height
		:returns a tuple with the entries:
		 - ray_start: the start positions of the rays of shape B*H*W*3
		 - ray_dir: the directions of the rays of shape B*H*W*3
		 The returned tuple can be directly cast to RendererInputsHost::CameraPerPixelRays
)doc"), py::arg("viewport"), py::arg("fovY"), py::arg("screen_width"), py::arg("screen_height"))
		.def_static("viewport_from_lookat", &Camera::viewportFromLookAt,
			py::doc("Constructs the viewport matrix for generate_rays() from the camera origin look-at target and up vector."),
			py::arg("origin"), py::arg("look_at"), py::arg("camera_up"))
		.def_static("viewport_from_sphere", &Camera::viewportFromSphere,
			py::doc("Constructs the viewport matrix for generate_rays() from a postion on the sphere around a center."),
			py::arg("center"), py::arg("yaw_radians"), py::arg("pitch_radians"),
			py::arg("distance"), py::arg("orientation"));

	
	py::class_<GPUTimer>(m, "GpuTimer")
		.def(py::init<>())
		.def("start", &GPUTimer::start)
		.def("stop", &GPUTimer::stop)
		.def("elapsed_ms", &GPUTimer::getElapsedMilliseconds);

	py::class_<TFPoint>(m, "TFPoint")
		.def(py::init<>())
		.def(py::init<real_t, real4>())
		.def_readwrite("pos", &TFPoint::pos)
		.def_readwrite("val", &TFPoint::val);

	py::class_<TFUtils>(m, "TFUtils")
		.def_static("assemble_from_settings", &TFUtils::assembleFromSettings)
		.def_static("get_piecewise_tensor", &TFUtils::getPiecewiseTensor)
		.def_static("get_texture_tensor", &TFUtils::getTextureTensor)
		.def_static("preshade_volume", &TFUtils::preshadeVolume,
			py::doc(R"doc(
		Pre-shades a density volume of shape 1*X*Y*Z using
		the specified transfer function into a color volume of shape
		4*X*Y*Z.
		This function is differentiable.

		:param density the density volume of shape 1*X*Y*Z
		:param tf the transfer function tensor
		:param tfMode the type of transfer function
		:returns  the preshaded color volume of shape 4*X*Y*Z, channels are r,g,b,opacity
)doc"), py::arg("density"), py::arg("tf"), py::arg("tfMode"))
		.def_static("find_best_fit", 
			[](const torch::Tensor& colorVolume,
				const torch::Tensor& tf, kernel::TFMode tfMode,
				int numSamples, real_t opacityWeighting) {
					return TFUtils::findBestFit(colorVolume, tf, tfMode, numSamples, opacityWeighting);
			},
			py::doc(R"doc(
		Finds the best matching density value for the given 
		pre-shaded color volume and transfer function.
		The function samples 'num_samples' density values randomly,
		applies the TF, and compares the resulting color to the given 'color_volume'.
		The best match wins and that density is returned.

		The distance of the sampled (red,green,blue,opacity) is compared via:
		$ cost = ||rgb_samples - rgb_volume||_2^2 + opacity_weighting * (opacity_samples - opacity_volume)^2 $.

		:param color_volume the color volume of shape 4*X*Y*Z
		:param tf the transfer function tensor
		:param tf_mode the type of transfer function
		:param num_samples the number of samples to take per voxel
		:param opacity_weighting the weight of the opacity in the color comparison
		:returns the density volume of shape 1*X*Y*Z with the best matching density
)doc"), py::arg("color_volume"), py::arg("tf"), py::arg("tf_mode"),
			py::arg("num_samples"), py::arg("opacity_weighting"))
		.def_static("find_best_fit", [](const torch::Tensor& colorVolume,
			const torch::Tensor& tf, kernel::TFMode tfMode,
			int numSamples, real_t opacityWeighting,
			const torch::Tensor& previousDensities, real_t neighborWeighting) {
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				const auto ret = TFUtils::findBestFit(colorVolume, tf, tfMode, numSamples, opacityWeighting, &previousDensities, neighborWeighting);
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				return ret;
			},
			py::doc(R"doc(
		Finds the best matching density value for the given 
		pre-shaded color volume and transfer function.
		The function samples 'num_samples' density values randomly,
		applies the TF, and compares the resulting color to the given 'color_volume'.
		The best match wins and that density is returned.

		The distance of the sampled (red,green,blue,opacity) is compared via:
		$ cost = ||rgb_samples - rgb_volume||_2^2 + opacity_weighting * (opacity_samples - opacity_volume)^2 $.

		:param color_volume the color volume of shape 4*X*Y*Z
		:param tf the transfer function tensor
		:param tf_mode the type of transfer function
		:param num_samples the number of samples to take per voxel
		:param opacity_weighting the weight of the opacity in the color comparison
		:returns the density volume of shape 1*X*Y*Z with the best matching density
)doc"), py::arg("color_volume"), py::arg("tf"), py::arg("tf_mode"),
			py::arg("num_samples"), py::arg("opacity_weighting"),
			py::arg("previous_densities"), py::arg("neighbor_weighting"));


	m.def("mul_log", &mul_log);
	m.def("log_mse", &logMSE,
		py::doc(R"doc(
Computes the log-MSE loss, that is, for inputs x and y it holds:

\f[
   (x-y)^2 = MSE(x,y) = \exp(logMSE(\log(x), \log(y)))
\f]

Note that this is only defined for positive x and y.
This function is differentiable.
)doc"));
	m.def("log_l1", &logL1,
		py::doc(R"doc(
Computes the log-L1 loss, that is, for inputs x and y it holds:

\f[
   (x-y)^2 = L1(x,y) = \exp(logL1(\log(x), \log(y)))
\f]

Note that this is only defined for positive x and y.
This function is differentiable.
)doc"));
	
	
	m.def("sample_importance", sampleImportance,
		py::doc(R"doc(
 brief Performs importance sampling on the volume.
 There are four sampling categories, weighted by the respective
 'weight*' parameters:
 - uniform: accept all samples
 - density gradient: rejection sampling based on the gradient norm of the density
 - opacity: rejection sampling based on the opacity after TF mapping
 - opacity gradient: rejection sampling based on the opacity gradient norm after TF mapping
 The weights will be normalized.
 
 Supported device: GPU
 
 :param sample_locations the locations of the samples, float tensor on the GPU of
 shape (3,N) where N is the number of samples and each sample is in [0,1]^3.
 :param densityVolume the density volume of shape 1*X*Y*Z
 :param tf the transfer function tensor
 :param tfMode the type of transfer function
 :param weightUniform the weight for uniform sampling
 :param weightDensityGradient the weight for density gradient norm sampling
 :param weightOpacity  the weight for opacity sampling
 :param weightOpacityGradient the weight for opacity gradient norm sampling
 :param seed the seed for the random number generator
 :returns a boolean tensor of shape (N,) with true iff that sample should be taken.
)doc"), py::arg("sample_locations"), py::arg("density_volume"),
		py::arg("tf"), py::arg("tf_mode"), py::arg("weight_uniform"),
		py::arg("weight_density_gradient"), py::arg("weight_opacity"),
		py::arg("weight_opacity_gradient"), py::arg("seed"));
	
}