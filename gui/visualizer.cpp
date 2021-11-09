#include "visualizer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>

#include <lib.h>
#include <cuMat/src/Errors.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
//#include <Windows.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "imgui/imgui.h"
#include "imgui/IconsFontAwesome5.h"
#include "imgui/imgui_extension.h"
#include "imgui/imgui_internal.h"

#include <json.hpp>
#include <lodepng.h>
#include <portable-file-dialogs.h>
#include <magic_enum.hpp>
#include <indicators/progress_bar.hpp>
#include "utils.h"

#if USE_DOUBLE_PRECISION==0
#define real_dtype c10::ScalarType::Float
#else
#define real_dtype c10::ScalarType::Double
#endif

namespace nlohmann {
	template <>
	struct adl_serializer<ImVec4> {
		static void to_json(json& j, const ImVec4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, ImVec4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
				v.w = j.at(3).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a ImVec4" << std::endl;
		}
	};
}

const char* Visualizer::VolumeInputNames[] = {
	"Synthetic", "External File"
};

const char* Visualizer::RedrawModeNames[] = {
	"None", "Post", "Renderer"
};

const char* Visualizer::TFModeNames[] = {
	"Id", "Tex", "Pw", "Gauss"
};

const char* Visualizer::VolumeFilterModeNames[] = {
	"Nearest", "Trilinear"
};

const char* Visualizer::BlendModeNames[] = {
	"Beer-Lambert", "Alpha"
};

Visualizer::Visualizer(GLFWwindow* window)
	: window_(window)
{
	//load cuda
	bool cudaAvailable = renderer::Renderer::initCuda();
	std::cout << "CUDA available: " << cudaAvailable << std::endl;
	if (!cudaAvailable) exit(-1);
	
	// Add .ini handle for ImGuiWindow type
	ImGuiSettingsHandler ini_handler;
	ini_handler.TypeName = "Visualizer";
	ini_handler.TypeHash = ImHashStr("Visualizer");
	static const auto replaceWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == ' ') cpy[i] = '%'; //'%' is not allowed in path names
		return cpy;
	};
	static const auto insertWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == '%') cpy[i] = ' '; //'%' is not allowed in path names
		return cpy;
	};
	auto settingsReadOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void*
	{
		return handler->UserData;
	};
	auto settingsReadLine = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		char path[MAX_PATH];
		int intValue = 0;
		memset(path, 0, sizeof(char)*MAX_PATH);
		std::cout << "reading \"" << line << "\"" << std::endl;
		if (sscanf(line, "VolumeDir=%s", path) == 1)
			vis->volumeDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "TfDir=%s", path) == 1)
			vis->tfDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsDir=%s", path) == 1)
			vis->settingsDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsToLoad=%d", &intValue) == 1)
			vis->settingsToLoad_ = intValue;
	};
	auto settingsWriteAll = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		buf->reserve(200);
		buf->appendf("[%s][Settings]\n", handler->TypeName);
		std::string volumeDir = replaceWhitespace(vis->volumeDirectory_);
		std::string tfDir = replaceWhitespace(vis->tfDirectory_);
		std::string settingsDirectory = replaceWhitespace(vis->settingsDirectory_);
		std::cout << "Write settings:" << std::endl;
		buf->appendf("VolumeDir=%s\n", volumeDir.c_str());
		buf->appendf("TfDir=%s\n", tfDir.c_str());
		buf->appendf("SettingsDir=%s\n", settingsDirectory.c_str());
		buf->appendf("SettingsToLoad=%d\n", vis->settingsToLoad_);
		buf->appendf("\n");
	};
	ini_handler.UserData = this;
	ini_handler.ReadOpenFn = settingsReadOpen;
	ini_handler.ReadLineFn = settingsReadLine;
	ini_handler.WriteAllFn = settingsWriteAll;
	GImGui->SettingsHandlers.push_back(ini_handler);

	//initialize test volume
	volumeInput_ = VolumeInput::SYNTHETIC;
	syntheticVolume_ = renderer::Volume::createImplicitDataset(
		1 << syntheticDatasetResolutionPower_,
		renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
	syntheticVolume_->getLevel(0)->copyCpuToGpu();
	volumeMipmapLevel_ = 0;
	volumeFilename_ = "Initial";
	volumeHistogram_ = syntheticVolume_->extractHistogram();
	minDensity_ = (minDensity_ < volumeHistogram_.maxDensity && minDensity_ > volumeHistogram_.minDensity) ? minDensity_ : volumeHistogram_.minDensity;
	maxDensity_ = (maxDensity_ < volumeHistogram_.maxDensity && maxDensity_ > volumeHistogram_.minDensity) ? maxDensity_ : volumeHistogram_.maxDensity;
	updateTFTensors(tfMode_);
}

Visualizer::~Visualizer()
{
	releaseResources();
}

void Visualizer::releaseResources()
{
	if (screenTextureCuda_)
	{
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(screenTextureCuda_));
		screenTextureCuda_ = nullptr;
	}
	if (screenTextureGL_)
	{
		glDeleteTextures(1, &screenTextureGL_);
		screenTextureGL_ = 0;
	}
	if (screenTextureCudaBuffer_)
	{
		CUMAT_SAFE_CALL(cudaFree(screenTextureCudaBuffer_));
		screenTextureCudaBuffer_ = nullptr;
	}
	if (postOutput_)
	{
		CUMAT_SAFE_CALL(cudaFree(postOutput_));
		postOutput_ = nullptr;
	}
}

void Visualizer::settingsSave()
{
	// save file dialog
	auto fileNameStr = pfd::save_file(
		"Save settings",
		settingsDirectory_,
		{ "Json file", "*.json" },
		true
	).result();
	if (fileNameStr.empty())
		return;

	auto fileNamePath = std::filesystem::path(fileNameStr);
	fileNamePath = fileNamePath.replace_extension(".json");
	std::cout << "Save settings to " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();

	// Build json
	nlohmann::json settings;
	settings["version"] = 2;
	//camera
	settings["camera"] = cameraGui_.toJson();
	//TF editor
	settings["tfEditor"] = {
		{"editorLinear", editorLinear_.toJson()},
		{"editorTexture", editorTexture_.toJson()},
		{"editorGaussian", editorGaussian_.toJson()},
		{"minDensity", minDensity_},
		{"maxDensity", maxDensity_},
		{"opacityScaling", opacityScaling_},
		{"showColorControlPoints", showColorControlPoints_},
		{"dvrUseShading", dvrUseShading_}
	};
	//render parameters
	settings["renderer"] = {
		{"tfMode", tfMode_},
		{"stepsize", stepSize_},
		{"filterMode", volumeFilterMode_},
	};
	//shading
	settings["shading"] = {
		{"materialColor", materialColor},
		{"ambientLight", ambientLightColor},
		{"diffuseLight", diffuseLightColor},
		{"specularLight", specularLightColor},
		{"specularExponent", specularExponent},
		{"aoStrength", aoStrength},
		{"lightDirection", lightDirectionScreen},
	};
	//dataset
	settings["dataset"] = {
		{"syntheticDataset", volumeInput_==VolumeInput::SYNTHETIC ? syntheticDatasetIndex_ : -1},
		{"syntheticDatasetResolutionPower", syntheticDatasetResolutionPower_},
		{"file", volumeFullFilename_},
		{"mipmap", volumeMipmapLevel_}
	};

	//save json to file
	std::ofstream o(fileNamePath);
	o << std::setw(4) << settings << std::endl;
	screenshotString_ = std::string("Settings saved to ") + fileNamePath.string();
	screenshotTimer_ = 2.0f;
}

namespace
{
	std::string getDir(const std::string& path)
	{
		if (path.empty())
			return path;
		std::filesystem::path p(path);
		if (std::filesystem::is_directory(p))
			return path;
		return p.parent_path().string();
	}
}

void Visualizer::settingsLoad()
{
	// load file dialog
	auto results = pfd::open_file(
        "Load settings",
        getDir(settingsDirectory_),
        { "Json file", "*.json" },
        false
    ).result();
	if (results.empty())
		return;

	auto fileNameStr = results[0];
	auto fileNamePath = std::filesystem::path(fileNameStr);
	std::cout << "Load settings from " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();
	const auto basePath = fileNamePath.parent_path();

	//load json
	std::ifstream i(fileNamePath);
	nlohmann::json settings;
	try
	{
		i >> settings;
	} catch (const nlohmann::json::exception& ex)
	{
		pfd::message("Unable to parse Json", std::string(ex.what()),
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}
	i.close();
	int version = settings.contains("version")
		? settings.at("version").get<int>()
		: 0;
	if (version != 2)
	{
		pfd::message("Illegal Json", "The loaded json does not contain settings in the correct format",
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}

	//Ask which part should be loaded
	static bool loadCamera, loadComputationMode, loadTFeditor, loadRenderer, loadShading, loadDataset;
	static bool popupOpened;
	loadCamera = settingsToLoad_ & CAMERA;
	loadComputationMode = settingsToLoad_ & COMPUTATION_MODE;
	loadTFeditor = settingsToLoad_ & TF_EDITOR;
	loadRenderer = settingsToLoad_ & RENDERER;
	loadShading = settingsToLoad_ & SHADING;
	loadDataset = settingsToLoad_ & DATASET;
	popupOpened = false;
	auto guiTask = [this, settings, basePath]()
	{
		if (!popupOpened)
		{
			ImGui::OpenPopup("What to load");
			popupOpened = true;
			std::cout << "Open popup" << std::endl;
		}
		if (ImGui::BeginPopupModal("What to load", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Checkbox("Camera##LoadSettings", &loadCamera);
			//ImGui::Checkbox("Computation Mode##LoadSettings", &loadComputationMode);
			ImGui::Checkbox("TF Editor##LoadSettings", &loadTFeditor);
			ImGui::Checkbox("Renderer##LoadSettings", &loadRenderer);
			ImGui::Checkbox("Shading##LoadSettings", &loadShading);
			ImGui::Checkbox("Dataset##LoadSettings", &loadDataset);
			if (ImGui::Button("Load##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Enter))
			{
				try
				{
					//apply new settings
					if (loadCamera)
					{
						cameraGui_.fromJson(settings.at("camera"));
					}
					if (loadTFeditor)
					{
						const auto& s = settings.at("tfEditor");
						if (s.contains("editorLinear"))
							editorLinear_.fromJson(s.at("editorLinear"));
						else if (s.contains("editor"))
							editorLinear_.fromJson(s.at("editor"));
						if (s.contains("editorTexture"))
							editorTexture_.fromJson(s.at("editorTexture"));
						if (s.contains("editorGaussian"))
							editorGaussian_.fromJson(s.at("editorGaussian"));
						minDensity_ = s.at("minDensity").get<float>();
						maxDensity_ = s.at("maxDensity").get<float>();
						opacityScaling_ = s.at("opacityScaling").get<float>();
						showColorControlPoints_ = s.at("showColorControlPoints").get<bool>();
						dvrUseShading_ = s.at("dvrUseShading").get<bool>();
					}
					if (loadRenderer)
					{
						const auto& s = settings.at("renderer");
						tfMode_ = s.value("tfMode", kernel::TFMode::TFLinear);
						stepSize_ = s.at("stepsize").get<double>();
						volumeFilterMode_ = s.at("filterMode").get<kernel::VolumeFilterMode>();
						updateTFTensors(tfMode_);
					}
					if (loadShading)
					{
						const auto& s = settings.at("shading");
						materialColor = s.at("materialColor").get<float3>();
						ambientLightColor = s.at("ambientLight").get<float3>();
						diffuseLightColor = s.at("diffuseLight").get<float3>();
						specularLightColor = s.at("specularLight").get<float3>();
						specularExponent = s.at("specularExponent").get<float>();
						aoStrength = s.at("aoStrength").get<float>();
						lightDirectionScreen = s.at("lightDirection").get<float3>();
					}
					if (loadDataset)
					{
						const auto& s = settings.at("dataset");
						syntheticDatasetIndex_ = s.value("syntheticDataset", -1);
						syntheticDatasetResolutionPower_ = s.value("syntheticDatasetResolutionPower", 6);
						volumeFullFilename_ = s.at("file").get<std::string>();
						volumeMipmapLevel_ = s.at("mipmap").get<int>();
						if (syntheticDatasetIndex_ == -1 )
						{
							//external file
							if (!s.at("file").get<std::string>().empty())
							{
								auto targetPath = std::filesystem::path(s.at("file").get<std::string>());
								auto absPath = targetPath.is_absolute()
									? targetPath
									: std::filesystem::absolute(basePath / targetPath);
								try {
									loadVolume(absPath.string(), nullptr);
									volumeDirectory_ = absPath.string();
									volumeInput_ = VolumeInput::FILE;
								}
								catch (const std::exception& ex)
								{
									std::cerr << "Unable to load dataset with path " << absPath << ": " << ex.what() << std::endl;
								}
							}
						}
						else {
							//synthetic dataset
							syntheticVolume_ = renderer::Volume::createImplicitDataset(
								1 << syntheticDatasetResolutionPower_,
								renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
							syntheticVolume_->getLevel(0)->copyCpuToGpu();
							volumeInput_ = VolumeInput::SYNTHETIC;
						}
						selectMipmapLevel(volumeMipmapLevel_);
						triggerRedraw(RedrawRenderer);
					}
					//save last selection
					settingsToLoad_ =
						(loadCamera ? CAMERA : 0) |
						(loadComputationMode ? COMPUTATION_MODE : 0) |
						(loadTFeditor ? TF_EDITOR : 0) |
						(loadRenderer ? RENDERER : 0) |
						(loadShading ? SHADING : 0) |
						(loadDataset ? DATASET : 0);
					ImGui::MarkIniSettingsDirty();
					ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
					std::cout << "Settings applied" << std::endl;
				} catch (const nlohmann::json::exception& ex)
				{
					std::cerr << "Error: id=" << ex.id << ", message: " << ex.what() << std::endl;
					pfd::message("Unable to apply settings",
						std::string(ex.what()),
						pfd::choice::ok, pfd::icon::error).result();
				}
				//close popup
				this->backgroundGui_ = {};
				ImGui::CloseCurrentPopup();
				triggerRedraw(RedrawRenderer);
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel##LoadSettings") || ImGui::IsKeyPressedMap(ImGuiKey_Escape))
			{
				//close popup
				this->backgroundGui_ = {};
				ImGui::CloseCurrentPopup();
				triggerRedraw(RedrawRenderer);
			}
			ImGui::EndPopup();
		}
	};
	worker_.wait(); //wait for current task
	this->backgroundGui_ = guiTask;
}

void Visualizer::loadVolumeDialog()
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load volume",
		getDir(volumeDirectory_),
		{ "Volumes", "*.dat *.xyz *.cvol" },
		false
	).result();
	if (results.empty())
		return;
	std::string fileNameStr = results[0];

	std::cout << "Load " << fileNameStr << std::endl;
	auto fileNamePath = std::filesystem::path(fileNameStr);
	volumeDirectory_ = fileNamePath.string();
	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	//load the file
	worker_.wait(); //wait for current task
	std::shared_ptr<float> progress = std::make_shared<float>(0);
	auto guiTask = [progress]()
	{
		std::cout << "Progress " << *progress.get() << std::endl;
		if (ImGui::BeginPopupModal("Load Volume", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::ProgressBar(*progress.get(), ImVec2(200, 0));
			ImGui::EndPopup();
		}
	};
	this->backgroundGui_ = guiTask;
	ImGui::OpenPopup("Load Volume");
	auto loaderTask = [fileNameStr, progress, this](BackgroundWorker* worker)
	{
		loadVolume(fileNameStr, progress.get());

		//set it in the GUI and close popup
		this->backgroundGui_ = {};
		ImGui::CloseCurrentPopup();
		triggerRedraw(RedrawRenderer);
	};
	//start background task
	worker_.launch(loaderTask);
}

void Visualizer::loadVolume(const std::string& filename, float* progress)
{
	auto fileNamePath = std::filesystem::path(filename);
	//callbacks
	renderer::VolumeProgressCallback_t progressCallback = [progress](float v)
	{
		if (progress) *progress = v * 0.99f;
	};
	renderer::VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	renderer::VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
		throw std::exception(msg.c_str());
	};
	//load it locally
	try {
		std::unique_ptr<renderer::Volume> volume;
		if (fileNamePath.extension() == ".dat") {
			pfd::message(
				"Unsupported operation",
				"Loading RAW datasets is not supported at the moment.\nUse previous projects to convert them to .cvol",
				pfd::choice::ok);
			//volume.reset(renderer::loadVolumeFromRaw(filename, progressCallback, logging, error));
		}
		else if (fileNamePath.extension() == ".xyz") {
			pfd::message(
				"Unsupported operation",
				"Loading XYZ datasets is not supported at the moment.\nUse previous projects to convert them to .cvol",
				pfd::choice::ok);
			//volume.reset(renderer::loadVolumeFromXYZ(filename, progressCallback, logging, error));
		}
		else if (fileNamePath.extension() == ".cvol")
			volume = std::make_unique<renderer::Volume>(filename, progressCallback, logging, error);
		else {
			std::cerr << "Unrecognized extension: " << fileNamePath.extension() << std::endl;
		}
		if (volume != nullptr) {
			volume->getLevel(0)->copyCpuToGpu();
			std::swap(fileVolume_, volume);
			volumeMipmapLevel_ = 0;
			volumeFilename_ = fileNamePath.filename().string();
			volumeFullFilename_ = fileNamePath.string();
			std::cout << "Loaded" << std::endl;

			volumeHistogram_ = fileVolume_->extractHistogram();

			minDensity_ = (minDensity_ < volumeHistogram_.maxDensity&& minDensity_ > volumeHistogram_.minDensity) ? minDensity_ : volumeHistogram_.minDensity;
			maxDensity_ = (maxDensity_ < volumeHistogram_.maxDensity&& maxDensity_ > volumeHistogram_.minDensity) ? maxDensity_ : volumeHistogram_.maxDensity;
		}
	}
	catch (std::exception ex)
	{
		std::cerr << "Unable to load volume: " << ex.what() << std::endl;
	}
}

renderer::Volume* Visualizer::getCurrentVolume() const
{
	switch (volumeInput_)
	{
	case VolumeInput::SYNTHETIC: return syntheticVolume_.get();
	case VolumeInput::FILE: return fileVolume_.get();
	default: return nullptr;
	}
}

void Visualizer::selectMipmapLevel(int level)
{
	renderer::Volume* currentVolume = getCurrentVolume();
	if (currentVolume != nullptr) {
		currentVolume->createMipmapLevel(level);
		currentVolume->getLevel(level)->copyCpuToGpu();
		volumeMipmapLevel_ = level;
	}
}

static void HelpMarker(const char* desc)
{
	//ImGui::TextDisabled(ICON_FA_QUESTION);
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}
void Visualizer::specifyUI()
{
	uiMenuBar();

	ImGui::PushItemWidth(ImGui::GetFontSize() * -8);

	uiVolume();
	uiComputationMode();
	uiCamera();
	uiTfEditor();
	uiRenderer();
	uiForwardDifferences();
	uiShading();

	ImGui::PopItemWidth();

	if (backgroundGui_)
		backgroundGui_();

	uiScreenshotOverlay();
	uiFPSOverlay();
}

void Visualizer::uiMenuBar()
{
	ImGui::BeginMenuBar();
	ImGui::Text("Hotkeys");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted("'P': Screenshot");
		ImGui::TextUnformatted("'L': Lock foveated center");
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	if (ImGui::SmallButton("Save##Settings"))
		settingsSave();
	if (ImGui::SmallButton("Load##Settings"))
		settingsLoad();
	ImGui::EndMenuBar();
	//hotkeys
	if (ImGui::IsKeyPressed(GLFW_KEY_P, false))
	{
		screenshot();
	}
}

void Visualizer::uiVolume()
{
	if (ImGui::CollapsingHeader("Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::BeginCombo("Source", VolumeInputNames[int(volumeInput_)]))
		{
			for (int i = 0; i < int(VolumeInput::__COUNT__); ++i)
			{
				bool isSelected = int(volumeInput_) == i;
				if (ImGui::Selectable(VolumeInputNames[i], isSelected))
				{
					volumeInput_ = static_cast<VolumeInput>(i);
					triggerRedraw(RedrawRenderer);
				}
				if (isSelected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		renderer::Volume* currentVolume = nullptr;
		switch (volumeInput_)
		{
		case VolumeInput::SYNTHETIC:
		{
			std::vector<std::string> datasetIndexNames;
			for (const auto& n : magic_enum::enum_names<renderer::Volume::ImplicitEquation>())
			{
				datasetIndexNames.push_back(static_cast<std::string>(n));
			}
			datasetIndexNames.pop_back(); //counter
			if(ImGui::BeginCombo("Type", datasetIndexNames[syntheticDatasetIndex_].c_str()))
			{
				for (int i = 0; i < datasetIndexNames.size(); ++i)
				{
					bool isSelected = i == syntheticDatasetIndex_;
					if (ImGui::Selectable(datasetIndexNames[i].c_str(), isSelected))
					{
						syntheticDatasetIndex_ = i;
						syntheticVolume_ = renderer::Volume::createImplicitDataset(
							1 << syntheticDatasetResolutionPower_,
							renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
						syntheticVolume_->getLevel(0)->copyCpuToGpu();
						volumeMipmapLevel_ = 0;
						triggerRedraw(RedrawRenderer);
					}
					if (isSelected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			std::string resolutionStr = std::to_string(1 << syntheticDatasetResolutionPower_);
			if (ImGui::SliderInt("Resolution", &syntheticDatasetResolutionPower_, 4, 10, resolutionStr.c_str()))
			{
				syntheticVolume_ = renderer::Volume::createImplicitDataset(
					1 << syntheticDatasetResolutionPower_,
					renderer::Volume::ImplicitEquation(syntheticDatasetIndex_));
				syntheticVolume_->getLevel(0)->copyCpuToGpu();
				volumeMipmapLevel_ = 0;
				triggerRedraw(RedrawRenderer);
			}
			currentVolume = syntheticVolume_.get();
		} break;

		case VolumeInput::FILE:
		{
			ImGui::InputText("", &volumeFilename_[0], volumeFilename_.size() + 1, ImGuiInputTextFlags_ReadOnly);
			ImGui::SameLine();
			if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Volume"))
			{
				loadVolumeDialog();
			}
			currentVolume = fileVolume_.get();
		} break;
		}

		//statistics and mipmaps
		if (currentVolume != nullptr) {
			//functor for selecting the mipmap level, possible in a separate thread
			auto selectMipampLevel = [this, currentVolume](int level)
			{
				if (level == volumeMipmapLevel_) return;
				if (currentVolume->getLevel(level) == nullptr)
				{
					//resample in background thread
					worker_.wait(); //wait for current task
					auto guiTask = []()
					{
						if (ImGui::BeginPopupModal("Resample", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
						{
							const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
							ImGuiExt::Spinner("ResampleVolume", 50, 10, col);
							ImGui::EndPopup();
						}
					};
					this->backgroundGui_ = guiTask;
					ImGui::OpenPopup("Resample");
					auto resampleTask = [level, this](BackgroundWorker* worker)
					{
						selectMipmapLevel(level);
						//close popup
						this->backgroundGui_ = {};
						ImGui::CloseCurrentPopup();
						triggerRedraw(RedrawRenderer);
					};
					//start background task
					worker_.launch(resampleTask);
				}
				else
				{
					//just ensure, it is on the GPU
					currentVolume->getLevel(level)->copyCpuToGpu();
					volumeMipmapLevel_ = level;
					triggerRedraw(RedrawRenderer);
				}
			};
			//Level buttons
			for (int i = 0; i < sizeof(MipmapLevels) / sizeof(int); ++i)
			{
				int l = MipmapLevels[i];
				if (i > 0) ImGui::SameLine();
				std::string label = std::to_string(l + 1) + "x";
				if (ImGui::RadioButton(label.c_str(), volumeMipmapLevel_ == l))
					selectMipampLevel(l);
			}

			//print statistics
			ImGui::Text("Resolution: %d, %d, %d\nSize: %5.3f, %5.3f, %5.3f",
				currentVolume && currentVolume->getLevel(volumeMipmapLevel_) ? static_cast<int>(currentVolume->getLevel(volumeMipmapLevel_)->sizeX()) : 0,
				currentVolume && currentVolume->getLevel(volumeMipmapLevel_) ? static_cast<int>(currentVolume->getLevel(volumeMipmapLevel_)->sizeY()) : 0,
				currentVolume && currentVolume->getLevel(volumeMipmapLevel_) ? static_cast<int>(currentVolume->getLevel(volumeMipmapLevel_)->sizeZ()) : 0,
				currentVolume ? currentVolume->worldSizeX() : 0,
				currentVolume ? currentVolume->worldSizeY() : 0,
				currentVolume ? currentVolume->worldSizeZ() : 0);

			ImGui::Text("Min Density: %f\nMax Density: %f", volumeHistogram_.minDensity, volumeHistogram_.maxDensity);
		}
	}
}

void Visualizer::uiCamera()
{
	if (ImGui::CollapsingHeader("Camera")) {
		if (cameraGui_.specifyUI()) triggerRedraw(RedrawRenderer);
		renderer::Volume* currentVolume = getCurrentVolume();
		if (currentVolume) {
			if (ImGui::Button("Center"))
			{
				try {
					updateTFTensors(kernel::TFTexture);
					float3 centerOfMassVoxels = computeCenterOfMass(
						currentVolume, tfTensorTexture_.to(c10::kCPU));
					std::cout << "Center of mass in voxels: " << centerOfMassVoxels.x
						<< ", " << centerOfMassVoxels.y << ", " << centerOfMassVoxels.z << std::endl;
					float3 centerOfMassWorld = centerOfMassVoxels /
						make_float3(currentVolume->getLevel(0)->sizeX(), currentVolume->getLevel(0)->sizeY(), currentVolume->getLevel(0)->sizeZ());
					float3 differenceWorld = (centerOfMassWorld - make_float3(0.5f)) * currentVolume->worldSize();
					std::cout << "Difference: " << differenceWorld.x << ", " <<
						differenceWorld.y << ", " << differenceWorld.z << std::endl;
					cameraGui_.setLookAt(differenceWorld);
					triggerRedraw(RedrawRenderer);
				}
				catch (const std::exception& ex)
				{
					std::cerr << ex.what() << std::endl;
				}
			}
		}
	}
	if (cameraGui_.updateMouse())
		triggerRedraw(RedrawRenderer);
}

void Visualizer::uiRenderer()
{
	if (ImGui::CollapsingHeader("Render Parameters")) {
	
		float stepMin = 0.01f, stepMax = 1.0f;
		if (ImGui::SliderFloat("Stepsize", &stepSize_, stepMin, stepMax, "%.5f", 2)) triggerRedraw(RedrawRenderer);

		const char* currentFilterModeName = VolumeFilterModeNames[volumeFilterMode_];
		if (ImGui::SliderInt("Filter Mode", reinterpret_cast<int*>(&volumeFilterMode_),
			0, kernel::__VolumeFilterModeCount__ - 1, currentFilterModeName))
			triggerRedraw(RedrawRenderer);

		const char* currentBlendModeName = BlendModeNames[blendMode_];
		if (ImGui::SliderInt("Blend Mode", reinterpret_cast<int*>(&blendMode_),
			0, kernel::__BlendModeCount__ - 1, currentBlendModeName))
			triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::uiComputationMode()
{
	if (ImGui::RadioButton("CPU##runOnCUDA", !runOnCUDA_))
	{
		runOnCUDA_ = false;
		triggerRedraw(RedrawRenderer);
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("CUDA##runOnCUDA", runOnCUDA_))
	{
		runOnCUDA_ = true;
		triggerRedraw(RedrawRenderer);
	}
}

void Visualizer::updateTFTensors(kernel::TFMode tfMode)
{
	switch (tfMode)
	{
	case kernel::TFIdentity:
	{
		tfTensorIdentity_ = torch::tensor({ real_t(opacityScaling_), real_t(1.0f) }, 
			at::TensorOptions().dtype(real_dtype));
		tfTensorIdentity_ = tfTensorIdentity_.unsqueeze(0).unsqueeze(0);
	} break;
	case kernel::TFLinear:
	{
		tfTensorLinear_ = editorLinear_.getPiecewiseTensor(
			minDensity_, maxDensity_, opacityScaling_, true);
	} break;
	case kernel::TFTexture:
	{
		static constexpr int TEXTURE_RESOLUTION = 256;
		tfTensorTexture_ = editorTexture_.getTextureTensor(
			TEXTURE_RESOLUTION, minDensity_, maxDensity_, opacityScaling_, false);
	} break;
	case kernel::TFGaussian:
	{
		tfTensorGaussian_ = editorGaussian_.getGaussianTensor(
			minDensity_, maxDensity_, opacityScaling_);
	} break;
	}
	
	triggerRedraw(RedrawRenderer);
}

void Visualizer::uiTfEditor()
{
	if (ImGui::CollapsingHeader("TF Editor"))
	{
		for (int i = 0; i < kernel::__TFModeCount__; ++i) 
		{
			if (i > 0) ImGui::SameLine();
			if (ImGui::RadioButton(TFModeNames[i], static_cast<int>(tfMode_)==i))
			{
				tfMode_ = static_cast<kernel::TFMode>(i);
				updateTFTensors(tfMode_);
			}
		}
		
		if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load TF"))
		{
			// open file dialog
			auto results = pfd::open_file(
				"Load transfer function",
				tfDirectory_,
				{ "Transfer Function", "*.tf" },
				false
			).result();
			if (results.empty())
				return;
			std::string fileNameStr = results[0];

			auto fileNamePath = std::filesystem::path(fileNameStr);
			std::cout << "TF is loaded from " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			if (tfMode_ == kernel::TFLinear)
				editorLinear_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (tfMode_ == kernel::TFTexture)
				editorTexture_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (tfMode_ == kernel::TFGaussian)
				editorGaussian_.loadFromFile(fileNamePath.string(), minDensity_, maxDensity_);
			updateTFTensors(tfMode_);
		}
		ImGui::SameLine();
		if (ImGui::Button(ICON_FA_SAVE " Save TF"))
		{
			// save file dialog
			auto fileNameStr = pfd::save_file(
				"Save transfer function",
				tfDirectory_,
				{ "Transfer Function", "*.tf" },
				true
			).result();
			if (fileNameStr.empty())
				return;

			auto fileNamePath = std::filesystem::path(fileNameStr);
			fileNamePath = fileNamePath.replace_extension(".tf");
			std::cout << "TF is saved under " << fileNamePath << std::endl;
			tfDirectory_ = fileNamePath.string();

			if (tfMode_ == kernel::TFLinear)
				editorLinear_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (tfMode_ == kernel::TFTexture)
				editorTexture_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
			else if (tfMode_ == kernel::TFGaussian)
				editorGaussian_.saveToFile(fileNamePath.string(), minDensity_, maxDensity_);
		}
		ImGui::SameLine();
		ImGui::Checkbox("Show CPs", &showColorControlPoints_);

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;

		//prepare histogram
		auto histogramRes = (volumeHistogram_.maxDensity - volumeHistogram_.minDensity) / volumeHistogram_.getNumOfBins();
		int histogramBeginOffset = (minDensity_ - volumeHistogram_.minDensity) / histogramRes;
		int histogramEndOffset = (volumeHistogram_.maxDensity - maxDensity_) / histogramRes;
		auto maxFractionVal = *std::max_element(std::begin(volumeHistogram_.bins) + histogramBeginOffset, std::end(volumeHistogram_.bins) - histogramEndOffset);

		//draw editor
		if (tfMode_ == kernel::TFLinear || tfMode_ == kernel::TFTexture) {
			//Color
			const ImGuiID tfEditorColorId = window->GetID("TF Editor Color");
			auto pos = window->DC.CursorPos;
			auto tfEditorColorWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
			auto tfEditorColorHeight = 50.0f;
			const ImRect tfEditorColorRect(pos, ImVec2(pos.x + tfEditorColorWidth, pos.y + tfEditorColorHeight));
			ImGui::ItemSize(tfEditorColorRect, style.FramePadding.y);
			ImGui::ItemAdd(tfEditorColorRect, tfEditorColorId);

			//Opacity
			const ImGuiID tfEditorOpacityId = window->GetID("TF Editor Opacity");
			pos = window->DC.CursorPos;
			auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
			auto tfEditorOpacityHeight = 100.0f;
			const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

			ImGui::PlotHistogram("", volumeHistogram_.bins + histogramBeginOffset, volumeHistogram_.getNumOfBins() - histogramEndOffset - histogramBeginOffset,
				0, NULL, 0.0f, maxFractionVal, ImVec2(tfEditorOpacityWidth, tfEditorOpacityHeight));

			if (tfMode_ == kernel::TFLinear) {
				editorLinear_.init(tfEditorOpacityRect, tfEditorColorRect, showColorControlPoints_);
				editorLinear_.handleIO();
				editorLinear_.render();
				if (editorLinear_.getIsChanged())
				{
					updateTFTensors(tfMode_);
				}
			} else
			{
				editorTexture_.init(tfEditorOpacityRect, tfEditorColorRect, showColorControlPoints_);
				editorTexture_.handleIO();
				editorTexture_.render();
				if (editorTexture_.getIsChanged())
				{
					updateTFTensors(tfMode_);
				}
			}
		}
		else if (tfMode_ == kernel::TFGaussian)
		{
			auto pos = window->DC.CursorPos;
			auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
			auto tfEditorOpacityHeight = 100.0f;
			const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

			ImGui::PlotHistogram("TF Editor", volumeHistogram_.bins + histogramBeginOffset, volumeHistogram_.getNumOfBins() - histogramEndOffset - histogramBeginOffset,
				0, NULL, 0.0f, maxFractionVal, ImVec2(tfEditorOpacityWidth, tfEditorOpacityHeight));

			editorGaussian_.render(tfEditorOpacityRect);
			if (editorGaussian_.getIsChanged())
				updateTFTensors(tfMode_);
		}

		if (ImGui::SliderFloat("Opacity Scaling", &opacityScaling_, 1.0f, 500.0f, "%.3f", 2))
		{
			updateTFTensors(tfMode_);
		}

		if (tfMode_ != kernel::TFIdentity) {
			if (ImGui::SliderFloat("Min Density", &minDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
			{
				updateTFTensors(tfMode_);
			}
			if (ImGui::SliderFloat("Max Density", &maxDensity_, volumeHistogram_.minDensity, volumeHistogram_.maxDensity))
			{
				updateTFTensors(tfMode_);
			}
		}
			
		//not supported yet
		//if (ImGui::Checkbox("Use Shading", &dvrUseShading_))
		//{
		//	triggerRedraw(RedrawRenderer);
		//}
	}
}

void Visualizer::uiShading()
{
	if (ImGui::CollapsingHeader("Output - Shading", ImGuiTreeNodeFlags_DefaultOpen)) {
		auto redraw = RedrawRenderer;

		//ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel;
		//if (ImGui::ColorEdit3("Material Color", &materialColor.x, colorFlags)) triggerRedraw(redraw);
		//if (ImGui::ColorEdit3("Ambient Light", &ambientLightColor.x, colorFlags)) triggerRedraw(redraw);
		//if (ImGui::ColorEdit3("Diffuse Light", &diffuseLightColor.x, colorFlags)) triggerRedraw(redraw);
		//if (ImGui::ColorEdit3("Specular Light", &specularLightColor.x, colorFlags)) triggerRedraw(redraw);
		//float minSpecular = 0, maxSpecular = 64;
		//if (ImGui::SliderScalar("Spec. Exp.", ImGuiDataType_Float, &specularExponent, &minSpecular, &maxSpecular, "%.3f", 2)) triggerRedraw(redraw);
		//float minAO = 0, maxAO = 1;
		//if (ImGui::SliderScalar("AO Strength", ImGuiDataType_Float, &aoStrength, &minAO, &maxAO)) triggerRedraw(RedrawPost);
		//if (ImGuiExt::DirectionPicker2D("Light direction", &lightDirectionScreen.x, ImGuiExtDirectionPickerFlags_InvertXY))
		//	triggerRedraw(redraw);

		static const char* ClearColorNames[] = { "Black", "White" };
		int clearColorIndex = clearColor_.x > 0.5;
		if (ImGui::SliderInt("Background", &clearColorIndex, 0, 1, ClearColorNames[clearColorIndex]))
		{
			if (clearColorIndex)
				clearColor_ = ImVec4(1, 1, 1, 1);
			else
				clearColor_ = ImVec4(0, 0, 0, 1);
			triggerRedraw(RedrawPost);
		}
	}
}

void Visualizer::uiScreenshotOverlay()
{
	if (screenshotTimer_ <= 0) return;

	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y - 10);
	ImVec2 window_pos_pivot = ImVec2(0.5f, 1.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	//ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
	ImGui::Begin("Example: Simple overlay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::TextUnformatted(screenshotString_.c_str());
	ImGui::End();
	//ImGui::PopStyleVar(ImGuiStyleVar_Alpha);

	screenshotTimer_ -= io.DeltaTime;
}

void Visualizer::uiFPSOverlay()
{
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x - 5, 5);
	ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.5f);
	ImGui::Begin("FPSDisplay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::Text("FPS: %.1f", io.Framerate);
	std::string extraText = extraFrameInformation_.str();
	if (!extraText.empty())
	{
		extraText = extraText.substr(1); //strip initial '\n'
		ImGui::TextUnformatted(extraText.c_str());
	}
	extraFrameInformation_ = std::stringstream();
	ImGui::End();
}

void Visualizer::uiForwardDifferences()
{
	static int prevFdEnabled = -1;
	if (ImGui::CollapsingHeader("Forward Differences")) {
		fd_enabled = true;
		if (ImGui::RadioButton("Stepsize derivatives", fd_stepsize))
		{
			fd_stepsize = true;
			fd_camera = false; fd_tf = false;
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::RadioButton("Camera derivatives", fd_camera))
		{
			fd_camera = true;
			fd_stepsize = false; fd_tf = false;
			triggerRedraw(RedrawRenderer);
		}
		if (ImGui::RadioButton("TF derivatives", fd_tf))
		{
			fd_tf = true;
			fd_camera = false; fd_stepsize = false;
			triggerRedraw(RedrawRenderer);
		}
		if (fd_camera)
		{
			static const char* NAMES[] = {
				"Start-X", "Start-Y", "Start-Z", "Dir-X", "Dir-Y", "Dir-Z" };
			if (ImGui::SliderInt("Channel", &fd_camera_index, 0, 5,
				NAMES[fd_camera_index]))
			{
				triggerRedraw(RedrawRenderer);
			}
		}
		if (fd_tf)
		{
			if (tfMode_ == kernel::TFIdentity)
			{
				fd_tf_point = 0;
				fd_tf_channel = clamp(fd_tf_channel, 0, 1);
				static const char* NAMES[] = {
					"Opacity-Scaling", "Color-Scaling" };
				if (ImGui::SliderInt("Channel", &fd_tf_channel, 0, 1,
					NAMES[fd_tf_channel]))
				{
					triggerRedraw(RedrawRenderer);
				}
			}
			else if (tfMode_ == kernel::TFTexture)
			{
				int R = tfTensorTexture_.size(1);
				fd_tf_point = clamp(fd_tf_point, 0, R - 1);
				fd_tf_channel = clamp(fd_tf_channel, 0, 3);
				if (ImGui::SliderInt("Control point", &fd_tf_point, 0, R - 1))
				{
					triggerRedraw(RedrawRenderer);
				}
				static const char* NAMES[] = {
					"Red", "Green", "Blue", "Opacity"};
				if (ImGui::SliderInt("Channel", &fd_tf_channel, 0, 3,
					NAMES[fd_tf_channel]))
				{
					triggerRedraw(RedrawRenderer);
				}
			}
			else if (tfMode_ == kernel::TFLinear)
			{
				int R = tfTensorLinear_.size(1);
				fd_tf_point = clamp(fd_tf_point, 1, R-2);
				fd_tf_channel = clamp(fd_tf_channel, 0, 4);
				if (ImGui::SliderInt("Control points", &fd_tf_point, 1, R-2))
				{
					triggerRedraw(RedrawRenderer);
				}
				static const char* NAMES[] = {
					"Red", "Green", "Blue", "Opacity", "Position" };
				if (ImGui::SliderInt("Channel", &fd_tf_channel, 0, 4,
					NAMES[fd_tf_channel]))
				{
					triggerRedraw(RedrawRenderer);
				}
			}
			else
			{
				std::cerr << "Unknown tf mode for forward differences" << std::endl;
				fd_enabled = false;
			}
		}

		static const char* OUTPUT_NAMES[] = {
					"Red", "Green", "Blue", "Alpha", "Difference" };
		if (ImGui::SliderInt("Output Channel", &fd_outputChannel, 0, 4,
			OUTPUT_NAMES[fd_outputChannel]))
		{
			triggerRedraw(RedrawPost);
		}
		if (fd_outputChannel==4)
		{
			if (ImGui::Button("Clear Reference"))
			{
				forwardDifferencesReference = torch::Tensor();
				triggerRedraw(RedrawPost);
			}
			ImGui::SameLine();
			if (ImGui::Button("Record Reference"))
			{
				fd_recordReference = true;
				triggerRedraw(RedrawRenderer);
			}
			if (!forwardDifferencesReference.defined())
				ImGui::TextColored(ImVec4(1, 0, 0, 1), "No reference image available");
			
		}

		ImGui::Text("derivatives: min=%.4f, max=%.4f", fd_minGradient, fd_maxGradient);

		if (fd_outputChannel == 4)
		{
			ImGui::Text("L2-Loss: %.4f, gradient=%.4f", fd_avgLoss, fd_avgGradient);
		}
	} else
	{
		fd_enabled = false;
	}

	if (prevFdEnabled != int(fd_enabled))
		triggerRedraw(RedrawRenderer);
	prevFdEnabled = int(fd_enabled);
}

renderer::RendererInputsHost Visualizer::setupRendererArgs()
{
	renderer::RendererInputsHost inputs;
	inputs.screenSize = make_int2(displayWidth_, displayHeight_);
	
	inputs.volumeFilterMode = volumeFilterMode_;
	renderer::Volume* currentVolume = getCurrentVolume();
	real3 voxelSize;
	if (currentVolume) {
		const auto level = currentVolume->getLevel(volumeMipmapLevel_);
		inputs.volume = runOnCUDA_
			? level->dataGpu()
			: level->dataCpu();
		real3 boxSize = make_real3(currentVolume->worldSizeX(), currentVolume->worldSizeY(), currentVolume->worldSizeZ());
		voxelSize = boxSize / make_real3(level->sizeX() - 1, level->sizeY() - 1, level->sizeZ() - 1);
		real3 boxMin = (-boxSize * 0.5) - (voxelSize * 0.5);
		inputs.boxMin = boxMin;
		inputs.boxSize = boxSize;
	} else
	{
		inputs.volume = runOnCUDA_
			? torch::empty({1,1,1,1}, at::TensorOptions().dtype(real_dtype).device(c10::kCUDA))
			: torch::empty({ 1,1,1,1 }, at::TensorOptions().dtype(real_dtype).device(c10::kCPU));
		voxelSize = make_real3(1);
		inputs.boxMin = make_real3(-0.5f);
		inputs.boxSize = make_real3(1);
	}

	inputs.cameraMode = kernel::CameraInverseViewMatrix;
	glm::mat4 invViewMatrix = cameraGui_.getInverseViewProjectionMatrix(displayWidth_, displayHeight_);
	//invViewMatrix = glm::transpose(invViewMatrix);
	inputs.camera = invViewMatrix;

	inputs.stepSize = stepSize_ * voxelSize.x;

	inputs.tfMode = tfMode_;
	switch (tfMode_)
	{
	case kernel::TFIdentity:
		tfTensorIdentity_ = tfTensorIdentity_.to(inputs.volume.device(), real_dtype);
		inputs.tf = tfTensorIdentity_;
		break;
	case kernel::TFLinear:
		tfTensorLinear_ = tfTensorLinear_.to(inputs.volume.device(), real_dtype);
		inputs.tf = tfTensorLinear_;
		break;
	case kernel::TFTexture:
		tfTensorTexture_ = tfTensorTexture_.to(inputs.volume.device(), real_dtype);
		inputs.tf = tfTensorTexture_;
		break;
	case kernel::TFGaussian:
		tfTensorGaussian_ = tfTensorGaussian_.to(inputs.volume.device(), real_dtype);
		inputs.tf = tfTensorGaussian_;
		break;
	}
	
	inputs.blendMode = blendMode_;
	return inputs;
}

renderer::ForwardDifferencesSettingsHost Visualizer::setupForwardDifferencesArgs(
	const renderer::RendererInputsHost& inputs)
{
	renderer::ForwardDifferencesSettingsHost settings;
	settings.D = 1;
	settings.d_stepsize = fd_stepsize ? 0 : -1;
	if (fd_camera)
	{
		settings.d_rayStart = make_int3(
			fd_camera_index == 0 ? 0 : -1,
			fd_camera_index == 1 ? 0 : -1,
			fd_camera_index == 2 ? 0 : -1
		);
		settings.d_rayDir = make_int3(
			fd_camera_index == 3 ? 0 : -1,
			fd_camera_index == 4 ? 0 : -1,
			fd_camera_index == 5 ? 0 : -1
		);
	} else
	{
		settings.d_rayStart = make_int3(-1);
		settings.d_rayDir = make_int3(-1);
	}
	settings.hasTfDerivatives = fd_tf;
	settings.d_tf = torch::empty(inputs.tf.sizes(), 
		at::TensorOptions().dtype(c10::kInt).device(c10::kCPU));
	auto acc = settings.d_tf.accessor<int, 3>();
	for (int r = 0; r < settings.d_tf.size(1); ++r)
		for (int c = 0; c < settings.d_tf.size(2); ++c)
			acc[0][r][c] = -1;
	acc[0][fd_tf_point][fd_tf_channel] = 0;
	settings.d_tf = settings.d_tf.to(inputs.tf.device());
	return settings;
}

void Visualizer::render(int display_w, int display_h)
{
	resize(display_w, display_h);

	if (redrawMode_ == RedrawNone)
	{
		//just draw the precomputed texture
		drawer_.drawQuad(screenTextureGL_);
		return;
	}

	auto currentVolume = getCurrentVolume();
	if (currentVolume == nullptr) return;
	if (currentVolume->getLevel(volumeMipmapLevel_) == nullptr) return;
	try {
		renderImpl();
	}
	catch (torch::Error ex)
	{
		std::cerr << "Torch exception during rendering: " << ex.what() << std::endl;
		redrawMode_ = RedrawNone;
	}
	catch (std::exception ex)
	{
		std::cerr << "Unknown exception during rendering: " << ex.what() << std::endl;
		redrawMode_ = RedrawNone;
	}
}

void Visualizer::renderImpl()
{
	if (stepSize_<1e-4)
	{
		std::cout << "Step size is zero, don't render" << std::endl;
		return;
	}
	
	//render to rendererOutput_
	if (!fd_enabled) {
		if (redrawMode_ == RedrawRenderer)
		{
			renderer::RendererInputsHost inputs = setupRendererArgs();
			if (runOnCUDA_) {
				renderer::Renderer::renderForward(inputs, rendererOutputsGpu_);
			}
			else
			{
				renderer::Renderer::renderForward(inputs, rendererOutputsCpu_);
				rendererOutputsGpu_.color = rendererOutputsCpu_.color.to(c10::kCUDA);
				rendererOutputsGpu_.terminationIndex = rendererOutputsCpu_.terminationIndex.to(c10::kCUDA);
				//rendererOutputsGpu_.color.copy_(rendererOutputsCpu_.color, true);
				//rendererOutputsGpu_.terminationIndex.copy_(rendererOutputsCpu_.terminationIndex, true);
			}
			redrawMode_ = RedrawPost;
			CUMAT_SAFE_CALL(cudaDeviceSynchronize());
		}

		//select channel and write to screen texture
		if (redrawMode_ == RedrawPost)
		{
			const auto& inputTensor = rendererOutputsGpu_.color;
			kernel::copyOutputToTexture(
				kernel::Tensor4Read(inputTensor.data_ptr<real_t>(), inputTensor.sizes().data(), inputTensor.strides().data()),
				postOutput_);
			redrawMode_ = RedrawNone;
		}
	}
	else
	{
		//with forward differences
		if (redrawMode_ == RedrawRenderer)
		{
			renderer::RendererInputsHost inputs = setupRendererArgs();
			renderer::ForwardDifferencesSettingsHost settings = setupForwardDifferencesArgs(inputs);
			forwardDifferencesOutput = torch::zeros({ 1, displayHeight_, displayWidth_, 1, 4 }, inputs.volume.options());
			if (runOnCUDA_) {
				renderer::Renderer::renderForwardGradients(inputs, settings, rendererOutputsGpu_, forwardDifferencesOutput);
			}
			else
			{
				renderer::Renderer::renderForwardGradients(inputs, settings, rendererOutputsCpu_, forwardDifferencesOutput);
				rendererOutputsGpu_.color = rendererOutputsCpu_.color.to(c10::kCUDA);
				rendererOutputsGpu_.terminationIndex = rendererOutputsCpu_.terminationIndex.to(c10::kCUDA);
				forwardDifferencesOutput = forwardDifferencesOutput.to(c10::kCUDA);
			}
			if (fd_recordReference)
			{
				forwardDifferencesReference = rendererOutputsGpu_.color.clone();
				fd_recordReference = false;
			}
			redrawMode_ = RedrawPost;
		}
		
		//select channel and write to screen texture
		if (redrawMode_ == RedrawPost)
		{
			//select for which output channel the gradients should be displayed
			if (fd_outputChannel==4)
			{
				//difference to a baseline
				if (forwardDifferencesReference.defined())
				{
					//no reduction for visualization
					torch::Tensor differenceOut = torch::empty(
						{ 1, displayHeight_, displayWidth_, 1 }, forwardDifferencesReference.options());
					torch::Tensor gradientsOut = torch::empty(
						{ 1, displayHeight_, displayWidth_, 1 }, forwardDifferencesReference.options());
					if (runOnCUDA_) {
						renderer::Renderer::compareToImage(
							rendererOutputsGpu_.color,
							forwardDifferencesOutput,
							forwardDifferencesReference,
							differenceOut,
							gradientsOut,
							false);
					} else
					{
						differenceOut = differenceOut.to(c10::kCPU);
						gradientsOut = gradientsOut.to(c10::kCPU);
						renderer::Renderer::compareToImage(
							rendererOutputsGpu_.color.to(c10::kCPU),
							forwardDifferencesOutput.to(c10::kCPU),
							forwardDifferencesReference.to(c10::kCPU),
							differenceOut,
							gradientsOut,
							false);
						differenceOut = differenceOut.to(c10::kCUDA);
						gradientsOut = gradientsOut.to(c10::kCUDA);
					}
					fd_minGradient = gradientsOut.min().item<real_t>();
					fd_maxGradient = gradientsOut.max().item<real_t>();
					kernel::divergentColorMap(
						kernel::Tensor4Read(gradientsOut.data_ptr<real_t>(), gradientsOut.sizes().data(), gradientsOut.strides().data()),
						fd_minGradient, fd_maxGradient, postOutput_);

					//with reduction for measures
					differenceOut = torch::empty(
						{ 1 }, forwardDifferencesReference.options());
					gradientsOut = torch::empty(
						{ 1 }, forwardDifferencesReference.options());
					renderer::Renderer::compareToImage(
						rendererOutputsGpu_.color,
						forwardDifferencesOutput,
						forwardDifferencesReference,
						differenceOut,
						gradientsOut,
						true);
					fd_avgLoss = differenceOut.item<real_t>();
					fd_avgGradient = gradientsOut.item<real_t>();
				}
				else
				{
					//no baseline recorded, show color
					const auto& inputTensor = rendererOutputsGpu_.color;
					kernel::copyOutputToTexture(
						kernel::Tensor4Read(inputTensor.data_ptr<real_t>(), inputTensor.sizes().data(), inputTensor.strides().data()),
						postOutput_);
				}
			} else
			{
				//raw gradients for red, green, blue, alpha
				torch::Tensor fdInput = forwardDifferencesOutput.select(4 /*C*/, fd_outputChannel);
				fd_minGradient = fdInput.min().item<float>();
				fd_maxGradient = fdInput.max().item<float>();
				//fdInput is now of shape [1, H, W, 1] -> map to diverging color
				kernel::divergentColorMap(
					kernel::Tensor4Read(fdInput.data_ptr<real_t>(), fdInput.sizes().data(), fdInput.strides().data()),
					fd_minGradient, fd_maxGradient, postOutput_);
			}
			
			redrawMode_ = RedrawNone;
		}
	}

	cudaMemcpy(screenTextureCudaBuffer_, postOutput_, 4 * displayWidth_*displayHeight_,
		cudaMemcpyDeviceToDevice);
	copyBufferToOpenGL();

	//draw texture
	drawer_.drawQuad(screenTextureGL_);
}

void Visualizer::copyBufferToOpenGL()
{
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &screenTextureCuda_, 0));
	cudaArray* texture_ptr;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, screenTextureCuda_, 0, 0));
	size_t size_tex_data = sizeof(GLubyte) * displayWidth_ * displayHeight_ * 4;
	CUMAT_SAFE_CALL(cudaMemcpyToArray(texture_ptr, 0, 0, screenTextureCudaBuffer_, size_tex_data, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &screenTextureCuda_, 0));
}

void Visualizer::resize(int display_w, int display_h)
{
	//make it a nice multiplication of everything
	const int multiply = 4 * 3;
	display_w = display_w / multiply * multiply;
	display_h = display_h / multiply * multiply;

	if (display_w == displayWidth_ && display_h == displayHeight_)
		return;
	if (display_w == 0 || display_h == 0)
		return;
	releaseResources();
	displayWidth_ = display_w;
	displayHeight_ = display_h;

	//create texture
	glGenTextures(1, &screenTextureGL_);
	glBindTexture(GL_TEXTURE_2D, screenTextureGL_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA8,
		displayWidth_, displayHeight_, 0
		, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	//register with cuda
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(
		&screenTextureCuda_, screenTextureGL_,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//create channel output buffer
	CUMAT_SAFE_CALL(cudaMalloc(&screenTextureCudaBuffer_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));
	CUMAT_SAFE_CALL(cudaMalloc(&postOutput_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));

	glBindTexture(GL_TEXTURE_2D, 0);

	//create output structures
	rendererOutputsCpu_.color = torch::empty({ 1, displayHeight_, displayWidth_, 4 },
		at::TensorOptions().dtype(real_dtype).device(at::kCPU));
	rendererOutputsCpu_.terminationIndex = torch::empty({ 1, displayHeight_, displayWidth_ },
		at::TensorOptions().dtype(at::kInt).device(at::kCPU));
	rendererOutputsGpu_.color = torch::empty({ 1, displayHeight_, displayWidth_, 4 },
		at::TensorOptions().dtype(real_dtype).device(at::kCUDA));
	rendererOutputsGpu_.terminationIndex = torch::empty({ 1, displayHeight_, displayWidth_ },
		at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

	triggerRedraw(RedrawRenderer);
	std::cout << "Visualizer::resize(): " << displayWidth_ << ", " << displayHeight_ << std::endl;
}

void Visualizer::triggerRedraw(RedrawMode mode)
{
	redrawMode_ = std::max(redrawMode_, mode);
}

void Visualizer::screenshot()
{
	std::string folder = "screenshots";

	char time_str[128];
	time_t now = time(0);
	struct tm tstruct;
	localtime_s(&tstruct, &now);
	strftime(time_str, sizeof(time_str), "%Y%m%d-%H%M%S", &tstruct);

	char output_name[512];
	sprintf(output_name, "%s/screenshot_%s.png", folder.c_str(), time_str);

	std::cout << "Take screenshot: " << output_name << std::endl;
	std::filesystem::create_directory(folder);

	std::vector<GLubyte> textureCpu(4 * displayWidth_ * displayHeight_);
	CUMAT_SAFE_CALL(cudaMemcpy(&textureCpu[0], screenTextureCudaBuffer_, 4 * displayWidth_*displayHeight_, cudaMemcpyDeviceToHost));

	if (lodepng_encode32_file(output_name, textureCpu.data(), displayWidth_, displayHeight_) != 0)
	{
		std::cerr << "Unable to save image" << std::endl;
		screenshotString_ = std::string("Unable to save screenshot to ") + output_name;
	}
	else
	{
		screenshotString_ = std::string("Screenshot saved to ") + output_name;
	}
	screenshotTimer_ = 2.0f;
}

float3 Visualizer::computeCenterOfMass(const renderer::Volume* volume, const torch::Tensor& tfTensorTextureCpu)
{
	int X = volume->getLevel(0)->sizeX();
	int Y = volume->getLevel(0)->sizeY();
	int Z = volume->getLevel(0)->sizeZ();
	const auto volAcc = volume->getLevel(0)->dataCpu().accessor<real_t, 4>();
	const auto tfAcc = tfTensorTextureCpu.accessor<real_t, 3>();
	const int R = tfAcc.size(1);
	real3 center = make_real3(0, 0, 0);
	real_t mass = 0;

	using namespace indicators;
	ProgressBar bar{
		option::BarWidth{50},
		option::Start{"["},
		option::Fill{"="},
		option::Lead{">"},
		option::Remainder{" "},
		option::End{"]"},
		option::ForegroundColor{Color::white},
		option::ShowElapsedTime{true},
		option::ShowRemainingTime{true},
		option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
		option::MaxProgress{X}
	};
	for (int x = 0; x < X; ++x)
	{
#pragma omp parallel
		{
			real3 localCenter = make_real3(0, 0, 0);
			real_t localMass = 0;

#pragma omp for
			for (int y = 0; y < Y; ++y)
				for (int z = 0; z < Z; ++z)
				{
					real_t val = volAcc[0][x][y][z];
					int idx = clamp(static_cast<int>(val * (R - 1)), 0, R - 1);
					real_t alpha = tfAcc[0][idx][3];
					localCenter += alpha * make_real3(x, y, z);
					localMass += alpha;
				}

#pragma omp critical
			{
				center += localCenter;
				mass += localMass;
			}
		}

		bar.tick();
	}

	real3 c = (center / mass);
	return make_float3(c.x, c.y, c.z);
}
