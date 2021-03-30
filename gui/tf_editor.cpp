#include "tf_editor.h"
#include "visualizer_kernels.h"

#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuMat/src/Context.h>
#include <fstream>
#include <tinyxml2.h>


#include "pytorch_utils.h"
#include "renderer_utils.cuh"
#include "utils.h"

static std::vector<float> linspace(float a, float b, std::size_t N)
{
	//contains the endpoint!
	//Use N-1 if endpoint should be omitted
	float h = (b - a) / static_cast<float>(N);
	std::vector<float> xs(N);
	std::vector<float>::iterator x;
	float val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
		*x = val;
	}
	return xs;
}

TfPartPiecewiseOpacity::TfPartPiecewiseOpacity()
	: controlPoints_({ ImVec2(0.45f, 0.0f), ImVec2(0.5f, 0.8f), ImVec2(0.55f, 0.0f) })
	, densityAxis_({ 0.45f, 0.5f, 0.55f })
	, opacityAxis_({ 0.0f, 0.8f, 0.0f })
{}

void TfPartPiecewiseOpacity::init(const ImRect& rect)
{
	tfEditorRect_ = rect;
}

void TfPartPiecewiseOpacity::updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis)
{
	assert(densityAxis.size() == opacityAxis.size());
	assert(densityAxis.size() >= 1 && opacityAxis.size() >= 1);

	selectedControlPoint_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	opacityAxis_ = opacityAxis;

	int size = densityAxis.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.emplace_back(densityAxis_[i], opacityAxis_[i]);
	}
}

void TfPartPiecewiseOpacity::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!TfPiecewiseLinearEditor::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPoint_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		controlPoints_.push_back(screenToEditor(mousePosition));
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPoint_ >= 0)
		{
			isChanged_ = true;

			ImVec2 center(std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x),
				std::min(std::max(mousePosition.y, tfEditorRect_.Min.y), tfEditorRect_.Max.y));

			controlPoints_[selectedControlPoint_] = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			for (int idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx]));
				if (TfPiecewiseLinearEditor::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPoint_ = idx;
					break;
				}
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		for (int idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx]));
			if (TfPiecewiseLinearEditor::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPoint_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPoint_ = -1;
	}
}

void TfPartPiecewiseOpacity::render()
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect_.Min, tfEditorRect_.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ImVec2& p1, const ImVec2& p2)
		{
			return p1.x < p2.x;
		});

	//Fill densityAxis_ and opacityAxis_ and convert coordinates from editor space to screen space.
	densityAxis_.clear();
	opacityAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.x);
		opacityAxis_.push_back(cp.y);
		cp = editorToScreen(cp);
	}

	//Draw lines between the control points.
	int size = controlPointsRender.size();
	for (int i = 0; i < size + 1; ++i)
	{
		auto left = (i == 0) ? ImVec2(tfEditorRect_.Min.x, controlPointsRender.front().y) : controlPointsRender[i - 1];
		auto right = (i == size) ? ImVec2(tfEditorRect_.Max.x, controlPointsRender.back().y) : controlPointsRender[i];

		window->DrawList->AddLine(left, right, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 1.0f);
	}

	//Draw the control points
	for (const auto& cp : controlPointsRender)
	{
		window->DrawList->AddCircleFilled(cp, circleRadius_, ImColor(ImVec4(0.0f, 1.0f, 0.0f, 1.0f)), 16);
	}
}

ImRect TfPartPiecewiseOpacity::createControlPointRect(const ImVec2& controlPoint)
{
	return ImRect(ImVec2(controlPoint.x - circleRadius_, controlPoint.y - circleRadius_),
		ImVec2(controlPoint.x + circleRadius_, controlPoint.y + circleRadius_));
}

ImVec2 TfPartPiecewiseOpacity::screenToEditor(const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect_.Min.y) / (tfEditorRect_.Max.y - tfEditorRect_.Min.y);

	return editorPosition;
}

ImVec2 TfPartPiecewiseOpacity::editorToScreen(const ImVec2& editorPosition)
{
	ImVec2 screenPosition;
	screenPosition.x = editorPosition.x * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;
	screenPosition.y = (1.0f - editorPosition.y) * (tfEditorRect_.Max.y - tfEditorRect_.Min.y) + tfEditorRect_.Min.y;

	return screenPosition;
}



TfPartTextureOpacity::TfPartTextureOpacity(int resolution)
	: resolution_(resolution)
	, opacityAxis_(linspace(0, 1, resolution))
{}

void TfPartTextureOpacity::init(const ImRect& rect)
{
	resized_ = false;
	if (tfEditorRect_.GetWidth() != rect.GetWidth()) {
		int N = rect.GetWidth();
		resized_ = true;
		plot_.resize(N);
		for (int i=0; i<N; ++i)
		{
			float p = i * resolution_ / float(N);
			int a = int(p);
			float f = p - a;
			int a2 = clamp(a, 0, resolution_ - 1);
			int b2 = clamp(a + 1, 0, resolution_ - 1);
			plot_[i] = lerp(opacityAxis_[a2], opacityAxis_[b2], f);
		}
	}
	tfEditorRect_ = rect;
	
}

void TfPartTextureOpacity::updateControlPoints(const std::vector<float>& opacities)
{
	assert(opacities.size() == opacityAxis_.size());

	std::copy(opacities.begin(), opacities.end(), opacityAxis_.begin());
	
	isChanged_ = true;
}

void TfPartTextureOpacity::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!TfPiecewiseLinearEditor::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPoint_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
	}
	else if (isLeftClicked)
	{
		ImVec2 p = screenToEditor(mousePosition);
		//Draw line from lastPos_to p
		const auto drawLine = [](std::vector<float>& d, const ImVec2& start, const ImVec2& end)
		{
			int N = d.size();
			int xa = clamp(int(std::round(start.x * (N - 1))), 0, N - 1);
			int xb = clamp(int(std::round(end.x * (N - 1))), 0, N - 1);
			float ya = start.y;
			float yb = end.y;
			if (xa <= xb)
			{
				for (int x = xa; x <= xb; ++x)
				{
					float f = (x - xa) / float(xb - xa);
					d[x] = lerp(ya, yb, f);
				}
			}
			else
			{
				for (int x = xb; x <= xa; ++x)
				{
					float f = (x - xb) / float(xa - xb);
					d[x] = lerp(yb, ya, f);
				}
			}
		};
		if (wasClicked_) {
			//std::cout << "Line from " << lastPos_.x << "," << lastPos_.y
			//	<< " to " << p.x << "," << p.y << std::endl;
			drawLine(plot_, lastPos_, p);
			drawLine(opacityAxis_, lastPos_, p);
			isChanged_ = true;
		}
		lastPos_ = p;
		wasClicked_ = true;
	}
	else if (isRightClicked)
	{
	}
	else if (isLeftReleased)
	{
		wasClicked_ = false;
	}
}

void TfPartTextureOpacity::render()
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect_.Min, tfEditorRect_.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//Draw line
	int N = plot_.size();
	ImU32 col = ImGui::GetColorU32(ImGuiCol_PlotLines);
	for (int i=1; i<N; ++i)
	{
		ImVec2 a = ImLerp(tfEditorRect_.Min, tfEditorRect_.Max, 
			ImVec2((i-1)/(N-1.0f), lerp(0.05f, 0.95f, 1-plot_[i-1])));
		ImVec2 b = ImLerp(tfEditorRect_.Min, tfEditorRect_.Max,
			ImVec2(i / (N - 1.0f), lerp(0.05f, 0.95f, 1-plot_[i])));
		window->DrawList->AddLine(a, b, col, thickness_);
	}
}

ImVec2 TfPartTextureOpacity::screenToEditor(const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect_.Min.y) / (tfEditorRect_.Max.y - tfEditorRect_.Min.y);
	editorPosition.y = 1 / 18.0f * (20 * editorPosition.y - 1);
	editorPosition.x = clamp(editorPosition.x, 0.0f, 1.0f);
	editorPosition.y = clamp(editorPosition.y, 0.0f, 1.0f);
	return editorPosition;
}



TfPartPiecewiseColor::TfPartPiecewiseColor(ITextureEditor* parent)
	: densityAxis_({ 0.0f, 1.0f }), parent_(parent)
{
	auto red = renderer::rgbToLab(make_float3(1.0f, 0.0f, 0.0f));
	auto white = renderer::rgbToLab(make_float3(1.0f, 1.0f, 1.0f));

	controlPoints_.emplace_back(0.0f, red.x, red.y, red.z);
	controlPoints_.emplace_back(1.0f, white.x, white.y, white.z);

	colorAxis_.push_back(red);
	colorAxis_.push_back(white);
}

TfPartPiecewiseColor::~TfPartPiecewiseColor()
{
	destroy();
}

void TfPartPiecewiseColor::init(const ImRect& rect, bool showControlPoints)
{
	ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputHSV;
	ImGui::ColorEdit3("", &pickedColor_.x, colorFlags);

	showControlPoints_ = showControlPoints;

	//If editor is created for the first time or its size is changed, create CUDA texture.
	if (tfEditorRect_.Min.x == FLT_MAX ||
		!(rect.Min.x == tfEditorRect_.Min.x &&
			rect.Min.y == tfEditorRect_.Min.y &&
			rect.Max.x == tfEditorRect_.Max.x &&
			rect.Max.y == tfEditorRect_.Max.y))
	{
		destroy();
		tfEditorRect_ = rect;

		auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
		auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;
		glGenTextures(1, &colorMapImage_);

		glBindTexture(GL_TEXTURE_2D, colorMapImage_);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorMapWidth, colorMapHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);
		//CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&resource_, colorMapImage_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

		glBindTexture(GL_TEXTURE_2D, 0);

		//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		//CUMAT_SAFE_CALL(cudaMallocArray(&contentArray_, &channelDesc, colorMapWidth, colorMapHeight, cudaArraySurfaceLoadStore));

		//cudaResourceDesc resDesc;
		//memset(&resDesc, 0, sizeof(resDesc));
		//resDesc.resType = cudaResourceTypeArray;

		//resDesc.res.array.array = contentArray_;
		//CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&content_, &resDesc));
		isColorMapInitialized_ = false;
	}
}

void TfPartPiecewiseColor::updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis)
{
	selectedControlPointForMove_ = -1;
	selectedControlPointForColor_ = -1;
	controlPoints_.clear();
	densityAxis_ = densityAxis;
	colorAxis_ = colorAxis;

	int size = densityAxis.size();
	for (int i = 0; i < size; ++i)
	{
		controlPoints_.emplace_back(densityAxis_[i], colorAxis_[i].x, colorAxis_[i].y, colorAxis_[i].z);
	}
	isChanged_ = true;
	isColorMapInitialized_ = false;
}

void TfPartPiecewiseColor::handleIO()
{
	isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	if (selectedControlPointForColor_ >= 0)
	{
		auto& cp = controlPoints_[selectedControlPointForColor_];

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::rgbToLab(pickedColorLab);

		if (cp.y != pickedColorLab.x || cp.z != pickedColorLab.y ||
			cp.w != pickedColorLab.z)
		{
			cp.y = pickedColorLab.x;
			cp.z = pickedColorLab.y;
			cp.w = pickedColorLab.z;
			isChanged_ = true;
		}
	}

	//Early leave if mouse is not on color editor.
	if (!TfPiecewiseLinearEditor::testIntersectionRectPoint(tfEditorRect_, mousePosition) && selectedControlPointForMove_ == -1)
	{
		return;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		float3 pickedColorLab;
		ImGui::ColorConvertHSVtoRGB(pickedColor_.x, pickedColor_.y, pickedColor_.z, pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
		pickedColorLab = renderer::rgbToLab(pickedColorLab);
		controlPoints_.emplace_back(screenToEditor(mousePosition.x), pickedColorLab.x, pickedColorLab.y, pickedColorLab.z);
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPointForMove_ >= 0)
		{
			isChanged_ = true;

			float center = std::min(std::max(mousePosition.x, tfEditorRect_.Min.x), tfEditorRect_.Max.x);

			controlPoints_[selectedControlPointForMove_].x = screenToEditor(center);
		}
		//Check whether new point is selected.
		else
		{
			int size = controlPoints_.size();
			int idx;
			for (idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
				if (TfPiecewiseLinearEditor::testIntersectionRectPoint(cp, mousePosition))
				{
					selectedControlPointForColor_ = selectedControlPointForMove_ = idx;

					auto colorRgb = renderer::labToRgb(make_float3(controlPoints_[selectedControlPointForMove_].y,
						controlPoints_[selectedControlPointForMove_].z,
						controlPoints_[selectedControlPointForMove_].w));

					ImGui::ColorConvertRGBtoHSV(colorRgb.x, colorRgb.y, colorRgb.z, pickedColor_.x, pickedColor_.y, pickedColor_.z);
					break;
				}
			}

			//In case of no hit on any control point, unselect for color pick as well.
			if (idx == size)
			{
				selectedControlPointForColor_ = -1;
			}
		}
	}
	else if (isRightClicked)
	{
		int size = controlPoints_.size();
		int idx;
		for (idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(controlPoints_[idx].x));
			if (TfPiecewiseLinearEditor::testIntersectionRectPoint(cp, mousePosition) && controlPoints_.size() > 1)
			{
				isChanged_ = true;

				controlPoints_.erase(controlPoints_.begin() + idx);
				selectedControlPointForColor_ = selectedControlPointForMove_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPointForMove_ = -1;
	}
}

void TfPartPiecewiseColor::render()
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();

	//Copy the control points and sort them. We don't sort original one in order not to mess up with control point indices.
	auto controlPointsRender = controlPoints_;
	std::sort(controlPointsRender.begin(), controlPointsRender.end(),
		[](const ImVec4& cp1, const ImVec4& cp2)
		{
			return cp1.x < cp2.x;
		});

	//Fill densityAxis_ and colorAxis_.
	densityAxis_.clear();
	colorAxis_.clear();
	for (auto& cp : controlPointsRender)
	{
		densityAxis_.push_back(cp.x);
		colorAxis_.push_back(make_float3(cp.y, cp.z, cp.w));
	}

	auto colorMapWidth = tfEditorRect_.Max.x - tfEditorRect_.Min.x;
	auto colorMapHeight = tfEditorRect_.Max.y - tfEditorRect_.Min.y;

	//Write to color map texture.

	if (isChanged_ || !isColorMapInitialized_)
	{
		isColorMapInitialized_ = true;
		torch::Tensor colorTensor = parent_->getTextureTensor(
			colorMapWidth, 0.0, 1.0, 1.0, false);
		const auto acc = colorTensor.accessor<real_t, 3>();
		std::vector<unsigned int> pixelData(colorMapWidth * colorMapHeight);
		for (int x=0; x<colorMapWidth; ++x)
		{
			float r = acc[0][x][0];
			float g = acc[0][x][1];
			float b = acc[0][x][2];
			unsigned int rgba = kernel::rgbaToInt(r, g, b, 1.0f);
			for (int y=0; y<colorMapHeight; ++y)
			{
				pixelData[x + colorMapWidth * y] = rgba;
			}
		}
		glBindTexture(GL_TEXTURE_2D, colorMapImage_);
		glTexSubImage2D(
			GL_TEXTURE_2D, 0, 0, 0, colorMapWidth, colorMapHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV,
			pixelData.data());
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	
	//kernel::fillColorMap(content_, tfTexture_.getTextureObject(), colorMapWidth, colorMapHeight);
	////Draw color interpolation between control points.
	//cudaArray_t texturePtr = nullptr;
	//CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &resource_, 0));
	//CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texturePtr, resource_, 0, 0));
	//CUMAT_SAFE_CALL(cudaMemcpyArrayToArray(texturePtr, 0, 0, contentArray_, 0, 0, colorMapWidth * colorMapHeight * 4, cudaMemcpyDeviceToDevice));
	//CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource_, 0));

	window->DrawList->AddImage((void*)colorMapImage_, tfEditorRect_.Min, tfEditorRect_.Max);

	if (showControlPoints_)
	{
		//Draw the control points
		int cpIndex = 0;
		for (const auto& cp : controlPoints_)
		{
			//If this is the selected control point, use different color.
			auto rect = createControlPointRect(editorToScreen(cp.x));
			if (selectedControlPointForColor_ == cpIndex++)
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 0.8f, 0.1f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 3.0f);
			}
			else
			{
				window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 2.0f);
			}
		}
	}
}

void TfPartPiecewiseColor::destroy()
{
	if (colorMapImage_)
	{
		glDeleteTextures(1, &colorMapImage_);
		colorMapImage_ = 0;
	}
	//if (content_)
	//{
	//	CUMAT_SAFE_CALL(cudaDestroySurfaceObject(content_));
	//	content_ = 0;
	//}
	//if (contentArray_)
	//{
	//	CUMAT_SAFE_CALL(cudaFreeArray(contentArray_));
	//	contentArray_ = nullptr;
	//}
}

ImRect TfPartPiecewiseColor::createControlPointRect(float x)
{
	return ImRect(ImVec2(x - 0.5f * cpWidth_, tfEditorRect_.Min.y),
		ImVec2(x + 0.5f * cpWidth_, tfEditorRect_.Max.y));
}

float TfPartPiecewiseColor::screenToEditor(float screenPositionX)
{
	float editorPositionX;
	editorPositionX = (screenPositionX - tfEditorRect_.Min.x) / (tfEditorRect_.Max.x - tfEditorRect_.Min.x);

	return editorPositionX;
}

float TfPartPiecewiseColor::editorToScreen(float editorPositionX)
{
	float screenPositionX;
	screenPositionX = editorPositionX * (tfEditorRect_.Max.x - tfEditorRect_.Min.x) + tfEditorRect_.Min.x;

	return screenPositionX;
}



TfPiecewiseLinearEditor::TfPiecewiseLinearEditor()
	: editorColor_(this)
{
}

void TfPiecewiseLinearEditor::init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints)
{
	editorOpacity_.init(tfEditorOpacityRect);
	editorColor_.init(tfEditorColorRect, showColorControlPoints);
}

void TfPiecewiseLinearEditor::handleIO()
{
	editorOpacity_.handleIO();
	editorColor_.handleIO();
}

void TfPiecewiseLinearEditor::render()
{
	editorOpacity_.render();
	editorColor_.render();
}

void TfPiecewiseLinearEditor::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	nlohmann::json json;
	json["densityAxisOpacity"] = editorOpacity_.getDensityAxis();
	json["opacityAxis"] = editorOpacity_.getOpacityAxis();
	json["densityAxisColor"] = editorColor_.getDensityAxis();
	json["colorAxis"] = editorColor_.getColorAxis();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;

	std::ofstream out(path);
	out << json;
	out.close();
}

void TfPiecewiseLinearEditor::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	if (path.size() > 4 && path.substr(path.size() - 4, 4) == ".xml")
	{
		//try to load colormap (no opacities)
		try
		{
			auto colormap = loadColormapFromXML(path);
			//convert rgb to lab
			for (auto& color : colormap.second)
				color = renderer::rgbToLab(color);
			editorColor_.updateControlPoints(colormap.first, colormap.second);
		}
		catch (const std::runtime_error& ex)
		{
			std::cerr << "Unable to load colormap from xml: " << ex.what() << std::endl;
		}
	}
	
	nlohmann::json json;
	std::ifstream file(path);
	file >> json;
	file.close();

	std::vector<float> densityAxisOpacity = json["densityAxisOpacity"];
	std::vector<float> opacityAxis = json["opacityAxis"];
	std::vector<float> densityAxisColor = json["densityAxisColor"];
	std::vector<float3> colorAxis = json["colorAxis"];
	minDensity = json["minDensity"];
	maxDensity = json["maxDensity"];

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

nlohmann::json TfPiecewiseLinearEditor::toJson() const
{
	const auto& densityAxisOpacity = editorOpacity_.getDensityAxis();
	const auto& opacityAxis = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();
	//std::vector<nlohmann::json> colorAxis2;
	//std::transform(colorAxis.begin(), colorAxis.end(), std::back_inserter(colorAxis2),
	//	[](float3 color)
	//{
	//	return nlohmann::json::array({ color.x, color.y, color.z });
	//});
	return {
		{"densityAxisOpacity", nlohmann::json(densityAxisOpacity)},
		{"opacityAxis", nlohmann::json(opacityAxis)},
		{"densityAxisColor", nlohmann::json(densityAxisColor)},
		{"colorAxis", nlohmann::json(colorAxis)}
	};
}

void TfPiecewiseLinearEditor::fromJson(const nlohmann::json& s)
{
	const std::vector<float> densityAxisOpacity = s.at("densityAxisOpacity");
	const std::vector<float> opacityAxis = s.at("opacityAxis");
	const std::vector<float> densityAxisColor = s.at("densityAxisColor");
	const std::vector<float3> colorAxis = s.at("colorAxis");
	//std::vector<float3> colorAxis;
	//std::transform(colorAxis2.begin(), colorAxis2.end(), std::back_inserter(colorAxis),
	//	[](nlohmann::json color)
	//{
	//	return make_float3(color[0].get<float>(), color[1].get<float>(), color[2].get<float>());
	//});
	editorOpacity_.updateControlPoints(densityAxisOpacity, opacityAxis);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

std::pair<std::vector<float>, std::vector<float3>> TfPiecewiseLinearEditor::loadColormapFromXML(const std::string& path)
{
	using namespace tinyxml2;
	XMLDocument doc;
	doc.LoadFileThrow(path.c_str());
	const XMLElement* element = doc.FirstChildElementThrow("ColorMaps")
		->FirstChildElementThrow("ColorMap");

	std::vector<float> positions;
	std::vector<float3> rgbColors;
	const XMLElement* point = element->FirstChildElementThrow("Point");
	do
	{
		positions.push_back(point->FloatAttribute("x"));
		rgbColors.push_back(make_float3(
			point->FloatAttribute("r"),
			point->FloatAttribute("g"),
			point->FloatAttribute("b")
		));
		point = point->NextSiblingElement("Point");
	} while (point != nullptr);

	return std::make_pair(positions, rgbColors);
}

bool TfPiecewiseLinearEditor::testIntersectionRectPoint(const ImRect& rect, const ImVec2& point)
{
	return (rect.Min.x <= point.x &&
		rect.Max.x >= point.x &&
		rect.Min.y <= point.y &&
		rect.Max.y >= point.y);
}

torch::Tensor TfPiecewiseLinearEditor::getTextureTensor(
	int resolution, float minDensity, float maxDensity, float opacityScaling,
	bool purgeZeroOpacityRegions) const
{
	const auto points = assembleMergedPoints(
		minDensity, maxDensity, opacityScaling, purgeZeroOpacityRegions);
	return renderer::TFUtils::getTextureTensor(points, resolution);
}

torch::Tensor TfPiecewiseLinearEditor::getPiecewiseTensor(
	float minDensity, float maxDensity, float opacityScaling,
	bool purgeZeroOpacityRegions) const
{
	const auto points = assembleMergedPoints(
		minDensity, maxDensity, opacityScaling, purgeZeroOpacityRegions);
	return renderer::TFUtils::getPiecewiseTensor(points);
}

std::vector<renderer::TFPoint> TfPiecewiseLinearEditor::assembleMergedPoints(
	float minDensity, float maxDensity, float opacityScaling,
	bool purgeZeroOpacityRegions) const
{
	auto colorValues = getColorAxis();
	auto colorPositions = getDensityAxisColor();
	auto opacityValues = getOpacityAxis();
	auto opacityPositions = getDensityAxisOpacity();

	return renderer::TFUtils::assembleFromSettings(colorValues, colorPositions,
		opacityValues, opacityPositions, minDensity, maxDensity, opacityScaling,
		purgeZeroOpacityRegions);
}



TfTextureEditor::TfTextureEditor()
	: editorColor_(this)
{
}

void TfTextureEditor::init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints)
{
	editorOpacity_.init(tfEditorOpacityRect);
	editorColor_.init(tfEditorColorRect, showColorControlPoints);
}

void TfTextureEditor::handleIO()
{
	editorOpacity_.handleIO();
	editorColor_.handleIO();
}

void TfTextureEditor::render()
{
	editorOpacity_.render();
	editorColor_.render();
}

void TfTextureEditor::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	nlohmann::json json;
	json["opacities"] = editorOpacity_.getOpacityAxis();;
	json["densityAxisColor"] = editorColor_.getDensityAxis();
	json["colorAxis"] = editorColor_.getColorAxis();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;

	std::ofstream out(path);
	out << json;
	out.close();
}

void TfTextureEditor::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	if (path.size() > 4 && path.substr(path.size() - 4, 4) == ".xml")
	{
		//try to load colormap (no opacities)
		try
		{
			auto colormap = TfPiecewiseLinearEditor::loadColormapFromXML(path);
			//convert rgb to lab
			for (auto& color : colormap.second)
				color = renderer::rgbToLab(color);
			editorColor_.updateControlPoints(colormap.first, colormap.second);
		}
		catch (const std::runtime_error& ex)
		{
			std::cerr << "Unable to load colormap from xml: " << ex.what() << std::endl;
		}
	}

	nlohmann::json json;
	std::ifstream file(path);
	file >> json;
	file.close();

	std::vector<float> opacities = json["opacities"];
	std::vector<float> densityAxisColor = json["densityAxisColor"];
	std::vector<float3> colorAxis = json["colorAxis"];
	minDensity = json["minDensity"];
	maxDensity = json["maxDensity"];

	assert(densityAxisOpacity.size() == opacityAxis.size());
	assert(densityAxisColor.size() == colorAxis.size());

	editorOpacity_.updateControlPoints(opacities);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

nlohmann::json TfTextureEditor::toJson() const
{
	const auto& opacities = editorOpacity_.getOpacityAxis();
	const auto& densityAxisColor = editorColor_.getDensityAxis();
	const auto& colorAxis = editorColor_.getColorAxis();
	//std::vector<nlohmann::json> colorAxis2;
	//std::transform(colorAxis.begin(), colorAxis.end(), std::back_inserter(colorAxis2),
	//	[](float3 color)
	//{
	//	return nlohmann::json::array({ color.x, color.y, color.z });
	//});
	return {
		{"opacities", nlohmann::json(opacities)},
		{"densityAxisColor", nlohmann::json(densityAxisColor)},
		{"colorAxis", nlohmann::json(colorAxis)}
	};
}

void TfTextureEditor::fromJson(const nlohmann::json& s)
{
	const std::vector<float> opacities = s.at("opacities");
	const std::vector<float> densityAxisColor = s.at("densityAxisColor");
	const std::vector<float3> colorAxis = s.at("colorAxis");
	editorOpacity_.updateControlPoints(opacities);
	editorColor_.updateControlPoints(densityAxisColor, colorAxis);
}

torch::Tensor TfTextureEditor::getTextureTensor(
	int resolution, float minDensity, float maxDensity, float opacityScaling,
	bool purgeZeroOpacityRegions) const
{
	auto colorValues = getColorAxis();
	auto colorPositions = getDensityAxisColor();
	auto opacityValues = getOpacities();
	auto opacityPositions = linspace(0, 1, opacityValues.size());

	const auto points = renderer::TFUtils::assembleFromSettings(colorValues, colorPositions,
		opacityValues, opacityPositions, minDensity, maxDensity, opacityScaling,
		purgeZeroOpacityRegions);

	return renderer::TFUtils::getTextureTensor(points, resolution);
}

TfGaussianEditor::TfGaussianEditor()
	: isChanged_(true)
	, points_{
		Point{ImVec4(1,0,0,alpha1_), 0.6f, 0.7f, 0.05f},
		Point{ImVec4(0,1,0,alpha1_), 0.3f, 0.3f, 0.03f}
	}
{
}

void TfGaussianEditor::render(const ImRect& tfEditorRect)
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(tfEditorRect.Min, tfEditorRect.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//draw lines
	static const int Samples = 100;
	const int numPoints = points_.size();
	//mixture
	const auto eval = [this, numPoints](float x)
	{
		float opacity = 0;
		float3 color = make_float3(0, 0, 0);
		for (int i = 0; i < numPoints; ++i) {
			float w = points_[i].opacity * normal(x, points_[i].mean, points_[i].variance);
			opacity += w;
			color += make_float3(points_[i].color.x, points_[i].color.y, points_[i].color.z) * w;
		}
		color /= opacity;
		return std::make_pair(opacity, ImGui::GetColorU32(ImVec4(color.x, color.y, color.z, alpha2_)));
	};
	auto [o0, c0] = eval(0);
	float x0 = 0;
	for (int j = 1; j <= Samples; ++j)
	{
		float x1 = j / float(Samples);
		const auto [o1, c1] = eval(x1);

		ImVec2 a = ImLerp(tfEditorRect.Min, tfEditorRect.Max,
			ImVec2(x0, 1 - o0));
		ImVec2 b = ImLerp(tfEditorRect.Min, tfEditorRect.Max,
			ImVec2(x1, 1 - o1));
		window->DrawList->AddLine(a, b, c0, thickness2_);

		x0 = x1;
		o0 = o1;
		c0 = c1;
	}
	//single gaussians
	for (int i=0; i<numPoints; ++i)
	{
		ImU32 col = ImGui::GetColorU32(points_[i].color);
		float opacity = points_[i].opacity;
		float mean = points_[i].mean;
		float variance = points_[i].variance;
		float denom = 1.0f / (2 * variance * variance);
		float y0 = opacity*expf(-mean * mean * denom);
		float x0 = 0;
		for (int j=1; j<=Samples; ++j)
		{
			float x1 = j / float(Samples);
			float y1 = opacity*expf(-(x1 - mean) * (x1 - mean) * denom);
			if (y0 > 1e-3 || y1 > 1e-3) {
				ImVec2 a = ImLerp(tfEditorRect.Min, tfEditorRect.Max,
					ImVec2(x0, 1 - y0));
				ImVec2 b = ImLerp(tfEditorRect.Min, tfEditorRect.Max,
					ImVec2(x1, 1 - y1));
				window->DrawList->AddLine(a, b, col, thickness1_);
			}
			y0 = y1; x0 = x1;
		}
	}

	//draw control points
	float minDistance = FLT_MAX;
	int bestSelection = -1;
	auto mousePosition = ImGui::GetMousePos();
	for (int i = 0; i < numPoints; ++i)
	{
		ImU32 col = ImGui::GetColorU32(points_[i].color);
		float opacity = points_[i].opacity;
		float mean = points_[i].mean;
		auto cp = editorToScreen(tfEditorRect, ImVec2(mean, opacity));
		window->DrawList->AddCircleFilled(cp, circleRadius_, col, 16);
		window->DrawList->AddCircle(cp, circleRadius_ + 1, ImGui::GetColorU32(ImVec4(0, 0, 0, 1)));
		if (i==selectedPoint_)
			window->DrawList->AddCircle(cp, circleRadius_ + 2, ImGui::GetColorU32(ImGuiCol_Text));
		float dist = ImLengthSqr(ImVec2(mousePosition.x - cp.x, mousePosition.y - cp.y));
		if (dist<minDistance)
		{
			minDistance = dist;
			bestSelection = i;
		}
	}

	//handle io
	isChanged_ = false;
	if (TfPiecewiseLinearEditor::testIntersectionRectPoint(tfEditorRect, mousePosition) 
		|| draggedPoint_ != -1)
	{
		bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
		bool isLeftClicked = ImGui::IsMouseDown(0);
		bool isRightClicked = ImGui::IsMouseClicked(1);
		bool isLeftReleased = ImGui::IsMouseReleased(0);

		auto mouseEditor = screenToEditor(tfEditorRect, mousePosition);
		
		if (isLeftDoubleClicked)
		{
			isChanged_ = true;
			points_.push_back({
				ImVec4(1,1,1,alpha1_),
				mouseEditor.y,
				mouseEditor.x,
				0.01f
				}
			);
			selectedPoint_ = points_.size() - 1;
			draggedPoint_ = -1;
		} else if (isLeftClicked)
		{
			if (draggedPoint_>=0)
			{
				//move selected point
				isChanged_ = true;
				ImVec2 center(std::min(std::max(mousePosition.x, tfEditorRect.Min.x), tfEditorRect.Max.x),
					std::min(std::max(mousePosition.y, tfEditorRect.Min.y), tfEditorRect.Max.y));
				auto p = screenToEditor(tfEditorRect, center);
				points_[draggedPoint_].mean = p.x;
				points_[draggedPoint_].opacity = p.y;
			} else
			{
				//select new point for hovering
				if (minDistance < (circleRadius_+2)*(circleRadius_+2))
				{
					draggedPoint_ = bestSelection;
					selectedPoint_ = bestSelection;
				}
			}
		} else if (isRightClicked)
		{
			if (minDistance < (circleRadius_ + 2) * (circleRadius_ + 2))
			{
				isChanged_ = true;
				points_.erase(points_.begin() + bestSelection);
				selectedPoint_ = points_.empty() ? -1 : 0;
				draggedPoint_ = -1;
			}
		} else if (isLeftReleased)
		{
			draggedPoint_ = -1;
		}
	}

	//controls
	if (selectedPoint_ >= 0) {
		ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB;
		if (ImGui::ColorEdit3("", &points_[selectedPoint_].color.x, colorFlags))
			isChanged_ = true;
		if (ImGui::SliderFloat("Variance", &points_[selectedPoint_].variance,
			0.001f, 0.5f, "%.3f", 2))
			isChanged_ = true;
	}
}

void TfGaussianEditor::saveToFile(const std::string& path, float minDensity, float maxDensity) const
{
	nlohmann::json json;
	json["type"] = "GaussianTF";
	json["data"] = toJson();
	json["minDensity"] = minDensity;
	json["maxDensity"] = maxDensity;

	std::ofstream out(path);
	out << json;
	out.close();
}

void TfGaussianEditor::loadFromFile(const std::string& path, float& minDensity, float& maxDensity)
{
	nlohmann::json json;
	std::ifstream file(path);
	file >> json;
	file.close();

	if (!json.contains("type") || json["type"] != "GaussianTF") {
		std::cerr << "Not a gaussian TF!" << std::endl;
		return;
		//throw std::runtime_error("not a gaussian transfer function");
	}
	auto data = json["data"];
	fromJson(data);
	minDensity = json["minDensity"];
	maxDensity = json["maxDensity"];
}

nlohmann::json TfGaussianEditor::toJson() const
{
	nlohmann::json a = nlohmann::json::array();
	for (int i = 0; i < points_.size(); ++i)
	{
		a.push_back(nlohmann::json::array({
			points_[i].color.x,
			points_[i].color.y,
			points_[i].color.z,
			points_[i].opacity,
			points_[i].mean,
			points_[i].variance
		}));
	}
	return a;
}

void TfGaussianEditor::fromJson(const nlohmann::json& s)
{
	assert(s.is_array());
	points_.resize(s.size());
	for (int i=0; i<s.size(); ++i)
	{
		auto e = s[i];
		assert(e.is_array());
		assert(e.size() == 6);
		points_[i] = {
			ImVec4(e[0], e[1], e[2], alpha1_), e[3], e[4], e[5]
		};
	}
	isChanged_ = true;
}

torch::Tensor TfGaussianEditor::getGaussianTensor(float minDensity, float maxDensity, float opacityScaling) const
{
	int numPoints = static_cast<int>(points_.size());
	torch::Tensor t = torch::empty(
		{ 1, numPoints, 6 },
		at::TensorOptions().dtype(real_dtype));
	auto acc = t.packed_accessor32<real_t, 3>();
	for (int i=0; i<numPoints; ++i)
	{
		acc[0][i][0] = points_[i].color.x;
		acc[0][i][1] = points_[i].color.y;
		acc[0][i][2] = points_[i].color.z;
		acc[0][i][3] = points_[i].opacity * opacityScaling;
		acc[0][i][4] = lerp(minDensity, maxDensity, points_[i].mean);
		acc[0][i][5] = points_[i].variance * (maxDensity-minDensity);
	}
	return t;
}

float TfGaussianEditor::normal(float x, float mean, float variance)
{
	return expf(-(x - mean) * (x - mean) / (2 * variance * variance));
}

ImVec2 TfGaussianEditor::screenToEditor(const ImRect& tfEditorRect, const ImVec2& screenPosition)
{
	ImVec2 editorPosition;
	editorPosition.x = (screenPosition.x - tfEditorRect.Min.x) / (tfEditorRect.Max.x - tfEditorRect.Min.x);
	editorPosition.y = 1.0f - (screenPosition.y - tfEditorRect.Min.y) / (tfEditorRect.Max.y - tfEditorRect.Min.y);

	return editorPosition;
}

ImVec2 TfGaussianEditor::editorToScreen(const ImRect& tfEditorRect, const ImVec2& editorPosition)
{
	ImVec2 screenPosition;
	screenPosition.x = editorPosition.x * (tfEditorRect.Max.x - tfEditorRect.Min.x) + tfEditorRect.Min.x;
	screenPosition.y = (1.0f - editorPosition.y) * (tfEditorRect.Max.y - tfEditorRect.Min.y) + tfEditorRect.Min.y;

	return screenPosition;
}
