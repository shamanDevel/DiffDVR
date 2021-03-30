#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <deque>
#include <vector>

#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "tf_utils.h"
#include <json.hpp>

#include <torch/types.h>

struct ITextureEditor
{
	virtual torch::Tensor getTextureTensor(
		int resolution, float minDensity, float maxDensity, float opacityScaling,
		bool purgeZeroOpacityRegions) const = 0;
};

class TfPartPiecewiseOpacity
{
public:
	TfPartPiecewiseOpacity();
	void init(const ImRect& rect);
	void updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float>& opacityAxis);
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float>& getOpacityAxis() const { return opacityAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float circleRadius_{ 4.0f };

	ImRect tfEditorRect_;
	int selectedControlPoint_{ -1 };
	std::deque<ImVec2> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float> opacityAxis_;

	bool isChanged_{ false };

private:
	ImRect createControlPointRect(const ImVec2& controlPoint);
	ImVec2 screenToEditor(const ImVec2& screenPosition);
	ImVec2 editorToScreen(const ImVec2& editorPosition);
};

class TfPartTextureOpacity
{
public:
	TfPartTextureOpacity(int resolution = 256);
	void init(const ImRect& rect);
	void updateControlPoints(const std::vector<float>& opacities);
	void handleIO();
	void render();

	const std::vector<float>& getOpacityAxis() const { return opacityAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float thickness_{ 2.0f };

	ImRect tfEditorRect_;
	int selectedControlPoint_{ -1 };
	const int resolution_;
	std::vector<float> opacityAxis_;

	bool resized_ = false;
	std::vector<float> plot_;

	bool wasClicked_ = false;
	ImVec2 lastPos_;

	bool isChanged_{ false };

private:
	ImVec2 screenToEditor(const ImVec2& screenPosition);
};

class TfPartPiecewiseColor
{
public:
	//Non-copyable and non-movable
	TfPartPiecewiseColor(ITextureEditor* parent);
	~TfPartPiecewiseColor();
	TfPartPiecewiseColor(const TfPartPiecewiseColor&) = delete;
	TfPartPiecewiseColor(TfPartPiecewiseColor&&) = delete;

	void init(const ImRect& rect, bool showControlPoints);
	void updateControlPoints(const std::vector<float>& densityAxis, const std::vector<float3>& colorAxis);
	void handleIO();
	void render();

	const std::vector<float>& getDensityAxis() const { return densityAxis_; }
	const std::vector<float3>& getColorAxis() const { return colorAxis_; }
	bool getIsChanged() const { return isChanged_; }

private:
	const float cpWidth_{ 8.0f };
	ITextureEditor* parent_;

	ImVec4 pickedColor_{ 0.0f, 0.0f, 1.0f, 1.0f };
	ImRect tfEditorRect_;
	int selectedControlPointForMove_{ -1 };
	int selectedControlPointForColor_{ -1 };
	std::deque<ImVec4> controlPoints_;
	std::vector<float> densityAxis_;
	std::vector<float3> colorAxis_;
	bool isChanged_{ false };
	bool showControlPoints_{ true };

	//Variables for color map texture.
	//cudaGraphicsResource* resource_{ nullptr };
	GLuint colorMapImage_{ 0 };
	bool isColorMapInitialized_{ false };
	//cudaSurfaceObject_t content_{ 0 };
	//cudaArray_t contentArray_{ nullptr };
	
	//RENDERER_NAMESPACE::TfTexture1D tfTexture_;

private:
	void destroy();
	ImRect createControlPointRect(float x);
	float screenToEditor(float screenPositionX);
	float editorToScreen(float editorPositionX);
};

class TfPiecewiseLinearEditor : public ITextureEditor
{
public:
	TfPiecewiseLinearEditor();
	void init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints);
	void handleIO();
	void render();
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);
	//Loads the colormap (position + rgb) from an xml.
	static std::pair< std::vector<float>, std::vector<float3> >
		loadColormapFromXML(const std::string& path);

	const std::vector<float>& getDensityAxisOpacity() const { return editorOpacity_.getDensityAxis(); }
	const std::vector<float>& getOpacityAxis() const { return editorOpacity_.getOpacityAxis(); }
	const std::vector<float>& getDensityAxisColor() const { return editorColor_.getDensityAxis(); }
	const std::vector<float3>& getColorAxis() const { return editorColor_.getColorAxis(); }
	bool getIsChanged() const { return editorOpacity_.getIsChanged() || editorColor_.getIsChanged(); }

	static bool testIntersectionRectPoint(const ImRect& rect, const ImVec2& point);

	/**
	 * \brief Returns the transfer function as a tensor / 1D texture
	 * of shape 1*R*C where R is the resolution and C=4 the channels
	 * with red, green, blue, opacity.
	 * The texels in the texture tensor are equal-spaced.
	 * The color space is RGB (even though the tf is stored in Lab)
	 */
	torch::Tensor getTextureTensor(
		int resolution, float minDensity, float maxDensity, float opacityScaling,
		bool purgeZeroOpacityRegions) const override;

	/**
	 * \brief Returns the transfer function as a piecewise function
	 * of shape 1*R*C where R is the resolution and C=5 the channels
	 * with red, green, blue, opacity, position of the control point.
	 * The first control point has always position 0, the last always
	 * position 1.
	 * The color space is RGB (even though the tf is stored in Lab)
	 */
	torch::Tensor getPiecewiseTensor(
		float minDensity, float maxDensity, float opacityScaling,
		bool purgeZeroOpacityRegions) const;
	
private:
	TfPartPiecewiseOpacity editorOpacity_;
	TfPartPiecewiseColor editorColor_;

	std::vector<renderer::TFPoint> assembleMergedPoints(
		float minDensity, float maxDensity, float opacityScaling,
		bool purgeZeroOpacityRegions) const;
};

class TfTextureEditor : public ITextureEditor
{
public:
	TfTextureEditor();
	void init(const ImRect& tfEditorOpacityRect, const ImRect& tfEditorColorRect, bool showColorControlPoints);
	void handleIO();
	void render();
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	const std::vector<float>& getOpacities() const { return editorOpacity_.getOpacityAxis(); }
	const std::vector<float>& getDensityAxisColor() const { return editorColor_.getDensityAxis(); }
	const std::vector<float3>& getColorAxis() const { return editorColor_.getColorAxis(); }
	bool getIsChanged() const { return editorOpacity_.getIsChanged() || editorColor_.getIsChanged(); }

	/**
	 * \brief Returns the transfer function as a tensor / 1D texture
	 * of shape 1*R*C where R is the resolution and C=4 the channels
	 * with red, green, blue, opacity.
	 * The texels in the texture tensor are equal-spaced.
	 * The color space is RGB (even though the tf is stored in Lab)
	 */
	torch::Tensor getTextureTensor(
		int resolution, float minDensity, float maxDensity, float opacityScaling,
		bool purgeZeroOpacityRegions) const override;

private:
	TfPartTextureOpacity editorOpacity_;
	TfPartPiecewiseColor editorColor_;
};

class TfGaussianEditor
{
public:
	TfGaussianEditor();
	void render(const ImRect& tfEditorRect);
	void saveToFile(const std::string& path, float minDensity, float maxDensity) const;
	void loadFromFile(const std::string& path, float& minDensity, float& maxDensity);
	nlohmann::json toJson() const;
	void fromJson(const nlohmann::json& s);

	bool getIsChanged() const { return isChanged_; }

	/**
	 * \brief Returns the transfer function as a tensor / 1D texture
	 * of shape 1*R*C where R is the resolution and C=6 the channels
	 * with red, green, blue, opacity, mean, variance.
	 * The color space is RGB
	 */
	torch::Tensor getGaussianTensor(
		float minDensity, float maxDensity, float opacityScaling) const;

private:
	static constexpr float circleRadius_ = 4.0f;
	static constexpr float alpha1_ = 0.4f;
	static constexpr float alpha2_ = 0.4f;
	static constexpr float thickness1_ = 3;
	static constexpr float thickness2_ = 2;
	bool isChanged_ = false;
	struct Point
	{
		ImVec4 color;
		float opacity;
		float mean;
		float variance;
	};
	std::vector<Point> points_;
	int selectedPoint_ = 0;
	int draggedPoint_ = 0;

	float normal(float x, float mean, float variance);

	ImVec2 screenToEditor(const ImRect& tfEditorRect, const ImVec2& screenPosition);
	ImVec2 editorToScreen(const ImRect& tfEditorRect, const ImVec2& editorPosition);
};
