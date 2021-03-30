#include "imgui_extension.h"

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include "imgui_internal.h"

#include <cmath>
#include "IconsFontAwesome5.h"
#include <algorithm>

using namespace ImGui;

bool ImGuiExt::DirectionPicker2D(const char* label, float v[3], ImGuiExtDirectionPickerFlags flags)
{
	ImGuiContext& g = *GImGui;
	ImGuiWindow* window = GetCurrentWindow();
	if (window->SkipItems)
		return false;

	ImDrawList* draw_list = window->DrawList;
	ImGuiStyle& style = g.Style;
	ImGuiIO& io = g.IO;

	const float width = CalcItemWidth();
	g.NextItemData.ClearFlags();

	PushID(label);
	BeginGroup();

	bool value_changed = false;
	float backup_initial_v[3] = { v[0], v[1], v[2] };
	int invertX = (flags&ImGuiExtDirectionPickerFlags_InvertX) ? -1 : 1;
	int invertY = (flags&ImGuiExtDirectionPickerFlags_InvertY) ? -1 : 1;

	//normalize vector
	float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	float currentX = invertX * v[0] / len;
	float currentY = invertY * v[1] / len;
	float currentZ = v[2] / len;

	//setup
	PushItemFlag(ImGuiItemFlags_NoNav, true);
	float sv_picker_size = 3 * GetTextLineHeight() + 4 * g.Style.ItemSpacing.y;
	ImVec2 picker_pos = window->DC.CursorPos;
	ImVec2 wheel_center(picker_pos.x + sv_picker_size*0.5f, picker_pos.y + sv_picker_size * 0.5f);
	float wheel_radius = sv_picker_size * 0.47f;

	//logic
	InvisibleButton("dir", ImVec2(sv_picker_size, sv_picker_size));
	if (IsItemActive())
	{
		ImVec2 initial_off = g.IO.MouseClickedPos[0] - wheel_center;
		ImVec2 current_off = g.IO.MousePos - wheel_center;
		ImVec2 currentXY = current_off / wheel_radius;
		float currentLen = std::sqrt(ImLengthSqr(currentXY));
		if (currentLen > 1)
			currentXY /= currentLen;
		currentX = currentXY.x;
		currentY = currentXY.y;
		currentZ = std::sqrt(std::max(0.0f, 1 - currentX * currentX - currentY * currentY));
		value_changed = true;
	}
	PopItemFlag(); // ImGuiItemFlags_NoNav

	//text
	SameLine(0, style.ItemInnerSpacing.x);
	BeginGroup();
	const char* label_display_end = FindRenderedTextEnd(label);
	if (label != label_display_end)
	{
		TextEx(label, label_display_end);
	}
	Text("X=%+5.3f, Y=%+5.3f, Z=%+5.3f", currentX, currentY, currentZ);
	if (Button(ICON_FA_ARROW_ALT_CIRCLE_LEFT))
	{
		currentX = 0;
		currentY = 0;
		currentZ = 1;
		value_changed = true;
	}
	EndGroup();

	//render
	const int style_alpha8 = IM_F32_TO_INT8_SAT(style.Alpha);
	const ImU32 col_black = IM_COL32(0, 0, 0, style_alpha8);
	const ImU32 col_white = IM_COL32(255, 255, 255, style_alpha8);
	const ImU32 col_midgrey = IM_COL32(128, 128, 128, style_alpha8);
	const ImU32 col_lightgrey = IM_COL32(180, 180, 180, style_alpha8);
	//wheel
	draw_list->AddCircleFilled(wheel_center, wheel_radius, col_midgrey, 32);
	draw_list->AddCircle(wheel_center, wheel_radius, col_lightgrey, 32);
	draw_list->AddCircle(wheel_center, 0.05f*wheel_radius, col_black);
	//cursor
	ImVec2 hue_cursor_pos = wheel_center + ImVec2(currentX, currentY)*wheel_radius;
	float hue_cursor_rad = value_changed ? wheel_radius * 0.1f : wheel_radius * 0.07f;
	int hue_cursor_segments = ImClamp((int)(hue_cursor_rad / 1.4f), 9, 32);
	draw_list->AddCircleFilled(hue_cursor_pos, hue_cursor_rad, col_white, hue_cursor_segments);
	draw_list->AddCircle(hue_cursor_pos, hue_cursor_rad + 1, col_midgrey, hue_cursor_segments);
	draw_list->AddCircle(hue_cursor_pos, hue_cursor_rad, col_black, hue_cursor_segments);

	EndGroup();

	v[0] = currentX * invertX;
	v[1] = currentY * invertY;
	v[2] = currentZ;
	if (value_changed && memcmp(backup_initial_v, v, 3 * sizeof(float)) == 0)
		value_changed = false;
	if (value_changed) {
		MarkItemEdited(window->DC.LastItemId);
	}

	PopID();

	return value_changed;
}


// Progress spinner by zfedoran
// https://github.com/ocornut/imgui/issues/1901
bool ImGuiExt::Spinner(const char* label, float radius, float thickness, const ImU32& color) {
	ImGuiWindow* window = GetCurrentWindow();
	if (window->SkipItems)
		return false;

	ImGuiContext& g = *GImGui;
	const ImGuiStyle& style = g.Style;
	const ImGuiID id = window->GetID(label);

	ImVec2 pos = window->DC.CursorPos;
	ImVec2 size((radius) * 2, (radius + style.FramePadding.y) * 2);

	const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
	ItemSize(bb, style.FramePadding.y);
	if (!ItemAdd(bb, id))
		return false;

	// Render
	window->DrawList->PathClear();

	int num_segments = 30;
	int start = abs(ImSin(g.Time*1.8f)*(num_segments - 5));

	const float a_min = IM_PI * 2.0f * ((float)start) / (float)num_segments;
	const float a_max = IM_PI * 2.0f * ((float)num_segments - 3) / (float)num_segments;

	const ImVec2 centre = ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

	for (int i = 0; i < num_segments; i++) {
		const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
		window->DrawList->PathLineTo(ImVec2(centre.x + ImCos(a + g.Time * 8) * radius,
			centre.y + ImSin(a + g.Time * 8) * radius));
	}

	window->DrawList->PathStroke(color, false, thickness);
	return true;
}

bool ImGuiExt::Spinner(const char* label)
{
	const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
	return ImGuiExt::Spinner(label, GetCurrentContext()->Font->FontSize * 0.5f, 2.0f, col);
}