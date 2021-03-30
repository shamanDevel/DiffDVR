#pragma once

#include "imgui.h"

enum ImGuiExtDirectionPickerFlags_
{
	ImGuiExtDirectionPickerFlags_None		= 0,
	ImGuiExtDirectionPickerFlags_InvertX	= 1 << 0,
	ImGuiExtDirectionPickerFlags_InvertY	= 1 << 1,
	ImGuiExtDirectionPickerFlags_InvertXY	=
		ImGuiExtDirectionPickerFlags_InvertX | ImGuiExtDirectionPickerFlags_InvertY
};
typedef int ImGuiExtDirectionPickerFlags;

namespace ImGuiExt
{
	//Direction picker of a vector v=(x,y,z) where z is always positive
	IMGUI_API bool DirectionPicker2D(const char* label, float v[3], ImGuiExtDirectionPickerFlags flags = 0);
	IMGUI_API bool Spinner(const char* label, float radius, float thickness, const ImU32& color);
	IMGUI_API bool Spinner(const char* label);
}