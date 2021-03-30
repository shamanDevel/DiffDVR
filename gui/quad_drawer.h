#pragma once

#include <GL/glew.h>
#include "imgui/imgui.h"

class QuadDrawer
{
public:
	QuadDrawer();
	~QuadDrawer();

	void drawQuad(GLuint texture, ImVec2 min = ImVec2(-1,-1), ImVec2 max = ImVec2(+1,+1), bool hasAlpha = true) const;

private:
	GLuint program_;
	GLuint modelLoc_;
	GLuint texLoc_;
	GLuint transparentVAO{}, transparentVBO{};
};
