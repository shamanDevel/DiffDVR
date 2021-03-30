#include "quad_drawer.h"
#include <GL/glew.h>
#include <iostream>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.inl>

QuadDrawer::QuadDrawer()
	: program_(0)
{
	static const char* VERTEX_SHADER = R"shader(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 model;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = model * vec4(aPos, 1.0);
}
)shader";

	static const char* FRAGMENT_SHADER = R"shader(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D texture1;

void main()
{             
    FragColor = texture(texture1, TexCoords);
}
)shader";

	// compile shaders
	GLuint vertex, fragment;
	int success;
	char infoLog[512];

	// vertex Shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &VERTEX_SHADER, NULL);
	glCompileShader(vertex);
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		return;
	}

	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &FRAGMENT_SHADER, NULL);
	glCompileShader(fragment);
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		return;
	}

	program_ = glCreateProgram();
	glAttachShader(program_, vertex);
	//glAttachShader(program_, geometry);
	glAttachShader(program_, fragment);
	glLinkProgram(program_);
	// print linking errors if any
	glGetProgramiv(program_, GL_LINK_STATUS, &success);
	if (!success)
	{
		glGetProgramInfoLog(program_, 512, NULL, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		return;
	}

	//get uniforms
	modelLoc_ = glGetUniformLocation(program_, "model");
	texLoc_ = glGetUniformLocation(program_, "texture1");

	glDeleteShader(vertex);
	//glDeleteShader(geometry);
	glDeleteShader(fragment);

	//VBA
	float transparentVertices[] = {
		// positions         // texture Coords (swapped y coordinates because texture is flipped upside down)
		0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
		0.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		1.0f,  0.0f,  0.0f,  1.0f,  1.0f,

		0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
		1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  0.0f,  1.0f,  0.0f
	};
	glGenVertexArrays(1, &transparentVAO);
	glGenBuffers(1, &transparentVBO);
	glBindVertexArray(transparentVAO);
	glBindBuffer(GL_ARRAY_BUFFER, transparentVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(transparentVertices), transparentVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glBindVertexArray(0);
}

QuadDrawer::~QuadDrawer()
{
	glDeleteProgram(program_);
	glDeleteVertexArrays(1, &transparentVAO);
	glDeleteBuffers(1, &transparentVBO);
}

void QuadDrawer::drawQuad(GLuint texture, ImVec2 min, ImVec2 max, bool hasAlpha) const
{
	glUseProgram(program_);

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(min.x, min.y, 0));
	model = glm::scale(model, glm::vec3(max.x - min.x, max.y - min.y, 1));
	glUniformMatrix4fv(modelLoc_, 1, false, &model[0].x);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(texLoc_, 0);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glBindVertexArray(transparentVAO);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}
