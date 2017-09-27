#version 450
layout(location = 0) in vec4 VertexPosition;
layout(location = 1) in vec3 VertexColor;
//uniform mat4 MVP;
uniform mat4 scale;
out vec3 ColorV;
void main()
{
ColorV = VertexColor;
gl_Position =  VertexPosition; //scale *
}