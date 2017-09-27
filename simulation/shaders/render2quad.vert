#version 330 core
layout(location = 2) in vec3 vertexPosition_modelspace;
layout(location = 3) in vec2 vertexUV; //JS added

out vec2 UV;

void main(){
    gl_Position.xyz = vertexPosition_modelspace.xyz;
    gl_Position.w = 1.0;
    UV = vertexUV;
}
