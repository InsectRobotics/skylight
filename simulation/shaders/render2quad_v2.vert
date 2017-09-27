#version 450 core

//layout(location = 2) in vec3 vertexPosition_modelspace;
layout (location = 4) in vec3 position;
//layout(location = 3) in vec2 vertexUV; //JS added

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main(){
    //gl_Position = projection * view * vec4(position, 0.5); // vertexPosition_modelspace.xyz;
    //gl_Position.xyz = pos
    //gl_Position.w = 0.5;
    //UV = vertexUV;

    vec4 pos = projection * view * vec4(position, 0.5);
    gl_Position = pos.xyww;



}
