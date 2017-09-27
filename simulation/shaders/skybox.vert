#version 450
#define M_PI 3.1415926535897932384626433832795 // temp
layout (location = 10) in vec3 position;
out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

//#uniform mat4 rot = mat4(cos(-M_PI),0,-sin(-M_PI),0,     0,1,0,0,    sin(-M_PI),0,cos(-M_PI),0,   0,0,0,1);
//uniform mat4 rot = mat4(cos(M_PI),0,-sin(M_PI),0,     0,1,0,0,    sin(M_PI),0,cos(M_PI),0,   0,0,0,1);
// not z uniform mat4 rot = mat4(cos(M_PI),-sin(M_PI),0,0,     sin(M_PI),cos(M_PI),0,0,   0,0,1,0,  0,0,0,1);
//uniform mat4 rot = mat4(1,0,0,0,     0,cos(M_PI),-sin(M_PI),0,     0,sin(M_PI),cos(M_PI),0,     0,0,0,1);
void main()
{

    vec4 pos = projection * view *  vec4(position, 1.0);
    gl_Position = pos.xyww;
    TexCoords = position;

    //gl_Position =   projection * view * vec4(position, 1.0);
    //TexCoords = position;
}


