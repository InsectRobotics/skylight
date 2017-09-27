#version 450

layout(location = 10) in vec3 position;    // this is the container of cube verts
out vec4 gl_Position;

void main()
{
    gl_Position =  vec4(position, 1.0); //scale *
}

