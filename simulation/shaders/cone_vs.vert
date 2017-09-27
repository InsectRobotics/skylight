#version 330 core
layout(location = 8) in vec3 verts_in;
uniform vec2 cone_centre;

//todo - check if more efficient to process gl_position verts in a single line .xyzw

void main(){

    gl_Position.x = verts_in.x + cone_centre.x;
    gl_Position.y = verts_in.y + cone_centre.y;
    gl_Position.z = verts_in.z;
    gl_Position.w = 1.0;
}