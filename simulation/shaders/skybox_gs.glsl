#version 450 core
#define M_PI 3.1415926535897932384626433832795

// Triangles in, four invocations (instances)
layout (triangles, invocations = 6) in;
// Triangles (strips) out, 3 vertices each
layout (triangle_strip, max_vertices = 3) out;

in vec3 vs_TexCoords[];
out vec3 gs_TexCoords;

uniform mat4 projection;
uniform mat4 view;


const mat4 rots[6] = mat4[6]    // TODO perform this in python - opengl bible p553 (load as uniform buffer)
(
    mat4(cos(M_PI/2),0,-sin(M_PI/2),0,     0,1,0,0,    sin(M_PI/2),0,cos(M_PI/2),0,   0,0,0,1),    //-x
    mat4(cos(-M_PI/2),0,-sin(-M_PI/2),0,     0,1,0,0,    sin(-M_PI/2),0,cos(-M_PI/2),0,   0,0,0,1),      //-z
    mat4(1,0,0,0,   0,cos(-M_PI/2),-sin(-M_PI/2),0,     0, sin(-M_PI/2),cos(-M_PI/2),0,    0,0,0,1),    //+y
    mat4(1,0,0,0,   0,cos(M_PI/2),-sin(M_PI/2),0,     0, sin(M_PI/2),cos(M_PI/2),0,    0,0,0,1),       //-y
    mat4(1),         //+x
    mat4(cos(M_PI),0,-sin(M_PI),0,     0,1,0,0,    sin(M_PI),0,cos(M_PI),0,   0,0,0,1)              // +z
);

void main()
{
    for (int i = 0; i < gl_in.length(); i++)
        {
            gl_Layer = gl_InvocationID;
            vec4 pos = projection * rots[gl_InvocationID] * view * gl_in[i].gl_Position;
            gl_Position = pos.xyww;
            gs_TexCoords = gl_in[i].gl_Position.xyz;
            EmitVertex();
        }
}