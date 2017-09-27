#version 450 core
#define M_PI 3.1415926535897932384626433832795

 // Triangles (strips) out, 3 vertices each
layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;  // TOdo: should this be just triangles?

in vec3 ColorV[];

int gl_ViewportIndex;

out vec3 ColorG;

uniform mat4 projection;
uniform mat4 VIEW;

const mat4 mods[6] = mat4[6]    // TODO perform this in python - opengl bible p553
(
    mat4(cos(M_PI/2),0,-sin(M_PI/2),0,     0,1,0,0,    sin(M_PI/2),0,cos(M_PI/2),0,   0,0,0,1),    //-x
    mat4(cos(-M_PI/2),0,-sin(-M_PI/2),0,     0,1,0,0,    sin(-M_PI/2),0,cos(-M_PI/2),0,   0,0,0,1),      //-z

    // y is top and bottom
    mat4(1,0,0,0,   0,cos(-M_PI/2),-sin(-M_PI/2),0,     0, sin(-M_PI/2),cos(-M_PI/2),0,    0,0,0,1),    //+y
    mat4(1,0,0,0,   0,cos(M_PI/2),-sin(M_PI/2),0,     0, sin(M_PI/2),cos(M_PI/2),0,    0,0,0,1),       //-y

    mat4(1),         //+x
    mat4(cos(M_PI),0,-sin(M_PI),0,     0,1,0,0,    sin(M_PI),0,cos(M_PI),0,   0,0,0,1)              // +z

);

void main()
{


for (int face = 0; face < 6; face++) {
        gl_Layer = face;    // built-in variable that specifies which face of the cubemap we render
        for (int i = 0; i < gl_in.length(); i++) {   // for each triangle's vertices
            gl_Position = projection *  mods[face] * VIEW  * gl_in[i].gl_Position;
            ColorG  = ColorV[i];
            //gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
