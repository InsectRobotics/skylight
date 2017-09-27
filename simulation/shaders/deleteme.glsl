#version 430 core
layout(local_size_x=16, local_size_y=16) in;

uniform sampler2D tex;
layout(binding = 0) buffer Output {
    vec4 output[16][16];
};

void main()
{
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    //output[y][x] = texelFetch(tex, ivec2(x, y), 0);
}