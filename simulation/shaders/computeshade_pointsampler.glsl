#version 450

//layout(local_size_x=16, local_size_y=1) in;
layout(local_size_x=512, local_size_y=1) in;       // try thread by thread to start with

// cant use this structure with texture function
//struct Direction {
//  float x, y, z;
//};


layout(std430, binding = 4) buffer views_buffer              // todo specify mem protocol: layout( std140, binding=4 ) /std430
{
    //Todo - currently have to pass in nx4 dimension buffer where the 4th column is discarded, work out how to implement
    // todo - cont. glunpackalignment instead
    vec3 viewdirs[];
};

layout(std430, binding = 5) buffer response_buffer
{
    vec4 response[];
};

layout(std430, binding = 6) buffer check_buffer
{
    vec4 viewdirs_out[];
};
//
//layout(std430, binding = 3) buffer checkidx_buffer
//{
//    float idx_out[];
//};

layout(binding = 0) uniform samplerCube  Envmap;

void main() {

    response[gl_GlobalInvocationID.x] = texture(Envmap, viewdirs[gl_GlobalInvocationID.x].xyz);
    viewdirs_out[gl_GlobalInvocationID.x] = vec4(viewdirs[gl_GlobalInvocationID.x], 0.5);
    //idx_out[gl_GlobalInvocationID.x] = float(gl_GlobalInvocationID.x);

}
