#version 450
#extension GL_ARB_compute_variable_group_size : enable

//layout(local_size_variable) in ;
layout (local_size_x = 16, local_size_y = 16) in;             //definition of local work group size, default value is 1 (so don't need to set z in this example)
//layout (rgba32f, binding = 0 ) readonly uniform image2D input_image;
//layout (rgba32f, binding = 0 ) uniform sampler2D input_image;JS
layout (binding = 1 ) writeonly uniform image2D output_image;
layout (binding = 0 ) uniform sampler2D input_image;
//layout (binding = 1 ) uniform sampler2D output_image;

void main() {


    ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    //vec4 texel_color = imageLoad (input_image , index);JS
    vec4 texel_color = texture2D (input_image , index);
    vec3 tex_mex = vec3(1.0,1.0,1.0) - texel_color.rgb;
    //vec4 result_color = vec4(tex_mex ,texel_color.a);         //todo - revert back to original form?
    vec4 result_color = vec4(1.0,0.0,0.5,0.5);

            //vec4 result_color = vec4(1.0 − texel_color.rgb ,texel_color.a);
    imageStore (output_image , index , result_color) ;
    //output_image = texture(output_image, index);


}
