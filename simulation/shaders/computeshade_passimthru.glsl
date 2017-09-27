 #version 450
 // use OpenGL 4.3’s GLSL with Compute Shaders
 #define TILE_WIDTH 16
 #define TILE_HEIGHT 16
 const ivec2 tileSize = ivec2(TILE_WIDTH,TILE_HEIGHT);
 layout(binding=0,rgba8) uniform image2D input_image;
 layout(binding=1,rgba8) uniform image2D output_image;
 layout(local_size_x=TILE_WIDTH,local_size_y=TILE_HEIGHT) in;

 void main() {
     const ivec2 tile_xy = ivec2(gl_WorkGroupID);
     const ivec2 thread_xy = ivec2(gl_LocalInvocationID);
     const ivec2 pixel_xy = tile_xy*tileSize + thread_xy;
     vec4 pixel = imageLoad(input_image, pixel_xy);
     //vec4 pixel = vec4(0.0, 1.0, 0.0, 1.0);
     //vec4 pixel = vec4(0, 1, 0, 1);
     imageStore(output_image, pixel_xy, pixel);
 }