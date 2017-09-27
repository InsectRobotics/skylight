#version 450
in vec3 gs_TexCoords;
out vec4 color;

//uniform samplerCube skybox;
layout(binding=1) uniform samplerCube skybox;

void main()
{
    color = texture(skybox, gs_TexCoords);

}