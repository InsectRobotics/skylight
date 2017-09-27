#version 450
in vec3 TexCoords;
out vec4 color;

//layout(binding=1)
uniform samplerCube skybox;
float temp;
void main()
{
    //gl_Layer = 7;
    color = texture(skybox, TexCoords);
}