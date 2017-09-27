#version 450
in vec3 ColorG;
layout(location = 0) out vec4 FragColor;
void main() {
    FragColor = vec4(ColorG, 1.0);
}