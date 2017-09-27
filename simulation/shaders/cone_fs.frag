#version 450 core
uniform vec3 FragColor = vec3(0.0,1.0,0.0);
out vec4 color;


void main(){
  //color = vec3(0,1,0);
  color = vec4(FragColor,1);
}