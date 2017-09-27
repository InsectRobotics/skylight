#version 450 core

out vec4 color;

layout(binding=1) uniform samplerCube EnvMap;

uniform vec2 resolution;
void main()
{
    vec2 texCoord = gl_FragCoord.xy / resolution.xy;
    vec2 thetaphi = ((texCoord * 2.0) - vec2(1.0)) * vec2(3.1415926535897932384626433832795, 1.5707963267948966192313216916398);
    //vec2 c = cos(thetaphi), s = sin(thetaphi);
    //color = texture(renderedTexture, vec3(vec2(s.x, c.x) * c.y, s.y));
    vec3 rayDirection = vec3(cos(thetaphi.y) * cos(thetaphi.x), sin(thetaphi.y), cos(thetaphi.y) * sin(thetaphi.x));
    color = texture(EnvMap, rayDirection);
    //vec3 rayDirection = vec3(sin(thetaphi.x) * sin(thetaphi.y), cos(thetaphi.y), cos(thetaphi.x) * sin(thetaphi.y));
    //color = texture(renderedTexture, rayDirection);
}
