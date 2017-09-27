#version 450 core
in vec2 UV;

out vec4 color;

layout(binding=0) uniform samplerCube renderedTexture;

vec2 iResolution;
void main()
{
    iResolution=vec2(1080.0,720.0); // TODO: move this outside the loop
    vec2 texCoord = gl_FragCoord.xy / iResolution.xy;
    vec2 thetaphi = ((texCoord * 2.0) - vec2(1.0)) * vec2(3.1415926535897932384626433832795, 1.5707963267948966192313216916398);
    //vec2 c = cos(thetaphi), s = sin(thetaphi);
    //color = texture(renderedTexture, vec3(vec2(s.x, c.x) * c.y, s.y));
    vec3 rayDirection = vec3(cos(thetaphi.y) * cos(thetaphi.x), sin(thetaphi.y), cos(thetaphi.y) * sin(thetaphi.x));
    color = texture(renderedTexture, rayDirection);
    //vec3 rayDirection = vec3(sin(thetaphi.x) * sin(thetaphi.y), cos(thetaphi.y), cos(thetaphi.x) * sin(thetaphi.y));
    //color = texture(renderedTexture, rayDirection);
}
