#version 450

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vNormal;
layout (location = 2) in vec4 vColor;

layout (location = 0) out vec3 outColor;

//push constants block
layout( push_constant ) uniform constants
{
	vec4 data;
	mat4 render_matrix;
} PushConstants;

void main()
{
	gl_Position = PushConstants.render_matrix * vec4(vPosition.xyz, 1.0f);
	outColor = vec3(vColor.xyz);
}