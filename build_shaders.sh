#!/bin/bash

pushd shaders
glslangValidator.exe colored_triangle.vert -V -o colored_triangle.vert.spv
glslangValidator.exe colored_triangle.frag -V -o colored_triangle.frag.spv
glslangValidator.exe triangle.vert -V -o triangle.vert.spv
glslangValidator.exe triangle.frag -V -o triangle.frag.spv
glslangValidator.exe tri_mesh.vert -V -o tri_mesh.vert.spv
popd