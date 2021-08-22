use std::{
    convert::TryInto,
    fmt::{self, Debug},
    path::{Path, PathBuf},
};

use ash::vk::{
    BufferCreateInfo, BufferUsageFlags, Format, PipelineVertexInputStateCreateFlags,
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
};
use memoffset::offset_of;
use nalgebra_glm::{Vec3, Vec4, vec3, vec4};
use vk_mem::Allocator;

use crate::AllocatedBuffer;

pub struct VertexInputDescription {
    pub bindings: Vec<VertexInputBindingDescription>,
    pub attributes: Vec<VertexInputAttributeDescription>,
    pub flags: PipelineVertexInputStateCreateFlags,
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec4,
    pub color: Vec4,
}

impl Vertex {
    pub fn get_vertex_description() -> VertexInputDescription {
        //we will have just 1 vertex buffer binding, with a per-vertex rate
        let main_binding = VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>().try_into().unwrap())
            .input_rate(VertexInputRate::VERTEX)
            .build();

        //Position will be stored at Location 0
        let position_attribute = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(Format::R32G32B32A32_SFLOAT)
            .offset(offset_of!(Vertex, position) as u32)
            .build();

        //Normal will be stored at Location 1
        let normal_attribute = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32A32_SFLOAT)
            .offset(offset_of!(Vertex, normal) as u32)
            .build();

        //Color will be stored at Location 2
        let color_attribute = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(Format::R32G32B32A32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build();

        VertexInputDescription {
            bindings: vec![main_binding],
            attributes: vec![position_attribute, normal_attribute, color_attribute],
            flags: PipelineVertexInputStateCreateFlags::empty(),
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub vertex_buffer: AllocatedBuffer,
}

impl Mesh {
    pub unsafe fn load_from_obj<P>(path: P, allocator: &Allocator) -> Mesh
    where
        P: AsRef<Path> + fmt::Debug,
    {
        let (models, _materials) =
            tobj::load_obj(path, &tobj::LoadOptions::default()).expect("Failed to OBJ load file");

        // Note: If you don't mind missing the materials, you can generate a default.
        // let materials = materials.expect("Failed to load MTL file");
        //triangulate is false

        println!("Number of models = {}", models.len());
        // println!("Number of materials       = {}", materials.len());

        let mut vertices: Vec<Vertex> = vec![];

        for (_i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            assert!(mesh.indices.len() % 3 == 0);
            for index in 0..mesh.indices.len() {
                let position = vec4(
                    mesh.positions[(mesh.indices[index] as usize) * 3],
                    mesh.positions[(mesh.indices[index] as usize) * 3 + 1],
                    mesh.positions[(mesh.indices[index] as usize) * 3 + 2],
                    0.0);

                let normal = vec4(
                    mesh.normals[(mesh.normal_indices[index] as usize)* 3],
                    mesh.normals[(mesh.normal_indices[index] as usize)* 3 + 1],
                    mesh.normals[(mesh.normal_indices[index] as usize)* 3 + 2],
                    0.0);


                let vertex = Vertex{position, normal, color: normal};

                vertices.push(vertex)
            }

        }

        Mesh::new(vertices, allocator)
    }

    pub unsafe fn new(vertices: Vec<Vertex>, allocator: &Allocator) -> Mesh {
        let (buffer, allocation, _allocation_info) = allocator
            .create_buffer(
                &BufferCreateInfo::builder()
                    .size((vertices.len() * std::mem::size_of::<Vertex>()) as u64)
                    .usage(BufferUsageFlags::VERTEX_BUFFER)
                    .build(),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::CpuToGpu,
                    ..Default::default()
                },
            )
            .unwrap();

        let mesh = Mesh {
            vertices: vertices,
            vertex_buffer: AllocatedBuffer {
                buffer: buffer,
                allocation: allocation,
            },
        };

        //copy vertex data to gpu memory
        let data: *mut u8 = allocator.map_memory(&allocation).unwrap();

        std::ptr::copy_nonoverlapping(
            mesh.vertices.as_ptr().cast(),
            data,
            std::mem::size_of::<Vertex>() * mesh.vertices.len(),
        );

        allocator.unmap_memory(&allocation);

        mesh
    }
}
