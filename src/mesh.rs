use std::convert::TryInto;

use ash::vk::{
    Format, PipelineVertexInputStateCreateFlags, VertexInputAttributeDescription,
    VertexInputBindingDescription, VertexInputRate,
};
use memoffset::offset_of;
use nalgebra_glm::Vec3;

use crate::AllocatedBuffer;

pub struct VertexInputDescription {
    pub bindings: Vec<VertexInputBindingDescription>,
    pub attributes: Vec<VertexInputAttributeDescription>,
    pub flags: PipelineVertexInputStateCreateFlags,
}

#[repr(C, packed)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
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
            .format(Format::R32G32B32_SFLOAT)
            //NOTE(lucypero): this offset i do on my own, might be wrong
            .offset(offset_of!(Vertex, position) as u32)
            .build();

        //Normal will be stored at Location 1
        let normal_attribute = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            //NOTE(lucypero): this offset i do on my own, might be wrong
            .offset(offset_of!(Vertex, normal) as u32)
            .build();

        //Color will be stored at Location 2
        let color_attribute = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(Format::R32G32B32_SFLOAT)
            //NOTE(lucypero): this offset i do on my own, might be wrong
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
