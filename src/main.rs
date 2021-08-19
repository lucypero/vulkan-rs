#![allow(unused_imports, dead_code)]

mod mesh;
mod platforms;

use std::borrow::Borrow;
use std::collections::VecDeque;
use std::ffi::{c_void, CStr, CString};
use std::fs::File;
use std::{mem, slice};
use std::os::raw::c_char;
use std::path::Path;
use std::str::FromStr;

use bytemuck::{Pod, Zeroable};

use ash::prelude::VkResult;
use ash::vk::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
    BufferCreateInfo, BufferUsageFlags, ClearColorValue, ClearValue, ColorComponentFlags,
    ColorSpaceKHR, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo,
    CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool,
    CommandPoolCreateFlags, CommandPoolCreateInfo, CommandPoolResetFlags, ComponentMapping,
    ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags, DeviceSize, Extent2D, Fence,
    FenceCreateFlags, FenceCreateInfo, Format, Framebuffer, FramebufferCreateInfo, FrontFace,
    GraphicsPipelineCreateInfo, Image, ImageAspectFlags, ImageLayout, ImageSubresourceRange,
    ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, LogicOp, Offset2D, Pipeline,
    PipelineBindPoint, PipelineCache, PipelineColorBlendAttachmentState,
    PipelineColorBlendStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
    PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags,
    PipelineVertexInputStateCreateInfo, PipelineVertexInputStateCreateInfoBuilder,
    PipelineViewportStateCreateInfo, PolygonMode, PresentInfoKHR, PresentModeKHR,
    PrimitiveTopology, PushConstantRange, Rect2D, RenderPass, RenderPassBeginInfo,
    RenderPassCreateInfo, SampleCountFlags, Semaphore, SemaphoreCreateInfo, ShaderModule,
    ShaderModuleCreateInfo, ShaderStageFlags, SubmitInfo, SubpassContents, SubpassDescription,
    SurfaceFormatKHR, SwapchainCreateInfoKHR, SwapchainCreateInfoKHRBuilder, SwapchainKHR,
    Viewport,
};
use ash::{
    extensions::{ext::DebugUtils, khr::Surface, khr::Swapchain},
    vk::{self, Handle},
    Entry,
};

use image::ImageFormat;
use mesh::Mesh;
use nalgebra_glm::{Mat4, Vec3, Vec4, mat4, vec1, vec3, vec4};
use vk_mem::ffi::VkBuffer;
use vk_mem::AllocationInfo;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

use crate::mesh::Vertex;

// Constants
const WINDOW_TITLE: &'static str = "vulkan-rs";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct PipelineBuilder {
    shader_stages: Vec<PipelineShaderStageCreateInfo>,
    vertex_input_info: PipelineVertexInputStateCreateInfo,
    input_assembly: PipelineInputAssemblyStateCreateInfo,
    viewport: Viewport,
    scissor: Rect2D,
    rasterizer: PipelineRasterizationStateCreateInfo,
    color_blend_attachment: PipelineColorBlendAttachmentState,
    multisampling: PipelineMultisampleStateCreateInfo,
    pipeline_layout: PipelineLayout,
}

impl PipelineBuilder {
    unsafe fn build_pipeline(&mut self, device: &ash::Device, render_pass: RenderPass) -> Pipeline {
        let viewport_state = PipelineViewportStateCreateInfo::builder()
            .viewports(&[self.viewport])
            .scissors(&[self.scissor])
            .build();

        let color_blending = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&[self.color_blend_attachment])
            .build();

        //build the actual pipeline
        //we now use all of the info structs we have been writing into into this one to create the pipeline
        let pipeline_info = GraphicsPipelineCreateInfo::builder()
            .stages(&self.shader_stages[..])
            .vertex_input_state(&self.vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .layout(self.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_handle(Pipeline::null())
            .build();

        let new_pipeline = device
            .create_graphics_pipelines(PipelineCache::null(), &[pipeline_info], None)
            .expect("couldn't create the pipeline");

        new_pipeline[0]
    }
}
pub struct AllocatedBuffer {
    pub buffer: ash::vk::Buffer,
    pub allocation: vk_mem::Allocation,
}

struct DeletionQueue {
    deletors: VecDeque<Box<dyn Fn(&VulkanApp)>>,
}

impl DeletionQueue {
    fn push_function<T>(&mut self, function: T)
    where
        T: Fn(&VulkanApp) + 'static,
    {
        self.deletors.push_back(Box::new(function));
    }

    fn flush(&mut self, app: &VulkanApp) {
        // reverse iterate the deletion queue to execute all the functions
        for deletor in &self.deletors {
            deletor(app);
        }

        self.deletors.clear();
    }
}
struct VulkanApp {
    entry: Entry,
    instance: ash::Instance,
    debug_utils_fn: DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    surface_fn: Surface,
    surface: vk::SurfaceKHR,
    surface_format: SurfaceFormatKHR,
    device: ash::Device,
    swapchain_loader: Swapchain,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_views: Vec<ImageView>,
    command_pool: CommandPool,
    main_cmd_buffer: CommandBuffer,
    render_pass: RenderPass,
    framebuffers: Vec<Framebuffer>,
    present_semaphore: Semaphore,
    render_semaphore: Semaphore,
    render_fence: Fence,
    queues: Queues,

    triangle_pipeline_layout: PipelineLayout,
    mesh_pipeline_layout: PipelineLayout,

    triangle_pipeline: Pipeline,
    red_triangle_pipeline: Pipeline,
    mesh_pipeline: Pipeline,
    triangle_mesh: Mesh,

    // "game" state
    frame_number: i64,

    selected_shader: i32,
    main_deletion_queue: DeletionQueue,
    allocator: vk_mem::Allocator,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MeshPushConstants {
    data: Vec4,
    render_matrix: Mat4,
}

struct Queues {
    graphics: vk::Queue,
    graphics_family: u32,
    present: vk::Queue,
    present_family: u32,
}

impl VulkanApp {
    pub unsafe fn new(window: &Window) -> VulkanApp {
        let main_deletion_queue = DeletionQueue {
            deletors: VecDeque::new(),
        };

        // init vulkan stuff

        let entry = ash::Entry::new().unwrap();

        let app_info = vk::ApplicationInfo::builder().api_version(vk::make_api_version(1, 1, 0, 0));

        let extension_names = platforms::required_extension_names();

        let validation_layer: CString = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let enabled_layer_names = [validation_layer.as_ptr()];

        let mut debug_create_info = debug_utils_messenger_create_info();
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&enabled_layer_names)
            .push_next(&mut debug_create_info);

        let instance = entry
            .create_instance(&create_info, None)
            .expect("failed to create vk instance");

        //debug messenger
        let debug_utils_fn = DebugUtils::new(&entry, &instance);
        let debug_utils_messenger = debug_utils_fn
            .create_debug_utils_messenger(&debug_utils_messenger_create_info(), None)
            .unwrap();

        //surface
        let surface_fn = Surface::new(&entry, &instance);
        let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();

        ///////
        //creating device
        //////

        let (physical_device, graphics_queue_family, present_queue_family) =
            select_physical_device_and_queue_families(&instance, &surface_fn, surface).unwrap();

        let queue_priorities = [1.0];
        let queue_create_infos = [
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_family)
                .queue_priorities(&queue_priorities)
                .build(),
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(present_queue_family)
                .queue_priorities(&queue_priorities)
                .build(),
        ];
        let queue_create_infos = if graphics_queue_family == present_queue_family {
            &queue_create_infos[0..1]
        } else {
            &queue_create_infos
        };

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let enabled_layer_names = [validation_layer.as_ptr()];

        let enabled_extension_names = [Swapchain::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos)
            .enabled_extension_names(&enabled_extension_names)
            .enabled_layer_names(&enabled_layer_names);
        let device = instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap();

        let queues = Queues {
            graphics: device.get_device_queue(graphics_queue_family, 0),
            graphics_family: graphics_queue_family,
            present: device.get_device_queue(present_queue_family, 0),
            present_family: present_queue_family,
        };

        //printing device name
        //println!("{:?}",instance.get_physical_device_properties(stuff.0));

        /*****
         * swapchain creation
         */

        let surface_capabilities = surface_fn
            .get_physical_device_surface_capabilities(physical_device, surface)
            .unwrap();

        let max_image_count = match surface_capabilities.max_image_count {
            0 => u32::MAX,
            x => x,
        };
        let min_image_count = (surface_capabilities.min_image_count + 1).min(max_image_count);

        let transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let queue_families = [queues.graphics_family, queues.present_family];
        let (image_sharing_mode, queue_families) =
            if queues.graphics_family == queues.present_family {
                (vk::SharingMode::EXCLUSIVE, &queue_families[..1])
            } else {
                (vk::SharingMode::CONCURRENT, &queue_families[..])
            };

        let surface_format = SurfaceFormatKHR::builder()
            .format(Format::B8G8R8A8_SRGB)
            .color_space(ColorSpaceKHR::SRGB_NONLINEAR)
            .build();

        let swapchain_create_info = SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(Extent2D {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            })
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_families)
            .pre_transform(transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(PresentModeKHR::FIFO)
            .clipped(true)
            .build();

        let swapchain_loader = Swapchain::new(&instance, &device);
        let swapchain = swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap();

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        let mut swapchain_image_views: Vec<ImageView> = Vec::with_capacity(swapchain_images.len());

        for image in &swapchain_images {
            let componnent_mapping = ComponentMapping::builder()
                .r(ComponentSwizzle::IDENTITY)
                .g(ComponentSwizzle::IDENTITY)
                .b(ComponentSwizzle::IDENTITY)
                .a(ComponentSwizzle::IDENTITY)
                .build();

            let subresource_range = ImageSubresourceRange::builder()
                .aspect_mask(ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let create_info = ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(componnent_mapping)
                .subresource_range(subresource_range)
                .build();

            let image_view = device.create_image_view(&create_info, None).unwrap();

            swapchain_image_views.push(image_view);
        }

        /*
        //command stuff
         */

        let command_pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(queues.graphics_family)
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();

        let command_pool = device
            .create_command_pool(&command_pool_info, None)
            .unwrap();

        let cmd_buffer_allocate_info = CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(CommandBufferLevel::PRIMARY)
            .build();

        let main_cmd_buffers = device
            .allocate_command_buffers(&cmd_buffer_allocate_info)
            .unwrap();
        let main_cmd_buffer = main_cmd_buffers[0];

        /*

        Render pass and frame buffers

        */

        //color attachment
        let attachment_desc = AttachmentDescription::builder()
            .format(surface_format.format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build();

        let attachment_descs = [attachment_desc];

        let color_attachment_ref = AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let color_attachment_refs = [color_attachment_ref];

        //subpass
        let subpass_desc = SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .build();

        let subpass_descs = [subpass_desc];

        //creating render pass

        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .build();

        let render_pass = device.create_render_pass(&render_pass_info, None).unwrap();

        //frame buffers
        let mut framebuffers: Vec<Framebuffer> = Vec::with_capacity(swapchain_images.len());
        for swapchain_image in &swapchain_image_views {
            let fb_info = FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .width(WINDOW_WIDTH)
                .height(WINDOW_HEIGHT)
                .layers(1)
                .attachments(&[*swapchain_image])
                .build();

            let fb = device.create_framebuffer(&fb_info, None).unwrap();
            framebuffers.push(fb);
        }

        //initializing semaphores and fence

        let fence_create_info = FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED)
            .build();

        let render_fence = device.create_fence(&fence_create_info, None).unwrap();

        let semaphore_create_info = SemaphoreCreateInfo::default();
        let present_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();
        let render_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();

        let window_extent = Extent2D {
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
        };

        let (
            triangle_pipeline_layout,
            mesh_pipeline_layout,
            triangle_pipeline,
            red_triangle_pipeline,
            mesh_pipeline,
        ) = init_pipelines(&device, window_extent, render_pass);

        //allocator

        // Create the buffer (GPU only, 16KiB in this example)
        // let create_info = vk_mem::AllocationCreateInfo {
        //     usage: vk_mem::MemoryUsage::GpuOnly,
        //     ..Default::default()
        // };

        let allocator_info = vk_mem::AllocatorCreateInfo {
            physical_device: physical_device,
            device: device.clone(),
            instance: instance.clone(),
            ..Default::default()
        };

        let allocator = vk_mem::Allocator::new(&allocator_info).unwrap();

        //loading mesh

        // vertices
        let mesh_vertices = vec![
            Vertex {
                position: vec3(1., 1., 0.),
                color: vec3(0., 1., 0.),
                normal: Vec3::default(),
            },
            Vertex {
                position: vec3(-1., 1., 0.),
                color: vec3(0., 1., 0.),
                normal: Vec3::default(),
            },
            Vertex {
                position: vec3(0., -1., 0.),
                color: vec3(0., 1., 0.),
                normal: Vec3::default(),
            },
        ];

        //creating vertex buffer

        let (buffer, allocation, _allocation_info) = allocator
            .create_buffer(
                &BufferCreateInfo::builder()
                    .size((mesh_vertices.len() * mem::size_of::<Vertex>()) as u64)
                    .usage(BufferUsageFlags::VERTEX_BUFFER)
                    .build(),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::CpuToGpu,
                    ..Default::default()
                },
            )
            .unwrap();

        let triangle_mesh = Mesh {
            vertices: mesh_vertices,
            vertex_buffer: AllocatedBuffer {
                buffer: buffer,
                allocation: allocation,
            },
        };

        //copy vertex data to gpu memory
        let data: *mut u8 = allocator.map_memory(&allocation).unwrap();

        std::ptr::copy_nonoverlapping(
            triangle_mesh.vertices.as_ptr().cast(),
            data,
            std::mem::size_of::<Vertex>() * triangle_mesh.vertices.len(),
        );

        allocator.unmap_memory(&allocation);

        let mut app = VulkanApp {
            entry,
            instance,
            debug_utils_fn,
            debug_utils_messenger,
            surface_fn,
            surface,
            surface_format,
            device,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            command_pool,
            main_cmd_buffer,
            render_pass,
            framebuffers,
            render_fence,
            present_semaphore,
            render_semaphore,
            frame_number: 0,
            queues,

            triangle_pipeline_layout,
            triangle_pipeline,
            red_triangle_pipeline,
            selected_shader: 0,
            main_deletion_queue,
            allocator,
            triangle_mesh,
            mesh_pipeline,
            mesh_pipeline_layout,
        };

        app.main_deletion_queue.push_function(|app| {
            app.swapchain_loader.destroy_swapchain(app.swapchain, None);
            app.device.destroy_render_pass(app.render_pass, None);
            for fb in &app.framebuffers {
                app.device.destroy_framebuffer(*fb, None);
            }
            for image_view in &app.swapchain_image_views {
                app.device.destroy_image_view(*image_view, None);
            }

            app.device.destroy_command_pool(app.command_pool, None);
            app.device.destroy_fence(app.render_fence, None);
            app.device.destroy_semaphore(app.present_semaphore, None);
            app.device.destroy_semaphore(app.render_semaphore, None);

            app.allocator.destroy_buffer(
                app.triangle_mesh.vertex_buffer.buffer,
                &app.triangle_mesh.vertex_buffer.allocation,
            );

            app.device.destroy_pipeline(app.red_triangle_pipeline, None);
            app.device.destroy_pipeline(app.triangle_pipeline, None);
            app.device.destroy_pipeline(app.mesh_pipeline, None);
            app.device
                .destroy_pipeline_layout(app.triangle_pipeline_layout, None);
            app.device
                .destroy_pipeline_layout(app.mesh_pipeline_layout, None);
        });

        app
    }

    fn print_available_extensions(&self) {
        let extensions = self
            .entry
            .enumerate_instance_extension_properties()
            .unwrap();
        println!("available extensions:");
        for extension in &extensions {
            println!("{:?}", extension);
        }
    }

    unsafe fn draw_frame(&mut self) {
        self.device
            .wait_for_fences(&[self.render_fence], true, 1000000000)
            .unwrap();
        self.device.reset_fences(&[self.render_fence]).unwrap();

        let swapchain_index = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain,
                1000000000,
                self.present_semaphore,
                Fence::null(),
            )
            .unwrap();

        self.device
            .reset_command_buffer(self.main_cmd_buffer, CommandBufferResetFlags::empty())
            .unwrap();

        let cmd_begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        self.device
            .begin_command_buffer(self.main_cmd_buffer, &cmd_begin_info)
            .unwrap();

        let flash = f32::abs((self.frame_number as f32 / 120.).sin());
        let mut clear_value = ClearValue::default();
        clear_value.color = vk::ClearColorValue {
            float32: [0., 0., flash, 1.],
        };

        let rp_info = RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .render_area(Rect2D {
                offset: Offset2D::default(),
                extent: Extent2D {
                    width: WINDOW_WIDTH,
                    height: WINDOW_HEIGHT,
                },
            })
            .framebuffer(self.framebuffers[swapchain_index.0 as usize])
            .clear_values(&[clear_value])
            .build();

        self.device
            .cmd_begin_render_pass(self.main_cmd_buffer, &rp_info, SubpassContents::INLINE);

        //render stuff..

        if self.selected_shader == 0 {
            self.device.cmd_bind_pipeline(
                self.main_cmd_buffer,
                PipelineBindPoint::GRAPHICS,
                self.mesh_pipeline,
            );
        } else {
            self.device.cmd_bind_pipeline(
                self.main_cmd_buffer,
                PipelineBindPoint::GRAPHICS,
                self.red_triangle_pipeline,
            );
        }

        //bind the mesh vertex buffer with offset 0
        let offset: DeviceSize = 0;
        self.device.cmd_bind_vertex_buffers(
            self.main_cmd_buffer,
            0,
            &[self.triangle_mesh.vertex_buffer.buffer],
            &[offset],
        );

        //make a model view matrix for rendering the object

        //camera position
        let cam_pos = vec3(0., 0., -2.);
        let view = nalgebra_glm::translate(&nalgebra_glm::identity(), &cam_pos);
        //camera projection
        // let projection = nalgebra_glm::perspective(1700. / 900., 1.22173, 0.1, 200.);
        let projection = nalgebra_glm::perspective(WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32, nalgebra_glm::radians(&vec1(90.))[0], 0.1, 200.);
        // model rotations
        let model = nalgebra_glm::rotate::<f32>(&nalgebra_glm::identity(), nalgebra_glm::radians(&vec1(self.frame_number as f32 * 0.4))[0], &vec3(0.,1.,0.));

        //calculate final mesh matrix
        let mesh_matrix : Mat4 = projection * view * model;

        let constants = MeshPushConstants { data: vec4(0., 0., 0., 0.), render_matrix: mesh_matrix };

        //upload the matrix to the GPU via push constants
        self.device.cmd_push_constants(self.main_cmd_buffer, self.mesh_pipeline_layout, ShaderStageFlags::VERTEX, 0, bytemuck::cast_slice(&[constants]));

        self.device.cmd_draw(
            self.main_cmd_buffer,
            self.triangle_mesh.vertices.len() as u32,
            1,
            0,
            0,
        );

        //finalize the render pass
        self.device.cmd_end_render_pass(self.main_cmd_buffer);
        self.device
            .end_command_buffer(self.main_cmd_buffer)
            .unwrap();

        let vk_submit_info = SubmitInfo::builder()
            .wait_dst_stage_mask(&[PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .wait_semaphores(&[self.present_semaphore])
            .signal_semaphores(&[self.render_semaphore])
            .command_buffers(&[self.main_cmd_buffer])
            .build();

        self.device
            .queue_submit(self.queues.graphics, &[vk_submit_info], self.render_fence)
            .unwrap();

        let present_info = PresentInfoKHR::builder()
            .swapchains(&[self.swapchain])
            .wait_semaphores(&[self.render_semaphore])
            .image_indices(&[swapchain_index.0])
            .build();

        self.swapchain_loader
            .queue_present(self.queues.graphics, &present_info)
            .unwrap();
        self.frame_number += 1;
    }

    pub fn main_loop(mut self, mut event_loop: EventLoop<()>, window: Window) {
        event_loop.run_return(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            *control_flow = ControlFlow::Exit
                        }
                        (Some(VirtualKeyCode::Space), ElementState::Pressed) => {
                            self.selected_shader += 1;
                            if self.selected_shader > 1 {
                                self.selected_shader = 0;
                            }
                        }
                        _ => {}
                    },
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => unsafe {
                self.draw_frame();
            },
            _ => (),
        })
    }

    pub unsafe fn set_name<H: vk::Handle>(&self, object: H, name: &str) -> VkResult<()> {
        let name = CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(H::TYPE)
            .object_handle(object.as_raw())
            .object_name(&name);
        self.debug_utils_fn
            .debug_utils_set_object_name(self.device.handle(), &name_info)
    }
}

//returns (vkphysicaldevice, graphics queue family, present queue family)
unsafe fn select_physical_device_and_queue_families(
    instance: &ash::Instance,
    surface_fn: &Surface,
    surface: vk::SurfaceKHR,
) -> Option<(vk::PhysicalDevice, u32, u32)> {
    //selecting physical device and queue famillies
    for physical_device in instance.enumerate_physical_devices().unwrap() {
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let graphics_queue = queue_families
            .iter()
            .position(|info| info.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        for (present_queue, _) in queue_families.iter().enumerate() {
            let supports_surface = surface_fn
                .get_physical_device_surface_support(physical_device, present_queue as _, surface)
                .unwrap();
            if supports_surface {
                return graphics_queue.map(|graphics_queue| {
                    (physical_device, graphics_queue as _, present_queue as _)
                });
            }
        }
    }
    None
}

unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> u32 {
    let message_severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => log::Level::Error,
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    if let Ok(message) = message.to_str() {
        log::log!(message_severity, "{:?}: {}", message_types, message);
    } else {
        log::log!(message_severity, "{:?}: {:?}", message_types, message);
    }
    vk::FALSE
}

fn debug_utils_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(debug_utils_callback))
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.render_fence], true, 1000000000)
                .unwrap();

            //NOTE(lucypero): deletion queue in app is empty after this!!!
            let mut main_deletion_queue = mem::replace(
                &mut self.main_deletion_queue,
                DeletionQueue {
                    deletors: VecDeque::new(),
                },
            );

            main_deletion_queue.flush(&self);
            self.allocator.destroy();

            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.debug_utils_fn
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn load_shader_module(
    device: &ash::Device,
    file_path: String,
) -> Result<ShaderModule, std::io::Error> {
    let mut file = File::open(file_path)?;
    let file = ash::util::read_spv(&mut file)?;

    let shader_create_info = ShaderModuleCreateInfo::builder().code(&file[..]).build();

    let shader_module = device
        .create_shader_module(&shader_create_info, None)
        .unwrap();
    Ok(shader_module)
}

unsafe fn init_pipelines(
    device: &ash::Device,
    window_extent: Extent2D,
    render_pass: RenderPass,
) -> (PipelineLayout, PipelineLayout, Pipeline, Pipeline, Pipeline) {
    let triangle_vertex_shader =
        load_shader_module(device, "shaders/colored_triangle.vert.spv".to_string()).unwrap();

    let triangle_frag_shader =
        load_shader_module(device, "shaders/colored_triangle.frag.spv".to_string()).unwrap();

    let red_triangle_vertex_shader =
        load_shader_module(device, "shaders/triangle.vert.spv".to_string()).unwrap();

    let red_triangle_frag_shader =
        load_shader_module(device, "shaders/triangle.frag.spv".to_string()).unwrap();

    let mesh_vert_shader =
        load_shader_module(device, "shaders/tri_mesh.vert.spv".to_string()).unwrap();

    //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
    let name = CString::new("main").unwrap();

    let shader_stages: Vec<PipelineShaderStageCreateInfo> = vec![
        get_pipeline_shader_stage_create_info(
            ShaderStageFlags::VERTEX,
            triangle_vertex_shader,
            &name,
        ),
        get_pipeline_shader_stage_create_info(
            ShaderStageFlags::FRAGMENT,
            triangle_frag_shader,
            &name,
        ),
    ];

    //vertex input controls how to read vertices from vertex buffers. We aren't using it yet
    let vertex_input_info = get_vertex_input_state_create_info();

    //input assembly is the configuration for drawing triangle lists, strips, or individual points.
    //we are just going to draw triangle list
    let input_assembly = get_input_assembly_create_info(PrimitiveTopology::TRIANGLE_LIST);

    //build viewport and scissor from the swapchain extents
    let viewport = Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(window_extent.width as f32)
        .height(window_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build();

    let scissor: Rect2D = Rect2D::builder().extent(window_extent).build();

    //configure the rasterizer to draw filled triangles
    let rasterizer = get_rasterization_state_create_info(PolygonMode::FILL);

    //we don't use multisampling, so just run the default one
    let multisampling = get_multisampling_state_create_info();

    //a single blend attachment with no blending and writing to RGBA
    let color_blend_attachment = get_color_blend_attachment_state();

    //build the pipeline layout that controls the inputs/outputs of the shader
    //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
    let pipeline_layout_info = get_pipeline_layout_create_info();
    let pipeline_layout = device
        .create_pipeline_layout(&pipeline_layout_info, None)
        .expect("could not create pipeline layout");

    //finally build the pipeline
    let mut pipeline_builder = PipelineBuilder {
        shader_stages,
        vertex_input_info,
        input_assembly,
        viewport,
        scissor,
        rasterizer,
        color_blend_attachment,
        multisampling,
        pipeline_layout,
    };

    let triangle_pipeline = pipeline_builder.build_pipeline(device, render_pass);

    //making red triangle pipeline

    pipeline_builder.shader_stages.clear();

    pipeline_builder
        .shader_stages
        .push(get_pipeline_shader_stage_create_info(
            ShaderStageFlags::VERTEX,
            red_triangle_vertex_shader,
            &name,
        ));
    pipeline_builder
        .shader_stages
        .push(get_pipeline_shader_stage_create_info(
            ShaderStageFlags::FRAGMENT,
            red_triangle_frag_shader,
            &name,
        ));

    let red_triangle_pipeline = pipeline_builder.build_pipeline(device, render_pass);

    // making mesh triangle pipeline

    //push constants
    let push_constant = PushConstantRange::builder()
        .offset(0)
        .size(std::mem::size_of::<MeshPushConstants>() as u32)
        .stage_flags(ShaderStageFlags::VERTEX)
        .build();

    let mesh_pipeline_layout_info = PipelineLayoutCreateInfo::builder()
        .push_constant_ranges(&[push_constant])
        .build();

    let mesh_pipeline_layout = device
        .create_pipeline_layout(&mesh_pipeline_layout_info, None)
        .unwrap();

    let vertex_description = Vertex::get_vertex_description();

    //connect the pipeline builder vertex input info to the one we get from Vertex
    pipeline_builder.vertex_input_info = PipelineVertexInputStateCreateInfo::builder()
        .vertex_attribute_descriptions(&vertex_description.attributes)
        .vertex_binding_descriptions(&vertex_description.bindings)
        .build();

    pipeline_builder.shader_stages.clear();

    pipeline_builder
        .shader_stages
        .push(get_pipeline_shader_stage_create_info(
            ShaderStageFlags::VERTEX,
            mesh_vert_shader,
            &name,
        ));
    pipeline_builder
        .shader_stages
        .push(get_pipeline_shader_stage_create_info(
            ShaderStageFlags::FRAGMENT,
            triangle_frag_shader,
            &name,
        ));

    pipeline_builder.pipeline_layout = mesh_pipeline_layout;
    let mesh_pipeline = pipeline_builder.build_pipeline(device, render_pass);

    //destroy shader modules
    device.destroy_shader_module(triangle_frag_shader, None);
    device.destroy_shader_module(triangle_vertex_shader, None);
    device.destroy_shader_module(red_triangle_frag_shader, None);
    device.destroy_shader_module(red_triangle_vertex_shader, None);
    device.destroy_shader_module(mesh_vert_shader, None);

    (
        pipeline_layout,
        mesh_pipeline_layout,
        triangle_pipeline,
        red_triangle_pipeline,
        mesh_pipeline,
    )
}

fn get_pipeline_shader_stage_create_info(
    stage: ShaderStageFlags,
    shader_module: ShaderModule,
    name: &CString,
) -> PipelineShaderStageCreateInfo {
    let info = PipelineShaderStageCreateInfo::builder()
        .stage(stage)
        .module(shader_module)
        .name(name)
        .build();

    info
}

fn get_vertex_input_state_create_info() -> PipelineVertexInputStateCreateInfo {
    let info = PipelineVertexInputStateCreateInfo::builder().build();

    info
}

fn get_input_assembly_create_info(
    topology: PrimitiveTopology,
) -> PipelineInputAssemblyStateCreateInfo {
    let info = PipelineInputAssemblyStateCreateInfo::builder()
        .topology(topology)
        .primitive_restart_enable(false)
        .build();
    info
}

fn get_rasterization_state_create_info(
    polygon_mode: PolygonMode,
) -> PipelineRasterizationStateCreateInfo {
    let info = PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .line_width(1.0)
        .cull_mode(CullModeFlags::NONE)
        .front_face(FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
        .build();
    info
}

fn get_multisampling_state_create_info() -> PipelineMultisampleStateCreateInfo {
    let info = PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .sample_mask(&[])
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();
    info
}

fn get_color_blend_attachment_state() -> PipelineColorBlendAttachmentState {
    let info = PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            ColorComponentFlags::R
                | ColorComponentFlags::G
                | ColorComponentFlags::B
                | ColorComponentFlags::A,
        )
        .blend_enable(false)
        .build();
    info
}

fn get_pipeline_layout_create_info() -> PipelineLayoutCreateInfo {
    let info = PipelineLayoutCreateInfo::builder().build();
    info
}

fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    winit::window::WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
}

fn main() {
    env_logger::init();
    log::warn!("foobar");

    let event_loop = EventLoop::new();
    let window = init_window(&event_loop);

    unsafe {
        let vulkan_app: VulkanApp = VulkanApp::new(&window);
        vulkan_app.main_loop(event_loop, window);
    }
}
