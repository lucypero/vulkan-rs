#![allow(unused_imports, dead_code)]

mod platforms;

use std::borrow::Borrow;
use std::ffi::{c_void, CStr, CString};
use std::fs::File;
use std::os::raw::c_char;
use std::path::Path;
use std::str::FromStr;

use ash::vk::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, ClearColorValue, ClearValue, ColorSpaceKHR, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, CommandPoolResetFlags, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, Extent2D, Fence, FenceCreateFlags, FenceCreateInfo, Format, Framebuffer, FramebufferCreateInfo, Image, ImageAspectFlags, ImageLayout, ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, Offset2D, Pipeline, PipelineBindPoint, PipelineColorBlendAttachmentState, PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineVertexInputStateCreateInfo, PresentInfoKHR, PresentModeKHR, Rect2D, RenderPass, RenderPassBeginInfo, RenderPassCreateInfo, SampleCountFlags, Semaphore, SemaphoreCreateInfo, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags, SubmitInfo, SubpassContents, SubpassDescription, SurfaceFormatKHR, SwapchainCreateInfoKHR, SwapchainCreateInfoKHRBuilder, SwapchainKHR, Viewport};
use ash::{
    extensions::{ext::DebugUtils, khr::Surface, khr::Swapchain},
    vk::{self, Handle},
    Entry,
};

use image::ImageFormat;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

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
    fn build_pipeline(device: &ash::Device, render_pass: RenderPass) -> Pipeline {
        panic!();
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

    // "game" state
    frame_number: i64,
}

struct Queues {
    graphics: vk::Queue,
    graphics_family: u32,
    present: vk::Queue,
    present_family: u32,
}

impl VulkanApp {
    pub unsafe fn new(window: &Window) -> VulkanApp {
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

        // //create a command pool for commands submitted to the graphics queue.
        // VkCommandPoolCreateInfo commandPoolInfo = {};
        // commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // commandPoolInfo.pNext = nullptr;

        // //the command pool will be one that can submit graphics commands
        // commandPoolInfo.queueFamilyIndex = _graphicsQueueFamily;
        // //we also want the pool to allow for resetting of individual command buffers
        // commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        // VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

        let command_pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(queues.graphics_family)
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();

        let command_pool = device
            .create_command_pool(&command_pool_info, None)
            .unwrap();

        //command buffer
        // // --- other code ----

        // //allocate the default command buffer that we will use for rendering
        // VkCommandBufferAllocateInfo cmdAllocInfo = {};
        // cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        // cmdAllocInfo.pNext = nullptr;

        // //commands will be made from our _commandPool
        // cmdAllocInfo.commandPool = _commandPool;
        // //we will allocate 1 command buffer
        // cmdAllocInfo.commandBufferCount = 1;
        // // command level is Primary
        // cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        // VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

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

        init_pipelines(&device);

        VulkanApp {
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
        }
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

    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
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
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for fb in &self.framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }
            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }
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

unsafe fn init_pipelines(device: &ash::Device) {
    let triangle_frag_shader =
        load_shader_module(device, "shaders/triangle.frag.spv".to_string()).unwrap();

    let triangle_vertex_shader =
        load_shader_module(device, "shaders/triangle.vert.spv".to_string()).unwrap();

    println!("shaders successfully loaded.");
}

fn get_pipeline_shader_stage_create_info(stage: ShaderStageFlags, shader_module: ShaderModule) -> PipelineShaderStageCreateInfo {

    let name = CString::new("main").unwrap();

    let info = PipelineShaderStageCreateInfo::builder()
    .stage(stage)
    .module(shader_module)
    .name(&name)
    .build();

    info
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);

    unsafe {
        let vulkan_app = VulkanApp::new(&window);
        vulkan_app.main_loop(event_loop, window);
    }
}
