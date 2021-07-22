#![allow(unused_imports, dead_code)]

mod platforms;

use std::ffi::{CString, c_void, CStr};
use std::os::raw::c_char;
use std::str::FromStr;

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::Surface,
        khr::Swapchain
    },
    vk::{
        self, Handle
    },
    Entry,
    version::{
        EntryV1_0, InstanceV1_0
    }
};

use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

// Constants
const WINDOW_TITLE: &'static str = "vulkan-rs";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct VulkanApp {
    entry : Entry,
    instance : ash::Instance,
    debug_utils_fn : DebugUtils,
    debug_utils_messenger : vk::DebugUtilsMessengerEXT,
    surface_fn : Surface,
    surface : vk::SurfaceKHR
}

impl VulkanApp {

    pub unsafe fn new(window: &Window) -> VulkanApp {

        // init vulkan stuff

        let entry = ash::Entry::new().unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_version(1, 1, 0))
        ;

        let extension_names = platforms::required_extension_names();

        let validation_layer : CString = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let enabled_layer_names = [validation_layer.as_ptr()];

        let mut debug_create_info = debug_utils_messenger_create_info();
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&enabled_layer_names)
            .push_next(&mut debug_create_info)
        ;

        let instance = entry.create_instance(&create_info, None)
            .expect("failed to create vk instance");

        //debug messenger
        let debug_utils_fn = DebugUtils::new(&entry, &instance);
        let debug_utils_messenger = debug_utils_fn
            .create_debug_utils_messenger(&debug_utils_messenger_create_info(), None).unwrap();

        //surface
        let surface_fn = Surface::new(&entry, &instance);
        let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();

        //physical device and queue families
        let stuff = select_physical_device_and_queue_families(&instance, &surface_fn, surface).unwrap();

        //printing device name
        println!("{:?}",instance.get_physical_device_properties(stuff.0));

        VulkanApp{entry, instance, debug_utils_fn, debug_utils_messenger, surface_fn, surface}
    }

    
    fn print_available_extensions(&self) {
        let extensions = self.entry.enumerate_instance_extension_properties().unwrap();
        println!("available extensions:");
        for extension in &extensions {
            println!("{:?}", extension);
        }
    }

    fn draw_frame(&mut self) {
        // Drawing will be here
    }

    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }

    pub fn main_loop(mut self, mut event_loop: EventLoop<()>, window: Window) {

        event_loop.run_return(move |event, _, control_flow| {

            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | KeyboardInput { virtual_keycode, state, .. } => {
                                    match (virtual_keycode, state) {
                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                            *control_flow = ControlFlow::Exit
                                        },
                                        | _ => {},
                                    }
                                },
                            }
                        },
                        | _ => {},
                    }
                },
                | Event::MainEventsCleared => {
                    window.request_redraw();
                },
                | Event::RedrawRequested(_window_id) => {
                    self.draw_frame();
                },
                _ => (),
            }

        })
    }
}

unsafe fn select_physical_device_and_queue_families(
    instance: &ash::Instance,
    surface_fn: &Surface,
    surface: vk::SurfaceKHR
    ) -> Option<(vk::PhysicalDevice, u32, u32)> {
        //selecting physical device and queue famillies
        for physical_device in instance.enumerate_physical_devices().unwrap() {
            let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
            let graphics_queue = queue_families.iter().position(|info| info.queue_flags.contains(vk::QueueFlags::GRAPHICS));
            for (present_queue, _) in queue_families.iter().enumerate() {
                let supports_surface = surface_fn.get_physical_device_surface_support(physical_device, present_queue as _, surface).unwrap();
                if supports_surface {
                    return graphics_queue.map(|graphics_queue| {
                        (physical_device, graphics_queue as _, present_queue as _)
                    })
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
            self.debug_utils_fn.destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);
    
    unsafe {
        let vulkan_app = VulkanApp::new(&window);
        vulkan_app.main_loop(event_loop, window);
    }


}
