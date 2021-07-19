mod platforms;

use std::ffi::CString;

use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::Window;

use ash::{vk, Entry, version::EntryV1_1};
use ash::version::{EntryV1_0, InstanceV1_0};

// Constants
const WINDOW_TITLE: &'static str = "vulkan-rs";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct VulkanApp {
    entry : Entry,
    instance : ash::Instance
}

impl VulkanApp {

    pub unsafe fn new() -> VulkanApp {

        // init vulkan stuff
        let entry = ash::Entry::new().unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_version(1, 1, 0))
        ;

        let extension_names = platforms::required_extension_names();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
        ;

        let instance = entry.create_instance(&create_info, None)
            .expect("failed to create vk instance");
        
        VulkanApp{entry, instance}
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

    pub fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {

        event_loop.run(move |event, _, control_flow| {

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

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}


fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);
    
    unsafe {
        let vulkan_app = VulkanApp::new();
        vulkan_app.main_loop(event_loop, window);
    }

}
