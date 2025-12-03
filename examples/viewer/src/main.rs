use lottie_core::LottiePlayer;
use lottie_data::model::LottieJson;
use lottie_skia::SkiaRenderer;
use skia_safe::{AlphaType, ColorType, ImageInfo};
use softbuffer::{Context, Surface as SoftSurface};
use std::num::NonZeroU32;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const SAMPLE_LOTTIE: &str = r#"
{
  "v": "5.5.2",
  "fr": 60,
  "ip": 0,
  "op": 300,
  "w": 800,
  "h": 600,
  "nm": "Test Animation",
  "layers": [
    {
      "ty": 4,
      "nm": "Rotating Rect",
      "ind": 1,
      "ip": 0, "op": 300, "st": 0,
      "ks": {
        "p": { "a": 0, "k": [400, 300] },
        "s": { "a": 1, "k": [
            { "t": 0, "s": [100, 100], "e": [150, 150] },
            { "t": 100, "s": [150, 150], "e": [100, 100] }
        ]},
        "r": { "a": 1, "k": [
            { "t": 0, "s": [0], "e": [360] }
        ]}
      },
      "shapes": [
        {
          "ty": "gr",
          "it": [
            {
              "ty": "rc",
              "s": { "k": [200, 200] },
              "p": { "k": [0, 0] },
              "r": { "k": 20 }
            },
            {
              "ty": "fl",
              "c": { "k": [0.2, 0.6, 0.9, 1] },
              "o": { "k": 100 }
            },
            {
              "ty": "st",
              "c": { "k": [0.1, 0.1, 0.1, 1] },
              "w": { "k": 10 },
              "o": { "k": 100 }
            },
            {
              "ty": "tm",
              "s": { "a": 1, "k": [ { "t": 0, "s": [0], "e": [0] }, { "t": 100, "s": [0], "e": [50] } ] },
              "e": { "a": 0, "k": 100 }
            }
          ]
        }
      ]
    },
    {
      "ty": 5,
      "nm": "Text Layer",
      "ind": 2,
      "ip": 0, "op": 300, "st": 0,
      "ks": { "p": { "k": [400, 100] } },
      "t": {
        "d": {
            "k": [
                {
                    "t": 0,
                    "s": { "t": "Hello Lottie", "f": "Arial", "s": 64, "j": 2, "fc": [0, 0, 0, 1], "lh": 70 }
                }
            ]
        }
      }
    }
  ]
}
"#;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Lottie Viewer")
        .build(&event_loop)
        .unwrap();

    // Leak window to allow static lifetime for softbuffer context/surface
    let window: &'static winit::window::Window = Box::leak(Box::new(window));

    // Softbuffer init
    let context = Context::new(window).unwrap();
    let mut surface = SoftSurface::new(&context, window).unwrap();

    // Player init
    let mut player = LottiePlayer::new();
    let args: Vec<String> = std::env::args().collect();
    let lottie: LottieJson = if args.len() > 1 {
        let content = std::fs::read_to_string(&args[1]).expect("Failed to read file");
        serde_json::from_str(&content).expect("Failed to parse JSON")
    } else if let Ok(content) = std::fs::read_to_string("examples/viewer/assets/spinner.json") {
        serde_json::from_str(&content).expect("Failed to parse spinner JSON")
    } else {
        serde_json::from_str(SAMPLE_LOTTIE).unwrap()
    };
    player.load(lottie);

    let mut last_frame = Instant::now();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::RedrawRequested => {
                            // Time delta
                            let now = Instant::now();
                            let dt = now.duration_since(last_frame).as_secs_f32();
                            last_frame = now;

                            player.advance(dt);

                            // Resize softbuffer if needed
                            let size = window.inner_size();
                            if let Some(w) = NonZeroU32::new(size.width) {
                                if let Some(h) = NonZeroU32::new(size.height) {
                                    surface.resize(w, h).unwrap();
                                }
                            }

                            // Render
                            let mut buffer = surface.buffer_mut().unwrap();
                            let width = size.width as i32;
                            let height = size.height as i32;

                            if width > 0 && height > 0 {
                                // Create Skia Surface
                                let info = ImageInfo::new(
                                    (width, height),
                                    ColorType::BGRA8888,
                                    AlphaType::Premul,
                                    Some(skia_safe::ColorSpace::new_srgb()),
                                );

                                let ptr = buffer.as_mut_ptr() as *mut u8;
                                let len = (width * height * 4) as usize;
                                let row_bytes = width as usize * 4;

                                // Safety: Buffer is locked and valid for this scope.
                                let pixels_u8 = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

                                let skia_surface = skia_safe::surfaces::wrap_pixels(
                                    &info, pixels_u8, row_bytes, None,
                                );

                                if let Some(mut skia_surface) = skia_surface {
                                    let canvas = skia_surface.canvas();
                                    canvas.clear(skia_safe::Color::WHITE);

                                    let tree = player.render_tree();

                                    SkiaRenderer::draw(
                                        canvas,
                                        &tree,
                                        skia_safe::Rect::from_wh(width as f32, height as f32),
                                        1.0,
                                    );
                                }
                            }

                            buffer.present().unwrap();
                        }
                        _ => (),
                    }
                }
                _ => (),
            }
        })
        .unwrap();
}
