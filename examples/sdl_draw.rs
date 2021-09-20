use std::time::Instant;
use sdl2::{event::Event};
use sdl2::keyboard::Keycode;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rect::{Point, Rect};
use palette::{FromColor, Hsv, Srgb};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

pub fn main() -> Result<(), String> {
  let start_time = Instant::now();

  let sdl_context = sdl2::init()?;
  let video_subsystem = sdl_context.video()?;


  let window = video_subsystem
      .window("Mandelbrot", WIDTH, HEIGHT)
      .position_centered()
      .opengl()
      .build()
      .map_err(|e| e.to_string())?;

  let mut canvas = window
      .into_canvas()
      .target_texture()
      .present_vsync()
      .build()
      .map_err(|e| e.to_string())?;

  let texture_creator = canvas.texture_creator();

  let mut texture = texture_creator
  .create_texture_streaming(PixelFormatEnum::RGB24, WIDTH, HEIGHT).map_err(|e| e.to_string())?;

  let mut pixels: Vec<u8> = vec![0; (WIDTH * HEIGHT * 3) as usize];

  let rect: Rect = Rect::new(0,0, WIDTH, HEIGHT);

  let mut event_pump = sdl_context.event_pump()?;
  'running: loop {
    let frame_start_time = Instant::now();
    for event in event_pump.poll_iter() {
        match event {
            Event::Quit { .. }
            | Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => break 'running,
            _ => {}
        }
    }

    let elapsed_ms = start_time.elapsed().as_millis() as f32;
    let hue =  elapsed_ms / 10.0 % 360.0;

    let hsv = Hsv::new(hue, 1.0, 1.0);
    let rgb = Srgb::from_color(hsv);


    if true {
      for x in 0..(WIDTH) {
        for y in 0..(HEIGHT) {
          let base = (((y * WIDTH) + x) * 3) as usize;
          pixels[base] = (rgb.red * 255.0) as u8;
          pixels[base + 1] = (rgb.green * 255.0) as u8;
          pixels[base + 2] = (rgb.blue * 255.0) as u8;
        }
      }

      texture.update(rect, &pixels, (3*WIDTH) as usize).map_err(|e| e.to_string())?;
      canvas.copy(&texture, None, None)?;
    } else {
      let color = Color::RGB((rgb.red * 255.0) as u8,
      (rgb.green * 255.0) as u8,
      (rgb.blue * 255.0) as u8);
      canvas.set_draw_color(color);

      for x in 0..(WIDTH) {
        for y in 0..(HEIGHT) {
          canvas.draw_point(Point::new(x as i32, y as i32))
          .expect("could not draw point");
        }
      }

    }

    canvas.present();
    let frame_time_ms = frame_start_time.elapsed().as_millis();
    let fps = 1000.0 / frame_time_ms as f64;
    println!("Frame Time {}ms | {:.2} fps", frame_start_time.elapsed().as_millis(), fps);
  }

  Ok(())
}
