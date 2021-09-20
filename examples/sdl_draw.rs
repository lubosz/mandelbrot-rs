use std::time::Instant;

use palette::rgb::Rgb;
use sdl2::{Sdl, event::Event};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::{Canvas, Texture};
use sdl2::video::Window;
use palette::{FromColor, Hsl, Hsv, Lab, LinSrgb, RgbHue, Srgb};
//use image::{ImageBuffer, Pixel, Rgb};

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
      .create_texture_target(None, WIDTH, HEIGHT)
      .map_err(|e| e.to_string())?;


  //let mut pixel = Vec::<u8>::new();

  //texture.update((), array, WIDTH * );

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

    let hue = start_time.elapsed().as_millis() as f32 % 360.0;

    let hsv = Hsv::new(hue, 1.0, 1.0);
    let rgb = Srgb::from_color(hsv);

    for x in 0..(WIDTH) {
      for y in 0..(HEIGHT) {
        let color = Color::RGB((rgb.red * 255.0) as u8,
        (rgb.green * 255.0) as u8,
        (rgb.blue * 255.0) as u8);
        canvas.set_draw_color(color);
        canvas.draw_point(Point::new(x as i32, y as i32))
        .expect("could not draw point");
      }
    }
    canvas.present();
    //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
    let frame_time_ms = frame_start_time.elapsed().as_millis();
    let fps = 1000.0 / frame_time_ms as f64;
    println!("Frame Time {}ms | {:.2} fps", frame_start_time.elapsed().as_millis(), fps);
  }

  Ok(())
}
