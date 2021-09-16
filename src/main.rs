use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use std::time::Duration;
use sdl2::rect::Point;
use sdl2::render::Canvas;
use sdl2::video::Window;
use image::{Rgb, ImageBuffer};

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

fn draw(texture_canvas: &mut Canvas<Window>, img: &ImageBuffer<Rgb<u16>, Vec<u16>>) {

  for (x, y, pixel) in img.enumerate_pixels() {
    texture_canvas.set_draw_color(Color::RGB(pixel[0] as u8, pixel[1] as u8, pixel[2] as u8));
    texture_canvas
        .draw_point(Point::new(x as i32, y as i32))
        .expect("could not draw point");
  }
}

fn generate_image () -> ImageBuffer<Rgb<u16>, Vec<u16>> {
  let mut img = ImageBuffer::<Rgb<u16>, Vec<u16>>::new(WIDTH, HEIGHT);

  for (x, y, pixel) in img.enumerate_pixels_mut() {
    let x_percent = x as f32 / WIDTH as f32;
    let x_color = x_percent * 255 as f32;
    let x_color_int = x_color as u16;
    let y_percent = y as f32 / HEIGHT as f32;
    let y_color = y_percent * 255 as f32;
    let y_color_int = y_color as u16;

    *pixel = Rgb::<u16>([x_color_int, y_color_int, 0]);
  }

  img
}

pub fn main() -> Result<(), String> {
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


    let mut event_pump = sdl_context.event_pump()?;

    let img = generate_image();

    canvas.with_texture_canvas(&mut texture, | draw_canvs | {
      draw(draw_canvs, &img);
    }).map_err(|e| e.to_string())?;
    canvas.copy(&texture, None, None)?;
    canvas.present();

    'running: loop {
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
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
    }

    Ok(())
}


