use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::image::{InitFlag};
use std::time::Duration;
use sdl2::rect::{Point};

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;


    let window = video_subsystem
        .window("Mandelbrot", WIDTH, HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_target(None, WIDTH, HEIGHT).map_err(|e| e.to_string())?;

    canvas.with_texture_canvas(&mut texture, |texture_canvas| {
      texture_canvas.set_draw_color(Color::RGB(255, 0, 0));
      texture_canvas.clear();
      for x in 0..WIDTH {
        for y in 0..HEIGHT {
            if (x + y) % 4 == 0 {
                texture_canvas.set_draw_color(Color::RGB(255, 255, 0));
                texture_canvas
                    .draw_point(Point::new(x as i32, y as i32))
                    .expect("could not draw point");
            }
            if (x + y * 2) % 9 == 0 {
                texture_canvas.set_draw_color(Color::RGB(200, 200, 0));
                texture_canvas
                    .draw_point(Point::new(x as i32, y as i32))
                    .expect("could not draw point");
            }
        }
    }

    }).map_err(|e| e.to_string())?;

    canvas.copy(&texture, None, None)?;

    canvas.present();
    let mut event_pump = sdl_context.event_pump()?;

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


