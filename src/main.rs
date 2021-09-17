use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use std::time::Duration;
use sdl2::rect::Point;
use sdl2::render::Canvas;
use sdl2::video::Window;
use image::{Rgb, ImageBuffer};

pub const WIDTH: u32 = 1000;
pub const HEIGHT: u32 = 1000;

fn draw(texture_canvas: &mut Canvas<Window>, img: &ImageBuffer<Rgb<f64>, Vec<f64>>) {

  for (x, y, pixel) in img.enumerate_pixels() {
    texture_canvas.set_draw_color(Color::RGB((pixel[0] * 255.0) as u8,
                                             (pixel[1] * 255.0) as u8,
                                             (pixel[2] * 255.0) as u8));
    texture_canvas
        .draw_point(Point::new(x as i32, y as i32))
        .expect("could not draw point");
  }
}

fn generate_image () -> ImageBuffer<Rgb<f64>, Vec<f64>> {
  let mut img = ImageBuffer::<Rgb<f64>, Vec<f64>>::new(WIDTH, HEIGHT);

  let x_from: f64 = -2.0;
  let x_to: f64 = 0.47;
  let x_range = x_from.abs() + x_to.abs();
  let y_from: f64 = -1.12;
  let y_to: f64 = 1.12;
  let y_range = y_from.abs() + y_to.abs();

  for (x, y, pixel) in img.enumerate_pixels_mut() {
    let x_percent = x as f64 / WIDTH as f64;
    let x0 = x_percent * x_range + x_from;
    let y_percent = y as f64 / HEIGHT as f64;
    let y0 = y_percent * y_range + y_from;

    let mut iteration = 0;
    let max_iteration = 10000;

    let mut xres = 0.0;
    let mut yres = 0.0;

    while xres*xres + yres * yres <= 2.0*2.0 && iteration < max_iteration {
      let xtemp = xres*xres - yres*yres + x0;
      yres = 2.0*xres*yres + y0;
      xres = xtemp;
      iteration = iteration + 1;
    }

    let color = iteration as f64 / max_iteration as f64;

    if iteration == max_iteration {
      *pixel = Rgb::<f64>([0.0, 0.0, 0.0]);  /* In the set. Assign black. */
    } else if iteration < max_iteration / 64 {
      let r = color * 32.0;
      *pixel = Rgb::<f64>([r, 0.0, 0.0]);
    } else if iteration < max_iteration / 32 {
      let r = (((iteration - max_iteration/64) as f64 * 128.0/256.0) / max_iteration as f64 /32.0) + 128.0/256.0;
      *pixel = Rgb::<f64>([r, 0.0, 0.0]);
    } else if iteration < max_iteration / 16 {
      let r = (((iteration - max_iteration/32) as f64 * 62.0/256.0) / max_iteration as f64 /32.0) + 193.0/256.0;
      *pixel = Rgb::<f64>([r, 0.0, 0.0]);
    } else if iteration < max_iteration / 8 {
      let g = (((iteration - max_iteration/16) as f64 * 62.0/256.0) / max_iteration as f64 /16.0) + 1.0/256.0;
      *pixel = Rgb::<f64>([1.0, g, 0.0]);
    } else if iteration < max_iteration / 4 {
      let g = (((iteration - max_iteration/8) as f64 * 63.0/256.0) / max_iteration as f64 /8.0) + 64.0/256.0;
      *pixel = Rgb::<f64>([1.0, g, 0.0]);
    } else if iteration < max_iteration / 2 {
      let g = (((iteration - max_iteration/4) as f64 * 63.0/256.0) / max_iteration as f64 /4.0) + 128.0/256.0;
      *pixel = Rgb::<f64>([1.0, g, 0.0]);
    } else if iteration < max_iteration {
      let g = (((iteration - max_iteration/2) as f64 * 63.0/256.0) / max_iteration as f64 /2.0) + 192.0/256.0;
      *pixel = Rgb::<f64>([1.0, g, 0.0]);
    } else {
      *pixel = Rgb::<f64>([1.0, 1.0, 0.0]);
    }
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


