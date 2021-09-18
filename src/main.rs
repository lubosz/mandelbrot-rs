use num_complex::Complex;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::Canvas;
use sdl2::video::Window;
use image::{Rgb, ImageBuffer};
use vecmath::{Vector2, vec2_add, vec2_scale, vec2_sub};
use std::time::{Duration, Instant};

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

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

fn map_color(iteration: u32, max_iteration: u32) -> Rgb::<f64> {
  let color = iteration as f64 / max_iteration as f64;

  if iteration == max_iteration {
    Rgb::<f64>([0.0, 0.0, 0.0])  /* In the set. Assign black. */
  } else if iteration < max_iteration / 64 {
    let r = color * 32.0;
    Rgb::<f64>([r, 0.0, 0.0])
  } else if iteration < max_iteration / 32 {
    let r = (((iteration - max_iteration/64) as f64 * 128.0/256.0) / max_iteration as f64 /32.0) + 128.0/256.0;
    Rgb::<f64>([r, 0.0, 0.0])
  } else if iteration < max_iteration / 16 {
    let r = (((iteration - max_iteration/32) as f64 * 62.0/256.0) / max_iteration as f64 /32.0) + 193.0/256.0;
    Rgb::<f64>([r, 0.0, 0.0])
  } else if iteration < max_iteration / 8 {
    let g = (((iteration - max_iteration/16) as f64 * 62.0/256.0) / max_iteration as f64 /16.0) + 1.0/256.0;
    Rgb::<f64>([1.0, g, 0.0])
  } else if iteration < max_iteration / 4 {
    let g = (((iteration - max_iteration/8) as f64 * 63.0/256.0) / max_iteration as f64 /8.0) + 64.0/256.0;
    Rgb::<f64>([1.0, g, 0.0])
  } else if iteration < max_iteration / 2 {
    let g = (((iteration - max_iteration/4) as f64 * 63.0/256.0) / max_iteration as f64 /4.0) + 128.0/256.0;
    Rgb::<f64>([1.0, g, 0.0])
  } else if iteration < max_iteration {
    let g = (((iteration - max_iteration/2) as f64 * 63.0/256.0) / max_iteration as f64 /2.0) + 192.0/256.0;
    Rgb::<f64>([1.0, g, 0.0])
  } else {
    Rgb::<f64>([1.0, 1.0, 0.0])
  }
}

fn iterate(max_iteration: u32, pos: Vector2::<f64>) -> u32 {
  let mut i = 0;
  let mut x = 0.0;
  let mut y = 0.0;

  while x*x + y*y <= 2.0*2.0 && i < max_iteration {
    let ytemp = y*y - x*x + pos[1];
    x = 2.0 * x*y + pos[0];
    y = ytemp;
    i += 1;
  }
  i
}

fn iterate_opt(max_iteration: u32, pos: Vector2::<f64>) -> u32 {
  let mut i = 0;
  let mut x2 = 0.0;
  let mut y2 = 0.0;

  let mut x = 0.0;
  let mut y = 0.0;

  while x2 + y2 <= 4.0 && i < max_iteration {
    x = 2.0 * y * x + pos[0];
    y = y2 - x2 + pos[1];
    x2 = x * x;
    y2 = y * y;
    i += 1;
  }
  i
}

fn iterate_complex(max_iteration: u32, pos: Vector2::<f64>) -> u32 {
  let mut i = 0;
  let c: Complex<f64> = Complex::new(pos[1], pos[0]);
  let mut z: Complex<f64> = Complex::new(0.0, 0.0);

  while z.norm_sqr() <= 2.0*2.0 && i < max_iteration {
    z = z.powu(2) + c;
    i += 1;
  }
  i
}

struct Config {
  center: Vector2::<f64>,
  density: f64
}

fn origin_from_screen_size(config: &Config, w: u32, h: u32) -> Vector2::<f64> {
  let size_screen: Vector2::<f64> = [w as f64, h as f64];
  let size_units = vec2_scale(size_screen, 1.0/config.density);
  let half_size_units = vec2_scale(size_units, 0.5);
  vec2_sub(config.center, half_size_units)
}

fn image_to_world_position(config: &Config, origin: &Vector2::<f64>, x: u32, y: u32) -> Vector2::<f64> {
  let pos_screen_pixels: Vector2::<f64> = [x as f64, y as f64];
  let pos_screen = vec2_scale(pos_screen_pixels, 1.0/config.density);
  vec2_add(pos_screen, *origin)
}

fn generate_image (w: u32, h: u32, max_iteration: u32) -> ImageBuffer<Rgb<f64>, Vec<f64>> {
  let mut img = ImageBuffer::<Rgb<f64>, Vec<f64>>::new(w, h);

  let config: Config = Config {center: [0.0, -0.765], density: 437.246963563};
  let origin: Vector2::<f64> = origin_from_screen_size(&config, w, h);

  for (x, y, pixel) in img.enumerate_pixels_mut() {
    let pos = image_to_world_position(&config, &origin, x, y);
    let iteration = iterate_complex(max_iteration, pos);
    *pixel = map_color(iteration, max_iteration);
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

    let img = generate_image(WIDTH, HEIGHT, 1000);

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

fn benchmark(w: u32, h: u32, max_iteration: u32) {
  let now = Instant::now();
  generate_image(w, h, max_iteration);
  println!("Ran benchmark in {}ms", now.elapsed().as_millis());
}

#[test]
fn benchmark_1000() {
  benchmark(1000, 1000, 1000);
}