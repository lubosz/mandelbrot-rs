#[macro_use] extern crate itertools;
use num_complex::Complex;
use sdl2::{Sdl, event::Event};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::{Canvas, Texture};
use sdl2::video::Window;
use image::{Rgb, ImageBuffer};
use vecmath::{Vector2, vec2_add, vec2_scale, vec2_sub};
use std::sync::{Arc, mpsc};
use std::thread;
use std::{cell::UnsafeCell, time::{Duration, Instant}};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

type Iteration = fn(u32, Vector2::<f64>) -> u32;

struct ParallelPixelBuffer {
  width: u32,
  contents: UnsafeCell<Vec<f64>>,
}

unsafe impl Sync for ParallelPixelBuffer {}

impl ParallelPixelBuffer {
    pub fn new(width: u32, height: u32) -> ParallelPixelBuffer {
        ParallelPixelBuffer {
            width: width,
            contents: UnsafeCell::new(vec![0.0; (width * height * 3) as usize]),
        }
    }

    pub fn put_pixel(&self, l: u32, t: u32, x: f64, y: f64, z: f64) {
        unsafe {
            let contents = &mut *self.contents.get();
            let base = (((t * self.width) + l) * 3) as usize;
            *contents.get_unchecked_mut(base) = x;
            *contents.get_unchecked_mut(base + 1) = y;
            *contents.get_unchecked_mut(base + 2) = z;
        }
    }

    pub fn into_inner(self) -> Vec<f64> {
        self.contents.into_inner()
    }

    pub fn get_color(&self, x: u32, y: u32) -> Color {
      unsafe {
        let contents = &*self.contents.get();
        let base = (((y * self.width) + x) * 3) as usize;
        let r: u8 = (*contents.get_unchecked(base) * 255.0) as u8;
        let g: u8 = (*contents.get_unchecked(base + 1) * 255.0) as u8;
        let b: u8 = (*contents.get_unchecked(base + 2) * 255.0) as u8;
        Color::RGB(r, g, b)
      }
    }

    pub fn get_image(self) -> ImageBuffer<Rgb<f64>, Vec<f64>> {
      ImageBuffer::from_raw(WIDTH, HEIGHT, self.contents.into_inner()).expect("Buffer not big enough??")
    }
}

fn draw(texture_canvas: &mut Canvas<Window>, pixels: &ParallelPixelBuffer) {
    for x in 0..(WIDTH) {
      for y in 0..(HEIGHT) {
        let color = pixels.get_color(x, y);
        texture_canvas.set_draw_color(color);
        texture_canvas
        .draw_point(Point::new(x as i32, y as i32))
        .expect("could not draw point");
      }
    }

    /*
    let image: ImageBuffer<Rgb<f64>, Vec<f64>> = &pixels.get_image();

    for (x, y, pixel) in image.enumerate_pixels() {
      texture_canvas.set_draw_color(Color::RGB((pixel[0] * 255.0) as u8,
      (pixel[1] * 255.0) as u8,
      (pixel[2] * 255.0) as u8));
      texture_canvas
      .draw_point(Point::new(x as i32, y as i32))
      .expect("could not draw point");
    }
    */
}

fn map_color(iteration: u32, max_iteration: u32) -> Rgb::<f64> {
  let color = iteration as f64 / max_iteration as f64;

  if iteration >= max_iteration {
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


fn iterate_naive(max_iteration: u32, pos: Vector2::<f64>) -> u32 {
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

fn iterate_naive_interpolate(max_iteration: u32, pos: Vector2::<f64>) -> (u32, f64) {
  let mut i = 0;
  let mut x = 0.0;
  let mut y = 0.0;

  while x*x + y*y <= 2.0*2.0 && i < max_iteration {
    let ytemp = y*y - x*x + pos[1];
    x = 2.0 * x*y + pos[0];
    y = ytemp;
    i += 1;
  }

  let mut fi :f64 = 0.0;
  if i < max_iteration {

    // sqrt of inner term removed using log simplification rules.
    let log_zn = f64::log10(x*x + y*y) / 2.0;
    let nu = f64::log10(log_zn / f64::log10(2.0)) / f64::log10(2.0);
    // Rearranging the potential function.
    // Dividing log_zn by log(2) instead of log(N = 1<<8)
    // because we want the entire palette to range from the
    // center to radius 2, NOT our bailout radius.
    fi = i as f64 + 1.0 - nu;
    i = fi.floor() as u32;
  }

  (i, fi % 1.0)
}

fn iterate_optimized(max_iteration: u32, pos: Vector2::<f64>) -> u32 {
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

fn interpolate(color1: Rgb<f64>, color2: Rgb<f64>, factor: f64) -> Rgb<f64> {
  let factor_inv = 1.0 - factor;
  let r = color1[0] * factor + color2[0] * factor_inv;
  let g = color1[1] * factor + color2[1] * factor_inv;
  let b = color1[2] * factor + color2[2] * factor_inv;

  Rgb::<f64>([r, g, b])
}

fn generate_image (w: u32, h: u32, max_iteration: u32, it: Iteration) -> ImageBuffer<Rgb<f64>, Vec<f64>> {
  let mut img = ImageBuffer::<Rgb<f64>, Vec<f64>>::new(w, h);

  let config: Config = Config {center: [0.360240443437614363236125244449545308482607807958585750488375814740195346059218100311752936722773426396233731729724987737320035372683285317664532401218521579554288661726564324134702299962817029213329980895208036363104546639698106204384566555001322985619004717862781192694046362748742863016467354574422779443226982622356594130430232458472420816652623492974891730419252651127672782407292315574480207005828774566475024380960675386215814315654794021855269375824443853463117354448779647099224311848192893972572398662626725254769950976527431277402440752868498588785436705371093442460696090720654908973712759963732914849861213100695402602927267843779747314419332179148608587129105289166676461292845685734536033692577618496925170576714796693411776794742904333484665301628662532967079174729170714156810530598764525260869731233845987202037712637770582084286587072766838497865108477149114659838883818795374195150936369987302574377608649625020864292915913378927790344097552591919409137354459097560040374880346637533711271919419723135538377394364882968994646845930838049998854075817859391340445151448381853615103761584177161812057928, -0.6413130610648031748603750151793020665794949522823052595561775430644485741727536902556370230689681162370740565537072149790106973211105273740851993394803287437606238596262287731075999483940467161288840614581091294325709988992269165007394305732683208318834672366947550710920088501655704252385244481168836426277052232593412981472237968353661477793530336607247738951625817755401065045362273039788332245567345061665756708689359294516668271440525273653083717877701237756144214394870245598590883973716531691124286669552803640414068523325276808909040317617092683826521501539932397262012011082098721944643118695001226048977430038509470101715555439047884752058334804891389685530946112621573416582482926221804767466258346014417934356149837352092608891639072745930639364693513216719114523328990690069588676087923656657656023794484324797546024248328156586471662631008741349069961493817600100133439721557969263221185095951241491408756751582471307537382827924073746760884081704887902040036056611401378785952452105099242499241003208013460878442953408648178692353788153787229940221611731034405203519945313911627314900851851072122990492499999999999999999991],
    density: 43700.246963563};
  let origin: Vector2::<f64> = origin_from_screen_size(&config, w, h);

  for (x, y, pixel) in img.enumerate_pixels_mut() {
    let pos = image_to_world_position(&config, &origin, x, y);
    //let mut iteration = it(max_iteration, pos);
    let (iteration, rest) = iterate_naive_interpolate(max_iteration, pos);

    let color1 = map_color(iteration, max_iteration);
    let color2 = map_color(iteration + 1, max_iteration);

    *pixel = interpolate(color1, color2, rest);
  }

  img
}

fn generate_image_parallel(pixels: &ParallelPixelBuffer, w: u32, h: u32, max_iteration: u32) {

  let mut coords: Vec<_> = iproduct!(0..w, 0..h).collect();

  let mut rng = thread_rng();
  coords.shuffle(&mut rng);


  let config: Config = Config {center: [0.360240443437614363236125244449545308482607807958585750488375814740195346059218100311752936722773426396233731729724987737320035372683285317664532401218521579554288661726564324134702299962817029213329980895208036363104546639698106204384566555001322985619004717862781192694046362748742863016467354574422779443226982622356594130430232458472420816652623492974891730419252651127672782407292315574480207005828774566475024380960675386215814315654794021855269375824443853463117354448779647099224311848192893972572398662626725254769950976527431277402440752868498588785436705371093442460696090720654908973712759963732914849861213100695402602927267843779747314419332179148608587129105289166676461292845685734536033692577618496925170576714796693411776794742904333484665301628662532967079174729170714156810530598764525260869731233845987202037712637770582084286587072766838497865108477149114659838883818795374195150936369987302574377608649625020864292915913378927790344097552591919409137354459097560040374880346637533711271919419723135538377394364882968994646845930838049998854075817859391340445151448381853615103761584177161812057928, -0.6413130610648031748603750151793020665794949522823052595561775430644485741727536902556370230689681162370740565537072149790106973211105273740851993394803287437606238596262287731075999483940467161288840614581091294325709988992269165007394305732683208318834672366947550710920088501655704252385244481168836426277052232593412981472237968353661477793530336607247738951625817755401065045362273039788332245567345061665756708689359294516668271440525273653083717877701237756144214394870245598590883973716531691124286669552803640414068523325276808909040317617092683826521501539932397262012011082098721944643118695001226048977430038509470101715555439047884752058334804891389685530946112621573416582482926221804767466258346014417934356149837352092608891639072745930639364693513216719114523328990690069588676087923656657656023794484324797546024248328156586471662631008741349069961493817600100133439721557969263221185095951241491408756751582471307537382827924073746760884081704887902040036056611401378785952452105099242499241003208013460878442953408648178692353788153787229940221611731034405203519945313911627314900851851072122990492499999999999999999991],
    density: 43700.246963563};
  let origin: Vector2::<f64> = origin_from_screen_size(&config, w, h);

  //for (x, y, pixel) in img.enumerate_pixels_mut() {
  coords.par_iter().for_each(|&(x, y)| {
    let pos = image_to_world_position(&config, &origin, x, y);
    //let mut iteration = it(max_iteration, pos);
    let (iteration, rest) = iterate_naive_interpolate(max_iteration, pos);

    let color1 = map_color(iteration, max_iteration);
    let color2 = map_color(iteration + 1, max_iteration);

    let color = interpolate(color1, color2, rest);
    pixels.put_pixel(x, y, color[0], color[1], color[2]);
  });
}

fn render_loop(context: Sdl, canvas: &mut Canvas<Window>, texture: &mut Texture, pixels: &ParallelPixelBuffer) -> Result<(), String> {
  let mut event_pump = context.event_pump()?;
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
    draw_texture(canvas, texture, pixels)?;
    //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
  }
  Ok(())
}

fn draw_texture(canvas: &mut Canvas<Window>, texture: &mut Texture, img: &ParallelPixelBuffer) -> Result<(), String> {

  canvas.with_texture_canvas(texture, | draw_canvs | {
    draw(draw_canvs, &img);
  }).map_err(|e| e.to_string())?;

  canvas.copy(&texture, None, None)?;
  canvas.present();
  println!("Updating image.");

  Ok(())
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

    //let (tx, rx) = mpsc::channel();

    let p = ParallelPixelBuffer::new(WIDTH, HEIGHT);
    let pixels = Arc::new(p);
    let f = pixels.clone();

    thread::spawn(move || {
        generate_image_parallel(&f, WIDTH, HEIGHT, 10000);
    });
    render_loop(sdl_context, &mut canvas, &mut texture, &pixels)?;


    Ok(())
}

fn benchmark(w: u32, h: u32, max_iteration: u32, it: Iteration) {
  let now = Instant::now();
  generate_image(w, h, max_iteration, it);
  println!("Ran benchmark in {}ms", now.elapsed().as_millis());
}

fn benchmark_parrelel(w: u32, h: u32, max_iteration: u32, it: Iteration) {
  let now = Instant::now();
  // generate_image_parallel(w, h, max_iteration);
  println!("Ran benchmark in {}ms", now.elapsed().as_millis());
}


#[test]
fn benchmark_naive() {
  benchmark(1000, 1000, 1000, iterate_naive);
}

#[test]
fn benchmark_parrelel_naive() {
  benchmark_parrelel(1000, 1000, 1000, iterate_naive);
}


#[test]
fn benchmark_optimized() {
  benchmark(1000, 1000, 1000, iterate_optimized);
}

#[test]
fn benchmark_complex() {
  benchmark(1000, 1000, 1000, iterate_complex);
}

#[test]
fn test_fmod() {
  let foo = 2.46453;
  let bar = foo % 1.0;
  println!("bar {}", bar);
}

#[test]
fn test_zero_zero() {
  let pos: Vector2::<f64> = [0.0, 0.0];
  let (iteration, rest) = iterate_naive_interpolate(1000, pos);

  let color1 = map_color(iteration, 1000);
  let color2 = map_color(iteration + 1, 1000);

  println!("Iteration {} Rest {}", iteration, rest);
  println!("Color 1 {} {} {}", color1[0], color1[1], color1[2]);
  println!("Color 2 {} {} {}", color2[0], color2[1], color2[2]);
}
