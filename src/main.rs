#[macro_use] extern crate itertools;
use num_complex::Complex;
use sdl2::{Sdl, event::Event};
use sdl2::keyboard::Keycode;
use sdl2::pixels::{PixelFormatEnum};
use sdl2::rect::{Rect};
use sdl2::render::{Canvas, Texture};
use sdl2::video::Window;
use image::Rgb;
use vecmath::{Vector2, vec2_add, vec2_scale, vec2_sub};
use std::collections::HashMap;
use std::sync::{Arc};
use std::{num, thread};
use std::{cell::UnsafeCell, time::Instant};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use palette::{FromColor, Hsv, Srgb};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

struct ParallelPixelBuffer {
  width: u32,
  iter_map: UnsafeCell<HashMap<u32,u32>>,
  iterations: UnsafeCell<Vec<u32>>,
  rests: UnsafeCell<Vec<f32>>,
  max_iterations: u32
}

unsafe impl Sync for ParallelPixelBuffer {}

impl ParallelPixelBuffer {
    pub fn new(width: u32, height: u32, max_iterations: u32) -> ParallelPixelBuffer {
        ParallelPixelBuffer {
            width: width,
            iterations: UnsafeCell::new(vec![0; (width * height) as usize]),
            rests: UnsafeCell::new(vec![0.0; (width * height) as usize]),
            iter_map: UnsafeCell::new(HashMap::<u32,u32>::new()),
            max_iterations: max_iterations
        }
    }

    pub fn put_pixel(&self, l: u32, t: u32, it: u32, re: f32) {
        unsafe {
            let iterations = &mut *self.iterations.get();
            let rests = &mut *self.rests.get();
            let iter_map = &mut *self.iter_map.get();
            let base = ((t * self.width) + l) as usize;
            *iterations.get_unchecked_mut(base) = it;
            *rests.get_unchecked_mut(base) = re;

            if it < self.max_iterations {
              let mut count = 0;
              if iter_map.contains_key(&it) {
                count = *iter_map.get(&it).unwrap();
              }
              count += 1;
              //println!("count {}", count);
              iter_map.insert(it, count);
            }
        }
    }

    pub fn get_iterations(&self) -> &Vec<u32> {
      unsafe {
        &*self.iterations.get()
      }
    }

    pub fn get_iter_map(&self) -> &HashMap<u32,u32> {
      unsafe {
        &*self.iter_map.get()
      }
    }

    pub fn get_iter_total(&self) -> u32 {
      let mut total = 0;

      for count in self.get_iter_map().values() {
        total += count;
      }

      total
    }

    pub fn get_hue_for_iter(&self, it: u32, total: u32) -> f32 {
      let mut hue = 0.0;

      for (ito, count) in self.get_iter_map() {
        if ito <= &it {
          hue += *count as f32 / total as f32;
        }
      }

      hue
    }

    pub fn get_rests(&self) -> &Vec<f32> {
      unsafe {
        &*self.rests.get()
      }
    }
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
  density: f64,
  iterations: u32
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

fn interpolate_u8(color1: &Rgb<u8>, color2: &Rgb<u8>, factor: f64) -> Rgb<u8> {
  let factor_inv = 1.0 - factor;
  let r = (color1[0] as f64 * factor + color2[0] as f64 * factor_inv) as u8;
  let g = (color1[1] as f64 * factor + color2[1] as f64 * factor_inv) as u8;
  let b = (color1[2] as f64 * factor + color2[2] as f64 * factor_inv) as u8;

  Rgb::<u8>([r, g, b])
}

fn generate_image_parallel(config: &Config, pixels: &ParallelPixelBuffer, w: u32, h: u32) {

  let bob_ross = vec![
    Rgb::<u8>([0x00, 0x00, 0x00]), // Midnight black
    Rgb::<u8>([0x02, 0x1e, 0x44]), // Prussian blue
    Rgb::<u8>([0x0a, 0x34, 0x10]), // Sap green
    Rgb::<u8>([0x0c, 0x00, 0x40]), // Phthalo blue
    Rgb::<u8>([0x10, 0x2e, 0x3c]), // Phthalo green
    Rgb::<u8>([0x22, 0x1b, 0x15]), // Van Dyke brown
    Rgb::<u8>([0x4e, 0x15, 0x00]), // Alizarin crimson
    Rgb::<u8>([0x5f, 0x2e, 0x1f]), // Dark Sienna
    Rgb::<u8>([0xc7, 0x9b, 0x00]), // Yellow ochre
    Rgb::<u8>([0xdb, 0x00, 0x00]), // Bright red
    Rgb::<u8>([0xff, 0x3c, 0x00]), // Cadmium yellow
    Rgb::<u8>([0xff, 0xb8, 0x00]), // Indian yellow
    Rgb::<u8>([0xff, 0xff, 0xff]), // Titanium white
  ];

  let mut coords: Vec<_> = iproduct!(0..w, 0..h).collect();

  let mut rng = thread_rng();
  coords.shuffle(&mut rng);

  let origin: Vector2::<f64> = origin_from_screen_size(&config, w, h);

  coords.par_iter().for_each(|&(x, y)| {
    let pos = image_to_world_position(&config, &origin, x, y);
    /*
    let (iteration, rest) = iterate_naive_interpolate(config.iterations, pos);

    let color1 = map_color(iteration, config.iterations);
    let color2 = map_color(iteration + 1, config.iterations);

    let color = interpolate(color1, color2, rest);
    */

    /*
    let iteration = iterate_complex(config.iterations, pos);
    let color = map_color(iteration, config.iterations);
    */

    let iteration = iterate_complex(config.iterations, pos);
    //let (iteration, rest) = iterate_naive_interpolate(config.iterations, pos);

    /*
    */
    let iteration_ratio = iteration as f64 / config.iterations as f64;

    let iteration_ratio = iteration_ratio.powf(1.0/3.0);

    //let hue = iteration as f64 % 360.0;
    let hue = iteration_ratio * 360.0;
    let foo = 1.0 - iteration_ratio;
    let hsv = Hsv::new(hue, foo, foo);
    let rgb = Srgb::from_color(hsv);

    //let color = bob_ross.get((iteration%13) as usize).unwrap();

    /*
    let color1 = bob_ross.get((iteration%13) as usize).unwrap();
    let color2 = bob_ross.get(((iteration+1)%13) as usize).unwrap();

    let color = interpolate_u8(color1, color2, rest);
    */

    //pixels.put_pixel(x, y, (color[0] * 255.0) as u8, (color[1] * 255.0) as u8, (color[2] * 255.0) as u8);

    //pixels.put_pixel(x, y, (rgb.red * 255.0) as u8, (rgb.green * 255.0) as u8, (rgb.blue * 255.0) as u8);

    pixels.put_pixel(x, y, iteration, 0.0);

    //pixels.put_pixel(x, y, color[0], color[1], color[2]);
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
  }
  Ok(())
}

fn draw_texture(canvas: &mut Canvas<Window>, texture: &mut Texture, img: &ParallelPixelBuffer) -> Result<(), String> {
  let rect: Rect = Rect::new(0,0, WIDTH, HEIGHT);

  let mut colors = Vec::<u8>::new();

  let bob_ross = vec![
    Rgb::<u8>([0x00, 0x00, 0x00]), // Midnight black
    Rgb::<u8>([0x02, 0x1e, 0x44]), // Prussian blue
    Rgb::<u8>([0x0a, 0x34, 0x10]), // Sap green
    Rgb::<u8>([0x0c, 0x00, 0x40]), // Phthalo blue
    Rgb::<u8>([0x10, 0x2e, 0x3c]), // Phthalo green
    Rgb::<u8>([0x22, 0x1b, 0x15]), // Van Dyke brown
    Rgb::<u8>([0x4e, 0x15, 0x00]), // Alizarin crimson
    Rgb::<u8>([0x5f, 0x2e, 0x1f]), // Dark Sienna
    Rgb::<u8>([0xc7, 0x9b, 0x00]), // Yellow ochre
    Rgb::<u8>([0xdb, 0x00, 0x00]), // Bright red
    Rgb::<u8>([0xff, 0x3c, 0x00]), // Cadmium yellow
    Rgb::<u8>([0xff, 0xb8, 0x00]), // Indian yellow
    Rgb::<u8>([0xff, 0xff, 0xff]), // Titanium white
  ];

  let total = img.get_iter_total();
  let mut hue_cache: HashMap<u32,f32> = HashMap::<u32,f32>::new();

  for iteration in img.get_iterations().into_iter() {

    if iteration < &img.max_iterations {

      let mut hue = 0.0;
      match hue_cache.get(iteration) {
        Some(foo) => {
          hue = *foo;
        },
        None => {
          hue = img.get_hue_for_iter(*iteration, total);
          hue_cache.insert(*iteration, hue);
        }
      }

      /*
      if hue_cache.contains_key(&iteration) {
        hue = *hue_cache.get(iteration).unwrap();
      } else {
        hue = img.get_hue_for_iter(*iteration, total);
        hue_cache.insert(*iteration, hue);
      }
      */

      let hsv = Hsv::new(hue * 360.0, 1.0, 1.0);
      let rgb = Srgb::from_color(hsv);

      colors.push((rgb.red * 255.0) as u8);
      colors.push((rgb.green * 255.0) as u8);
      colors.push((rgb.blue * 255.0) as u8);


    } else {
      colors.push(0);
      colors.push(0);
      colors.push(0);
    }
  }

  texture.update(rect, &colors, (3*WIDTH) as usize).map_err(|e| e.to_string())?;

  canvas.copy(&texture, None, None)?;
  canvas.present();

  Ok(())
}

fn init_sdl() -> Result<(Sdl, Canvas<Window>), String> {
  let sdl_context = sdl2::init()?;
  let video_subsystem = sdl_context.video()?;

  let window = video_subsystem
      .window("Mandelbrot", WIDTH, HEIGHT)
      .position_centered()
      .opengl()
      .build()
      .map_err(|e| e.to_string())?;

  let canvas = window
      .into_canvas()
      .target_texture()
      .present_vsync()
      .build()
      .map_err(|e| e.to_string())?;

  Ok((sdl_context, canvas))
}

fn render_config(config: Config) -> Result<(), String> {
  let (sdl_context, mut canvas) = init_sdl().unwrap();

  let p = ParallelPixelBuffer::new(WIDTH, HEIGHT, config.iterations);
  let pixels = Arc::new(p);
  let f = pixels.clone();

  thread::spawn(move || {
    generate_image_parallel(&config, &f, WIDTH, HEIGHT);
  });

  let texture_creator = canvas.texture_creator();
  let mut texture: Texture = texture_creator
      .create_texture_streaming(PixelFormatEnum::RGB24, WIDTH, HEIGHT)
      .map_err(|e| e.to_string())?;
  render_loop(sdl_context, &mut canvas, &mut texture, &pixels)?;

  Ok(())
}

pub fn main() -> Result<(), String> {
  let nice_center: Vector2::<f64> = [0.360240443437614363236125244449545308482607807958585750488375814740195346059218100311752936722773426396233731729724987737320035372683285317664532401218521579554288661726564324134702299962817029213329980895208036363104546639698106204384566555001322985619004717862781192694046362748742863016467354574422779443226982622356594130430232458472420816652623492974891730419252651127672782407292315574480207005828774566475024380960675386215814315654794021855269375824443853463117354448779647099224311848192893972572398662626725254769950976527431277402440752868498588785436705371093442460696090720654908973712759963732914849861213100695402602927267843779747314419332179148608587129105289166676461292845685734536033692577618496925170576714796693411776794742904333484665301628662532967079174729170714156810530598764525260869731233845987202037712637770582084286587072766838497865108477149114659838883818795374195150936369987302574377608649625020864292915913378927790344097552591919409137354459097560040374880346637533711271919419723135538377394364882968994646845930838049998854075817859391340445151448381853615103761584177161812057928,
  -0.6413130610648031748603750151793020665794949522823052595561775430644485741727536902556370230689681162370740565537072149790106973211105273740851993394803287437606238596262287731075999483940467161288840614581091294325709988992269165007394305732683208318834672366947550710920088501655704252385244481168836426277052232593412981472237968353661477793530336607247738951625817755401065045362273039788332245567345061665756708689359294516668271440525273653083717877701237756144214394870245598590883973716531691124286669552803640414068523325276808909040317617092683826521501539932397262012011082098721944643118695001226048977430038509470101715555439047884752058334804891389685530946112621573416582482926221804767466258346014417934356149837352092608891639072745930639364693513216719114523328990690069588676087923656657656023794484324797546024248328156586471662631008741349069961493817600100133439721557969263221185095951241491408756751582471307537382827924073746760884081704887902040036056611401378785952452105099242499241003208013460878442953408648178692353788153787229940221611731034405203519945313911627314900851851072122990492499999999999999999991];

  let config: Config = Config {
    center: nice_center,
    density: 43700.246963563,
    iterations: 100000
  };

  let config: Config = Config {
    center: [nice_center[1], nice_center[0]],
    density: 43700.246963563 * 10000000.0,
    iterations: 100000
  };

  /*
  let config: Config = Config {
    center: [0.0, -0.765],
    density: 437.246963563,
    iterations: 1000
  };
  */

  render_config(config)?;

  Ok(())
}


fn benchmark_parallel(w: u32, h: u32) {
  let now = Instant::now();

  let config: Config = Config {
    center: [0.0, -0.765],
    density: 437.246963563,
    iterations: 1000
  };

  let p = ParallelPixelBuffer::new(w, h, config.iterations);

  generate_image_parallel(&config, &p, w, h);
  println!("Ran benchmark in {}ms", now.elapsed().as_millis());
}


#[test]
fn benchmark() {
  benchmark_parallel(1000, 1000);
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
