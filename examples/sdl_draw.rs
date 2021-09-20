use sdl2::{Sdl, event::Event};
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::{Canvas, Texture};
use sdl2::video::Window;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn render_loop(context: Sdl, canvas: &mut Canvas<Window>, texture: &mut Texture) -> Result<(), String> {
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
    draw_texture(canvas, texture)?;
    //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
  }
  Ok(())
}

fn draw(texture_canvas: &mut Canvas<Window>) {
  for x in 0..(WIDTH) {
    for y in 0..(HEIGHT) {
      let color = Color::RGB(255,0,0);
      texture_canvas.set_draw_color(color);
      texture_canvas
      .draw_point(Point::new(x as i32, y as i32))
      .expect("could not draw point");
    }
  }
}

fn draw_texture(canvas: &mut Canvas<Window>, texture: &mut Texture) -> Result<(), String> {

  canvas.with_texture_canvas(texture, | draw_canvs | {
    draw(draw_canvs);
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

    render_loop(sdl_context, &mut canvas, &mut texture)?;


    Ok(())
}
