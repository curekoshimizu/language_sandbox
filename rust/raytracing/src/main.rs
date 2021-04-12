mod color;
mod vec3;

use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;

fn main() -> io::Result<()> {
    const IMAGE_WIDTH: usize = 256;
    const IMAGE_HEIGHT: usize = 256;

    let mut f = BufWriter::new(File::create("image.ppg")?);

    write!(f, "P3\n")?;
    write!(f, "{} {}\n", IMAGE_WIDTH, IMAGE_HEIGHT)?;
    write!(f, "255\n")?;

    for j in (0..IMAGE_HEIGHT).rev() {
        for i in 0..IMAGE_WIDTH {
            let r: f64 = i as f64 / (IMAGE_WIDTH - 1) as f64;
            let g: f64 = j as f64 / (IMAGE_WIDTH - 1) as f64;
            let b: f64 = 0.25;

            let c = color::Color::new(r, g, b);
            write!(f, "{}\n", c)?;
        }
    }

    Ok(())
}
