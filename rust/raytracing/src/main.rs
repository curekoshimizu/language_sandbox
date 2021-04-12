mod color;
mod ray;
mod vec3;

use color::Color;
use ray::Ray;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use vec3::{Point3, Vec3};

fn hit_sphere(ray: &Ray, center: &Point3, radius: f64) -> bool {
    let oc = &ray.origin - center;
    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.dot(&oc) - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    discriminant > 0.0
}

fn ray_coloro(ray: &Ray) -> Color {
    if hit_sphere(&ray, &Point3::new(0.0, 0.0, -1.0), 0.5) {
        return Color::new(1.0, 0.0, 0.0);
    }

    let unit_direction: Vec3 = ray.direction.unit_vector();

    // unit_direction.y in [-1, 1] => t in [0, 1]
    let t = 0.5 * (unit_direction.y + 1.0);

    ((1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)).into()
}

// Image
const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: usize = 400;
const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

// Camera
const VIEWPOINT_HEIGHT: f64 = 2.0;
const VIEWPOINT_WIDTH: f64 = ASPECT_RATIO * VIEWPOINT_HEIGHT;
const FOCAL_LENGTH: f64 = 1.0;

fn main() -> io::Result<()> {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(VIEWPOINT_WIDTH, 0.0, 0.0);
    let vertical = Vec3::new(0.0, VIEWPOINT_HEIGHT, 0.0);
    let lower_left_cornerl =
        &origin - &horizontal / 2.0 - &vertical / 2.0 - Vec3::new(0.0, 0.0, FOCAL_LENGTH);

    // render

    let mut f = BufWriter::new(File::create("image.ppm")?);

    write!(f, "P3\n")?;
    write!(f, "{} {}\n", IMAGE_WIDTH, IMAGE_HEIGHT)?;
    write!(f, "255\n")?;

    for j in (0..IMAGE_HEIGHT).rev() {
        for i in 0..IMAGE_WIDTH {
            let u = i as f64 / (IMAGE_WIDTH - 1) as f64;
            let v = j as f64 / (IMAGE_HEIGHT - 1) as f64;
            let r = Ray::new(
                origin.clone(),
                &lower_left_cornerl + u * &horizontal + v * &vertical - &origin,
            );

            let c = ray_coloro(&r);
            write!(f, "{}\n", c)?;
        }
    }

    Ok(())
}
