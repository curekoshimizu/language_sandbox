mod color;
mod hittable;
mod ray;
mod sphere;
mod vec3;
mod world;

use crate::hittable::Hittable;
use color::Color;
use ray::Ray;
use sphere::Sphere;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use vec3::{Point3, Vec3};
use world::World;

fn ray_color(ray: &Ray, world: &mut World) -> Color {
    if let Some(hash_info) = world.hit(&ray, 0.0, f64::INFINITY) {
        Color::from(0.5 * (hash_info.outward_normal + 1.0))
    } else {
        // render sky

        let unit_direction: Vec3 = ray.direction.unit_vector();

        // unit_direction.y in [-1, 1] => t in [0, 1]
        let t = 0.5 * (unit_direction.y + 1.0);

        ((1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)).into()
    }
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

    let mut world = World::new();
    world.push(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));
    world.push(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));

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

            let c = ray_color(&r, &mut world);
            write!(f, "{}\n", c)?;
        }
    }

    Ok(())
}
