mod camera;
mod color;
mod hittable;
mod material;
mod rand;
mod ray;
mod sphere;
mod vec3;
mod world;

use crate::hittable::Hittable;
use crate::material::{Dielectric, Lambertian, Metal};
use camera::Camera;
use color::Color;
use ray::Ray;
use rayon::prelude::*;
use sphere::Sphere;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use vec3::{Point3, Vec3};
use world::World;

const MAX_DEPTH: usize = 50;

fn ray_color(ray: Ray, world: &mut World) -> Color {
    let mut cur_ray = ray;
    let mut cur_attenuation = Color::new(1.0, 1.0, 1.0);

    for _ in 0..MAX_DEPTH {
        if let Some(mut hit_info) = world.hit(&cur_ray, 0.001, f64::INFINITY) {
            if let Some(scattered) = hit_info.scatter(&cur_ray) {
                cur_ray = scattered.ray;
                cur_attenuation *= scattered.attenuation;
            } else {
                return Color::new(0.0, 0.0, 0.0);
            }
        } else {
            // render sky

            let unit_direction: Vec3 = cur_ray.direction.unit_vector();

            // unit_direction.y in [-1, 1] => t in [0, 1]
            let t = 0.5 * (unit_direction.y + 1.0);

            return cur_attenuation
                * ((1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)).into();
        }
    }

    return Color::new(0.0, 0.0, 0.0);
}

// Image
const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: usize = 400;
const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: usize = 100;

fn main() -> io::Result<()> {
    // render
    let mut f = BufWriter::new(File::create("image.ppm")?);

    writeln!(f, "P3")?;
    writeln!(f, "{} {}", IMAGE_WIDTH, IMAGE_HEIGHT)?;
    writeln!(f, "255")?;

    let mut result: Vec<Vec<[f64; 3]>> = Vec::with_capacity(IMAGE_HEIGHT);

    (0..IMAGE_HEIGHT)
        .rev()
        .collect::<Vec<usize>>()
        .par_iter()
        .map(|&j| {
            let mut rand_uniform = rand::RandUniform::new();
            let camera = Camera::new(ASPECT_RATIO);

            let mut world = World::new();
            world.push(Box::new(Sphere::new(
                Point3::new(0.0, -100.5, -1.0),
                100.0,
                Box::new(Lambertian::new(Color::new(0.8, 0.8, 0.0))),
            )));
            world.push(Box::new(Sphere::new(
                Point3::new(0.0, 0.0, -1.0),
                0.5,
                Box::new(Lambertian::new(Color::new(0.1, 0.2, 0.5))),
            )));
            world.push(Box::new(Sphere::new(
                Point3::new(-1.0, 0.0, -1.0),
                0.5,
                Box::new(Dielectric::new(1.5)),
            )));
            world.push(Box::new(Sphere::new(
                Point3::new(-1.0, 0.0, -1.0),
                -0.4,
                Box::new(Dielectric::new(1.5)),
            )));
            world.push(Box::new(Sphere::new(
                Point3::new(1.0, 0.0, -1.0),
                0.5,
                Box::new(Metal::new(Color::new(0.8, 0.6, 0.2), 0.0)),
            )));

            (0..IMAGE_WIDTH)
                .map(move |i| {
                    let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

                    // TODO: use sum
                    for _ in 0..SAMPLES_PER_PIXEL {
                        let u_delta = rand_uniform.gen();
                        let v_delta = rand_uniform.gen();

                        let u = (i as f64 + u_delta) / (IMAGE_WIDTH - 1) as f64;
                        let v = (j as f64 + v_delta) / (IMAGE_HEIGHT - 1) as f64;
                        let r = camera.get_ray(u, v);

                        let c: Vec3 = ray_color(r, &mut world).into();
                        pixel_color += c;
                    }
                    pixel_color /= SAMPLES_PER_PIXEL as f64;

                    pixel_color.to_xyz()
                })
                .collect::<Vec<[f64; 3]>>()
        })
        .collect_into_vec(&mut result);
    for xyzs in result.iter() {
        for xyz in xyzs {
            writeln!(f, "{}", Color::new(xyz[0], xyz[1], xyz[2]))?;
        }
    }

    Ok(())
}
