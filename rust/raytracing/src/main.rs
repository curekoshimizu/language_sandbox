mod camera;
mod color;
mod hittable;
mod rand;
mod ray;
mod sphere;
mod vec3;
mod world;

use crate::hittable::Hittable;
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

fn ray_color(ray: Ray, world: &mut World, rand_ball: &mut rand::BallUniform) -> Color {
    let mut cur_ray = ray;
    let mut cur_attenuation = 1.0;

    for _ in 0..MAX_DEPTH {
        if let Some(hash_info) = world.hit(&cur_ray, 0.0, f64::INFINITY) {
            let direction: Vec3 = &hash_info.outward_normal + rand_ball.gen_vec3();

            cur_attenuation *= 0.5;

            cur_ray = Ray::new(hash_info.point, direction);
        } else {
            // render sky

            let unit_direction: Vec3 = cur_ray.direction.unit_vector();

            // unit_direction.y in [-1, 1] => t in [0, 1]
            let t = 0.5 * (unit_direction.y + 1.0);

            return Color::from(
                cur_attenuation
                    * ((1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)),
            );
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
            let mut rand_ball = rand::BallUniform::new();
            let camera = Camera::new(ASPECT_RATIO);

            let mut world = World::new();
            world.push(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));
            world.push(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));

            (0..IMAGE_WIDTH)
                .map(move |i| {
                    let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

                    for _ in 0..SAMPLES_PER_PIXEL {
                        let u_delta = rand_uniform.gen();
                        let v_delta = rand_uniform.gen();

                        let u = (i as f64 + u_delta) / (IMAGE_WIDTH - 1) as f64;
                        let v = (j as f64 + v_delta) / (IMAGE_HEIGHT - 1) as f64;
                        let r = camera.get_ray(u, v);

                        let c: Vec3 = ray_color(r, &mut world, &mut rand_ball).into();
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
