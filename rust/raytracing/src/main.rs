mod camera;
mod color;
mod degree;
mod hittable;
mod material;
mod rand;
mod ray;
mod sphere;
mod vec3;
mod world;

use crate::degree::Degree;
use crate::hittable::Hittable;
use crate::material::{Dielectric, Lambertian, Metal};
use crate::rand::RandUniform;
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
const ASPECT_RATIO: f64 = 3.0 / 2.0;
const IMAGE_WIDTH: usize = 1200;
const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: usize = 500;

fn random_scene() -> World {
    let mut world = World::new();

    world.push(Box::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Box::new(Lambertian::new(Color::new(0.5, 0.5, 0.5))),
    )));

    let mut uniform = RandUniform::from_seed(0);

    for a in -11..11 {
        for b in -11..11 {
            let center = Point3::new(
                a as f64 + 0.9 * uniform.gen(),
                0.2,
                b as f64 + 0.9 * uniform.gen(),
            );

            if (&center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                let choose_mat = uniform.gen();
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::new(uniform.gen(), uniform.gen(), uniform.gen())
                        * Color::new(uniform.gen(), uniform.gen(), uniform.gen());
                    world.push(Box::new(Sphere::new(
                        center.clone(),
                        0.2,
                        Box::new(Lambertian::new(albedo)),
                    )));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::new(
                        0.5 * uniform.gen() + 0.5,
                        0.5 * uniform.gen() + 0.5,
                        0.5 * uniform.gen() + 0.5,
                    );
                    world.push(Box::new(Sphere::new(
                        center.clone(),
                        0.2,
                        Box::new(Metal::new(albedo, uniform.gen() / 10.0)),
                    )));
                } else {
                    // glass
                    world.push(Box::new(Sphere::new(
                        center.clone(),
                        0.2,
                        Box::new(Dielectric::new(1.5)),
                    )));
                }
            }
        }
    }

    world.push(Box::new(Sphere::new(
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        Box::new(Dielectric::new(1.5)),
    )));
    world.push(Box::new(Sphere::new(
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        Box::new(Lambertian::new(Color::new(0.4, 0.2, 0.1))),
    )));
    world.push(Box::new(Sphere::new(
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        Box::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0)),
    )));

    world
}

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

            let lookfrom = Point3::new(13.0, 3.0, 2.0);
            let lookat = Point3::new(0.0, 0.0, -1.0);
            let vup = Point3::new(0.0, 1.0, 0.0);
            let aperture = 0.1;
            let dist_to_focus = 10.0;

            let mut camera = Camera::new(
                lookfrom,
                lookat,
                vup,
                Degree::new(20.0),
                ASPECT_RATIO,
                aperture,
                dist_to_focus,
            );

            let mut world = random_scene();
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
