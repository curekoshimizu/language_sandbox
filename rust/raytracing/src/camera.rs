use crate::degree::Degree;
use crate::rand::R2BallUniform;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_cornerl: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f64,
    ball_rand: R2BallUniform,
}

impl Camera {
    pub fn new(
        lookform: Point3,
        lookat: Point3,
        vup: Vec3,
        field_of_view: Degree,
        aspect_ratio: f64,
        aperture: f64,
        focus_distance: f64,
    ) -> Self {
        let h = (field_of_view.radians() / 2.0).tan();

        let viewpoint_height: f64 = 2.0 * h;
        let viewpoint_width: f64 = aspect_ratio * viewpoint_height;

        let w = (&lookform - &lookat).unit_vector();
        let u = vup.cross(&w).unit_vector();
        let v = w.cross(&u);

        let origin = lookform;
        let horizontal = focus_distance * viewpoint_width * &u;
        let vertical = focus_distance * viewpoint_height * &v;
        let lower_left_cornerl =
            &origin - &horizontal / 2.0 - &vertical / 2.0 - focus_distance * &w;

        Camera {
            origin,
            horizontal,
            vertical,
            lower_left_cornerl,
            u,
            v,
            w,
            lens_radius: aperture / 2.0,
            ball_rand: R2BallUniform::new(),
        }
    }

    pub fn get_ray(&mut self, s: f64, t: f64) -> Ray {
        let (x, y) = self.ball_rand.gen();
        let offset = self.lens_radius * (&self.u * x + &self.v * y);

        Ray::new(
            &self.origin + &offset,
            &self.lower_left_cornerl + s * &self.horizontal + t * &self.vertical
                - &self.origin
                - offset,
        )
    }
}
