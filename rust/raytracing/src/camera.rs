use crate::degree::Degree;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_cornerl: Vec3,
}

impl Camera {
    pub fn new(
        lookform: Point3,
        lookat: Point3,
        vup: Vec3,
        field_of_view: Degree,
        aspect_raio: f64,
    ) -> Self {
        let h = (field_of_view.radians() / 2.0).tan();

        let viewpoint_height: f64 = 2.0 * h;
        let viewpoint_width: f64 = aspect_raio * viewpoint_height;
        let focal_length: f64 = 1.0;

        let w = (&lookform - &lookat).unit_vector();
        let u = vup.cross(&w).unit_vector();
        let v = w.cross(&u);

        let origin = lookform;
        let horizontal = viewpoint_width * u;
        let vertical = viewpoint_height * v;
        let lower_left_cornerl = &origin - &horizontal / 2.0 - &vertical / 2.0 - w;

        Camera {
            origin,
            horizontal,
            vertical,
            lower_left_cornerl,
        }
    }

    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray::new(
            self.origin.clone(),
            &self.lower_left_cornerl + s * &self.horizontal + t * &self.vertical - &self.origin,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn camera() {
        let camera = Camera::new(1.0);

        let ray = camera.get_ray(0.0, 0.0);
        assert!(approx_eq!(Vec3, ray.origin, Vec3::new(0.0, 0.0, 0.0)));
        assert!(approx_eq!(Vec3, ray.direction, Vec3::new(-1.0, -1.0, -1.0)));

        let ray = camera.get_ray(0.5, 0.5);
        assert!(approx_eq!(Vec3, ray.origin, Vec3::new(0.0, 0.0, 0.0)));
        assert!(approx_eq!(Vec3, ray.direction, Vec3::new(0.0, 0.0, -1.0)));

        let ray = camera.get_ray(1.0, 1.0);
        assert!(approx_eq!(Vec3, ray.origin, Vec3::new(0.0, 0.0, 0.0)));
        assert!(approx_eq!(Vec3, ray.direction, Vec3::new(1.0, 1.0, -1.0)));
    }
}
