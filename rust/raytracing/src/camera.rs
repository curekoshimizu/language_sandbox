use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_cornerl: Vec3,
}

impl Camera {
    pub fn new(aspect_raio: f64) -> Self {
        let viewpoint_height: f64 = 2.0;
        let viewpoint_width: f64 = aspect_raio * viewpoint_height;
        let focal_length: f64 = 1.0;

        let origin = Point3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewpoint_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewpoint_height, 0.0);
        let lower_left_cornerl =
            &origin - &horizontal / 2.0 - &vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Camera {
            origin: origin,
            horizontal: horizontal,
            vertical: vertical,
            lower_left_cornerl: lower_left_cornerl,
        }
    }

    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.origin.clone(),
            &self.lower_left_cornerl + u * &self.horizontal + v * &self.vertical - &self.origin,
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
