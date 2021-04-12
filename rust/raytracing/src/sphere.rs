use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub trait Hittable {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> bool;
}

pub struct Sphere {
    center: Point3,
    radius: f64,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64) -> Self {
        Sphere {
            center: center,
            radius: radius,
        }
    }
    pub fn set_face_normal(&self, ray: &Ray, normal: &Vec3) -> Vec3 {
        let front_face = ray.direction.dot(normal) < 0.0;
        let outward_normal = if front_face { normal.clone() } else { -normal };

        outward_normal
    }
}

impl Hittable for Sphere {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        // NOTE:
        //  || (O + tD) - C ||^2 = r^2
        //  t^2 || D ||^2 - 2t (O-C, D) + || O-C ||^2 - r^2 = 0
        let oc = &ray.origin - &self.center;
        let a = ray.direction.length_squared();
        let b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return false;
        }

        let mut root = (-b - discriminant.sqrt()) / a;

        if !(t_min <= root && root <= t_max) {
            root = (-b + discriminant.sqrt()) / a;
            if !(t_min <= root && root <= t_max) {
                return false;
            }
        }

        let _point = ray.at(root);
        let _normal = (_point - &self.center) / self.radius;
        let _outward_normal = self.set_face_normal(ray, &_normal);

        true
    }
}