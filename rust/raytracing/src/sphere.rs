use crate::hittable::{HitInfo, Hittable};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

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
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitInfo> {
        // NOTE:
        //  || (O + tD) - C ||^2 = r^2
        //  t^2 || D ||^2 - 2t (O-C, D) + || O-C ||^2 - r^2 = 0
        let oc = &ray.origin - &self.center;
        let a = ray.direction.length_squared();
        let b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let mut root = (-b - discriminant.sqrt()) / a;

        if !(t_min <= root && root <= t_max) {
            root = (-b + discriminant.sqrt()) / a;
            if !(t_min <= root && root <= t_max) {
                return None;
            }
        }

        let point = ray.at(root);
        let normal = (&point - &self.center) / self.radius;
        let outward_normal = self.set_face_normal(ray, &normal);

        Some(HitInfo {
            point: point,
            outward_normal: outward_normal,
            t: root,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));

        let mut sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5);

        assert!(sphere.hit(&ray, 0.0, f64::INFINITY).is_some());
    }

    #[test]
    fn no_hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.8, -1.0));

        let mut sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5);

        assert!(sphere.hit(&ray, 0.0, f64::INFINITY).is_none());
    }
}
