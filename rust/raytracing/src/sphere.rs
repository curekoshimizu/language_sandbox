use crate::hittable::{HitInfo, HitStatus, Hittable};
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct Sphere {
    center: Point3,
    radius: f64,
    material: Box<dyn Material>,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Box<dyn Material>) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
    pub fn set_face_normal(&self, ray: &Ray, normal: &Vec3) -> (Vec3, bool) {
        let front_face = ray.direction.dot(normal) < 0.0;
        let outward_normal = if front_face { normal.clone() } else { -normal };

        (outward_normal, front_face)
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
        let (outward_normal, front_face) = self.set_face_normal(ray, &normal);

        Some(HitInfo {
            hit_status: HitStatus {
                point,
                outward_normal,
                front_face,
                t: root,
            },
            material: &mut self.material,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::material::Lambertian;

    #[test]
    fn hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));

        let material = Box::new(Lambertian::new(Color::new(1.0, 1.0, 1.0)));
        let mut sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5, material);

        assert!(sphere.hit(&ray, 0.0, f64::INFINITY).is_some());
    }

    #[test]
    fn no_hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.8, -1.0));

        let material = Box::new(Lambertian::new(Color::new(1.0, 1.0, 1.0)));
        let mut sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5, material);

        assert!(sphere.hit(&ray, 0.0, f64::INFINITY).is_none());
    }
}
