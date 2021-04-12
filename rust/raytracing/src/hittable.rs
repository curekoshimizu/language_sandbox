use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct HitInfo {
    pub point: Point3,
    pub outward_normal: Vec3,
    pub t: f64,
    pub material: Box<dyn Material>,
}

pub trait Hittable {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitInfo>;
}
