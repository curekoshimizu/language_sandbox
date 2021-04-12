use crate::material::{Material, ScatterInfo};
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

pub struct HitStatus {
    pub point: Point3,
    pub outward_normal: Vec3,
    pub front_face: bool,
    pub t: f64,
}

pub struct HitInfo<'a> {
    pub hit_status: HitStatus,
    pub material: &'a mut Box<dyn Material>,
}

impl<'a> HitInfo<'a> {
    pub fn scatter(&mut self, ray: &Ray) -> Option<ScatterInfo> {
        self.material.scatter(ray, &self.hit_status)
    }
}

pub trait Hittable {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitInfo>;
}
