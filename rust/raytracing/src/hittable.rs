use crate::ray::Ray;

pub trait Hittable {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> bool;
}
