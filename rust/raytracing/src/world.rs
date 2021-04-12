use crate::hittable::{HitInfo, Hittable};
use crate::ray::Ray;

pub struct World {
    objects: Vec<Box<dyn Hittable>>,
}

impl World {
    pub fn new() -> Self {
        World {
            objects: Vec::new(),
        }
    }

    pub fn push(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
    pub fn clear(&mut self) {
        self.objects.clear();
    }
}

impl Hittable for World {
    fn hit(&mut self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitInfo> {
        self.objects
            .iter_mut()
            .find_map(|object| object.hit(ray, t_min, t_max))
    }
}
