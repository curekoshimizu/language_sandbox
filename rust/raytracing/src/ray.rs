use crate::vec3::{Point3, Vec3};

pub struct Ray {
    origin: Vec3,
    direction: Point3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin: origin,
            direction: direction,
        }
    }

    pub fn at(&self, t: f64) -> Point3 {
        &self.origin + &Vec3::new(t, t, t) * &self.direction
    }
}
