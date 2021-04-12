use crate::vec3::{Point3, Vec3};

#[derive(Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Point3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin: origin,
            direction: direction,
        }
    }

    pub fn at(&self, t: f64) -> Point3 {
        &self.origin + t * &self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn ray() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 2.0, 1.0));

        assert!(approx_eq!(Point3, ray.at(0.0), Point3::new(1.0, 2.0, 3.0)));
        assert!(approx_eq!(Point3, ray.at(1.0), Point3::new(3.0, 4.0, 4.0)));
    }
}
