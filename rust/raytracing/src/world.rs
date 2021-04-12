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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Lambertian;
    use crate::sphere::Sphere;
    use crate::vec3::{Point3, Vec3};

    #[test]
    fn empty_world() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));

        let mut world = World::new();
        assert!(world.hit(&ray, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));

        let mut world = World::new();
        let material = Box::new(Lambertian::new());
        world.push(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -1.0),
            0.5,
            material,
        )));

        assert!(world.hit(&ray, 0.0, f64::INFINITY).is_some());

        world.clear();
        assert!(world.hit(&ray, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn no_hit() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.8, -1.0));

        let mut world = World::new();
        let material = Box::new(Lambertian::new());
        world.push(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -1.0),
            0.5,
            material,
        )));

        assert!(world.hit(&ray, 0.0, f64::INFINITY).is_none());
    }
}
