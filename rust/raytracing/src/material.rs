use crate::hittable::HitStatus;
use crate::rand;
use crate::rand::SphereUniform;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub trait Material {
    fn scatter(&mut self, ray: &Ray, hit_info: &HitStatus) -> Option<Ray>;
}

pub struct Lambertian {
    rand_sphere: SphereUniform,
}

impl Lambertian {
    pub fn new() -> Self {
        Lambertian {
            rand_sphere: rand::SphereUniform::new(),
        }
    }
}

impl Material for Lambertian {
    fn scatter(&mut self, _: &Ray, hit_info: &HitStatus) -> Option<Ray> {
        let direction: Vec3 = &hit_info.outward_normal + self.rand_sphere.gen_vec3();
        let scattered = Ray::new(hit_info.point.clone(), direction);

        Some(scattered)
    }
}
