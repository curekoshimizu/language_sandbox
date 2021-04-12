use crate::hittable::HitInfo;
use crate::rand::SphereUniform;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub trait Material {
    fn scatter(&mut self, ray: &Ray, hit_info: &HitInfo) -> Option<Ray>;
}

struct Lambertian {
    rand_sphere: SphereUniform,
}

impl Material for Lambertian {
    fn scatter(&mut self, ray: &Ray, hit_info: &HitInfo) -> Option<Ray> {
        let direction: Vec3 = &hit_info.outward_normal + self.rand_sphere.gen_vec3();
        let scattered = Ray::new(hit_info.point, direction);

        Some(scattered)
    }
}
