use crate::color::Color;
use crate::hittable::HitStatus;
use crate::rand;
use crate::rand::SphereUniform;
use crate::ray::Ray;
use crate::vec3::Vec3;

pub struct ScatterInfo {
    pub ray: Ray,
    pub attenuation: Color,
}

pub trait Material {
    fn scatter(&mut self, ray: &Ray, hit_info: &HitStatus) -> Option<ScatterInfo>;
}

pub struct Lambertian {
    rand_sphere: SphereUniform,
    albert: Color,
}

impl Lambertian {
    pub fn new(albert: Color) -> Self {
        Lambertian {
            rand_sphere: rand::SphereUniform::new(),
            albert: albert,
        }
    }
}

impl Material for Lambertian {
    fn scatter(&mut self, _: &Ray, hit_info: &HitStatus) -> Option<ScatterInfo> {
        let mut scatter_direction: Vec3 = &hit_info.outward_normal + self.rand_sphere.gen_vec3();

        if scatter_direction.near_zero() {
            scatter_direction = hit_info.outward_normal.clone();
        }

        Some(ScatterInfo {
            ray: Ray::new(hit_info.point.clone(), scatter_direction),
            attenuation: self.albert.clone(),
        })
    }
}

pub struct Metal {
    albert: Color,
}

impl Metal {
    pub fn new(albert: Color) -> Self {
        Metal { albert: albert }
    }
}

impl Material for Metal {
    fn scatter(&mut self, r_in: &Ray, hit_info: &HitStatus) -> Option<ScatterInfo> {
        let reflected = r_in
            .direction
            .unit_vector()
            .reflect(&hit_info.outward_normal);
        let scattered = Ray::new(hit_info.point.clone(), reflected);
        if scattered.direction.dot(&hit_info.outward_normal) > 0.0 {
            Some(ScatterInfo {
                ray: scattered,
                attenuation: self.albert.clone(),
            })
        } else {
            None
        }
    }
}
