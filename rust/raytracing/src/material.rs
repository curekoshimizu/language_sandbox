use crate::color::Color;
use crate::hittable::HitStatus;
use crate::rand::BallUniform;
use crate::rand::RandUniform;
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
            rand_sphere: SphereUniform::new(),
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
    rand_sphere: BallUniform,
    albert: Color,
    fuzz: f64,
}

impl Metal {
    pub fn new(albert: Color, fuzz: f64) -> Self {
        Metal {
            albert,
            fuzz,
            rand_sphere: BallUniform::new(),
        }
    }
}

impl Material for Metal {
    fn scatter(&mut self, r_in: &Ray, hit_info: &HitStatus) -> Option<ScatterInfo> {
        let reflected = r_in
            .direction
            .unit_vector()
            .reflect(&hit_info.outward_normal)
            + self.fuzz * self.rand_sphere.gen_vec3();
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

pub struct Dielectric {
    rand_unitform: RandUniform,
    index_refraction: f64,
}

impl Dielectric {
    pub fn new(index_refraction: f64) -> Self {
        Dielectric {
            rand_unitform: RandUniform::new(),
            index_refraction,
        }
    }

    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
    }
}

impl Material for Dielectric {
    fn scatter(&mut self, r_in: &Ray, hit_info: &HitStatus) -> Option<ScatterInfo> {
        let refraction_ratio = if hit_info.front_face {
            1.0 / self.index_refraction
        } else {
            self.index_refraction
        };

        let unit_direction = r_in.direction.unit_vector();

        let cos_theta = (-unit_direction.clone())
            .dot(&hit_info.outward_normal)
            .min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction: Vec3;
        let random_double = self.rand_unitform.gen() * 2.0 - 1.0;
        if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > random_double {
            direction = unit_direction.reflect(&hit_info.outward_normal)
        } else {
            direction = unit_direction.refract(&hit_info.outward_normal, refraction_ratio);
        }

        Some(ScatterInfo {
            ray: Ray::new(hit_info.point.clone(), direction),
            attenuation: Color::new(1.0, 1.0, 1.0),
        })
    }
}
