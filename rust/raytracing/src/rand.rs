use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::thread_rng;

pub struct RandUniform {
    uniform: Uniform<f64>,
    rng: ThreadRng,
}

impl RandUniform {
    pub fn new() -> Self {
        RandUniform {
            uniform: Uniform::from(0.0..1.0),
            rng: thread_rng(),
        }
    }

    pub fn gen(&mut self) -> f64 {
        self.uniform.sample(&mut self.rng)
    }
}

pub struct BallUniform(RandUniform);

impl BallUniform {
    pub fn new() -> Self {
        BallUniform(RandUniform::new())
    }

    pub fn gen(&mut self) -> (f64, f64, f64) {
        let cos_theta = -2.0 * self.0.gen() + 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * std::f64::consts::PI * self.0.gen();
        let radius = self.0.gen().powf(1.0 / 3.0);

        (
            radius * sin_theta * phi.cos(),
            radius * sin_theta * phi.sin(),
            radius * cos_theta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_unitform() {
        {
            let mut uniform = RandUniform::new();
            let x = uniform.gen();
            let y = uniform.gen();
            assert!(0.0 <= x && x <= 1.0);
            assert!(0.0 <= y && y <= 1.0);
            assert_ne!(x, y);
        }
    }

    #[test]
    fn rand_ball() {
        {
            let mut uniform = BallUniform::new();
            let (x, y, z) = uniform.gen();
            assert!(-1.0 <= x && x <= 1.0);
            assert!(-1.0 <= y && y <= 1.0);
            assert!(-1.0 <= z && z <= 1.0);
            assert!(x * x + y * y + z * z <= 1.0);
            assert_ne!(x, y);
        }
    }
}
