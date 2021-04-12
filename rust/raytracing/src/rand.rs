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
}
