use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::thread::JoinHandle;

struct RandGenerator<T> {
    rx: Receiver<T>,
    handle: Option<JoinHandle<()>>,
}

impl<T> RandGenerator<T> {
    fn gen(&self) -> T {
        self.rx.recv().unwrap()
    }
    fn stop(&mut self) {
        self.handle.take().map(JoinHandle::join);
        assert!(self.handle.is_none());
    }
}

impl<T> Drop for RandGenerator<T> {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct RandUniform(RandGenerator<f64>);

impl RandUniform {
    pub fn new(nsize: usize) -> Self {
        let (tx, rx) = mpsc::channel();

        RandUniform(RandGenerator {
            rx: rx,
            handle: Some(thread::spawn(move || {
                // random number generator thread
                let uniform_gen = Uniform::from(0.0..1.0);
                let mut rng = thread_rng();
                let rand_val = uniform_gen.sample(&mut rng);

                for _ in 0..nsize {
                    tx.send(rand_val).unwrap();
                }
            })),
        })
    }

    pub fn gen(&self) -> f64 {
        self.0.gen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_unitform() {
        {
            let uniform = RandUniform::new(1);
            let x = uniform.gen();
            assert!(0.0 <= x && x <= 1.0);
        }
        {
            let u = RandUniform::new(10);
            drop(u);
        }
    }
}
