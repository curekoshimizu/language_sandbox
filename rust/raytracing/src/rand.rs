use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::thread::JoinHandle;

pub struct RandUniform {
    rx: Receiver<f64>,
    pub handle: Option<JoinHandle<()>>,
}

impl RandUniform {
    pub fn new(nsize: usize) -> Self {
        let (rand_tx, rand_rx) = mpsc::channel();

        RandUniform {
            rx: rand_rx,
            handle: Some(thread::spawn(move || {
                // random number generator thread
                let uniform_gen = Uniform::from(0.0..1.0);
                let mut rng = thread_rng();
                let rand_val = uniform_gen.sample(&mut rng);

                for _ in 0..nsize {
                    rand_tx.send(rand_val).unwrap();
                }
            })),
        }
    }

    pub fn gen(&self) -> f64 {
        self.rx.recv().unwrap()
    }

    pub fn stop(&mut self) {
        self.handle.take().map(JoinHandle::join);
        assert!(self.handle.is_none());
    }
}

impl Drop for RandUniform {
    fn drop(&mut self) {
        println!("stopped");
        self.stop();
    }
}
