use std::time::Instant;

#[derive(Debug)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Timer {
        Timer {
            start: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_millis();
        println!("elapsed time : {} msec!!!!!!", elapsed);
        assert!(elapsed >= 100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn it_works() {
        let t = Timer::new();
        thread::sleep(Duration::from_millis(100));
        println!("debug        : {:?}", t);
        println!("pretty debug : {:#?}", t);
    }
}
