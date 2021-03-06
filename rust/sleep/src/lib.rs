use std::thread;
use std::time::{Duration, Instant};

pub fn elapsed_time1(secs: u64) -> u64 {
    let start = Instant::now();

    thread::sleep(Duration::from_secs(secs));

    start.elapsed().as_secs()
}

pub fn elapsed_time2(millisec: u64) -> u128 {
    let start = Instant::now();

    thread::sleep(Duration::from_millis(millisec));

    (Instant::now() - start).as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep() {
        assert_eq!(elapsed_time1(1), 1);
        assert_eq!(elapsed_time2(100), 100);
    }
}
