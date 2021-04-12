#[cfg(test)]
mod tests {
    fn tarai(x: i32, y: i32, z: i32) -> i32 {
        if x <= y {
            y
        } else {
            tarai(tarai(x - 1, y, z), tarai(y - 1, z, x), tarai(z - 1, x, y))
        }
    }

    #[test]
    fn hoge() {
        const LOOP: usize = 100;

        let elapsed;
        {
            let start = std::time::Instant::now();
            let x: Vec<i32> = (0..LOOP).map(|_| tarai(12, 5, 0)).collect();
            x.iter().for_each(|w| assert_eq!(w, &12));
            elapsed = start.elapsed().as_secs_f64();
            println!("one thread : {:?} secs", start.elapsed().as_secs_f64());
        }

        use rayon::prelude::*;
        {
            let start = std::time::Instant::now();
            let x: Vec<i32> = (0..LOOP)
                .collect::<Vec<usize>>()
                .par_iter()
                .map(|_| tarai(12, 5, 0))
                .collect();
            x.par_iter().for_each(|w| assert_eq!(w, &12));
            assert!(elapsed > start.elapsed().as_secs_f64());
            println!("multi-thread : {:?} secs", start.elapsed().as_secs_f64());
        }
    }
}
