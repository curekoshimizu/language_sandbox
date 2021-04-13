#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};
    #[test]
    fn pcg_32() {
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = rand_pcg::Pcg32::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 1819159624);
        assert_eq!(rng.gen::<u64>(), 429246536024344263);

        let mut rng = rand_pcg::Pcg32::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 1819159624);
        assert_eq!(rng.gen::<u64>(), 429246536024344263);

        let mut rng = rand_pcg::Pcg32::from_entropy();
        assert_ne!(rng.gen::<u32>(), 4224100075);

        let uniform = Uniform::from(0.0..10.0);
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        assert!(0.0 <= x && x <= 10.0);
        assert!(0.0 <= y && y <= 10.0);
        assert_ne!(x, y);

        let z = rng.gen_range(0.0..10.0);
        assert!(0.0 <= z && z <= 10.0);
        assert_ne!(x, z);
        assert_ne!(y, z);
        let w = rng.gen::<f64>();
        assert_ne!(x, w);
        assert_ne!(y, w);
    }

    #[test]
    fn pcg_64_mcg() {
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 4125196422);
        assert_eq!(rng.gen::<u64>(), 3418526059257393691);

        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 4125196422);
        assert_eq!(rng.gen::<u64>(), 3418526059257393691);

        let mut rng = rand_pcg::Pcg64Mcg::from_entropy();
        assert_ne!(rng.gen::<u32>(), 4224100075);

        let uniform = Uniform::from(0.0..10.0);
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        assert!(0.0 <= x && x <= 10.0);
        assert!(0.0 <= y && y <= 10.0);
        assert_ne!(x, y);

        let z = rng.gen_range(0.0..10.0);
        assert!(0.0 <= z && z <= 10.0);
        assert_ne!(x, z);
        assert_ne!(y, z);
        let w = rng.gen::<f64>();
        assert_ne!(x, w);
        assert_ne!(y, w);
    }

    #[test]
    fn std_rng() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 4224100075);
        assert_eq!(rng.gen::<u64>(), 13541014214006172310);

        let mut rng = StdRng::seed_from_u64(123);
        assert_eq!(rng.gen::<u32>(), 4224100075);
        assert_eq!(rng.gen::<u64>(), 13541014214006172310);

        let mut rng = StdRng::from_entropy();
        assert_ne!(rng.gen::<u32>(), 4224100075);

        let uniform = Uniform::from(0.0..10.0);
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        assert!(0.0 <= x && x <= 10.0);
        assert!(0.0 <= y && y <= 10.0);
        assert_ne!(x, y);

        let z = rng.gen_range(0.0..10.0);
        assert!(0.0 <= z && z <= 10.0);
        assert_ne!(x, z);
        assert_ne!(y, z);
        let w = rng.gen::<f64>();
        assert_ne!(x, w);
        assert_ne!(y, w);
    }

    #[test]
    fn thread_local_rng() {
        use rand::thread_rng;
        use rand::Rng;

        let mut rng = thread_rng();

        let uniform = Uniform::from(0.0..10.0);
        let x = uniform.sample(&mut rng);
        let y = uniform.sample(&mut rng);
        assert!(0.0 <= x && x <= 10.0);
        assert!(0.0 <= y && y <= 10.0);
        assert_ne!(x, y);

        let z = rng.gen_range(0.0..10.0);
        assert!(0.0 <= z && z <= 10.0);
        assert_ne!(x, z);
        assert_ne!(y, z);
        let w = rng.gen::<f64>();
        assert_ne!(x, w);
        assert_ne!(y, w);
    }
}
