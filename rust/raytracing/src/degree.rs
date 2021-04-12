pub struct Degree(f64);

impl Degree {
    pub fn new(deg: f64) -> Self {
        Degree(deg)
    }

    pub fn degrees(&self) -> f64 {
        self.0
    }
    pub fn radians(&self) -> f64 {
        self.0 * std::f64::consts::PI / 180.0
    }
}
