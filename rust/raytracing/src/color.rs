use crate::vec3::Vec3;
use float_cmp::approx_eq;
use std::fmt;
use std::ops::Deref;

struct Color(Vec3);

impl Deref for Color {
    type Target = Vec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Color {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Color(Vec3::new(x, y, z))
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let x = (255.999 * self.x) as u64;
        let y = (255.999 * self.y) as u64;
        let z = (255.999 * self.z) as u64;
        write!(f, "{} {} {}", x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn add() {
    //     let a = Color::new(1.0, 1.0, 1.0);
    //     let b = Color::new(0.0, 1.0, 2.0);
    //     // approx_eq!(Vec3, &a + &b, Color::new(1.0, 2.0, 3.0));
    //     approx_eq!(Vec3, a + b, Color::new(1.0, 2.0, 3.0));
    // }

    #[test]
    fn display() {
        let color = Color::new(0.0, 0.5, 1.0);
        assert_eq!(format!("Color {}", color), "Color 0 127 255");
    }
}
