use crate::rand::RandUniform;
use crate::vec3::Vec3;
use float_cmp::{ApproxEq, F64Margin};
use std::fmt;
use std::ops::Deref;
use std::ops::{Mul, MulAssign};

#[derive(Clone)]
pub struct Color(pub Vec3);

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
    pub fn random() -> Self {
        let uniform = RandUniform::new();
        Color(Vec3::new(uniform.gen(), uniform.gen(), uniform.gen()))
    }
}

impl From<Vec3> for Color {
    fn from(v: Vec3) -> Self {
        Color(v)
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // gamma-collect for gamma = 2.0
        let x = (255.999 * (self.0.x).sqrt()) as u64;
        let y = (255.999 * (self.0.y).sqrt()) as u64;
        let z = (255.999 * (self.0.z).sqrt()) as u64;
        write!(f, "{} {} {}", x, y, z)
    }
}

impl Mul<Color> for Color {
    type Output = Color;
    fn mul(self, rhs: Color) -> Self::Output {
        let v = Vec3::mul(self.0, &rhs.0);
        Color(v)
    }
}
impl MulAssign<Color> for Color {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl ApproxEq for Color {
    type Margin = F64Margin;
    fn approx_eq<M: Into<Self::Margin>>(self, other: Color, margin: M) -> bool {
        let margin = margin.into();

        self.0.approx_eq(other.0, margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn add() {
        let a = Color::new(1.0, 1.0, 1.0);
        let b = Color::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(
            Color,
            a.add(&b).into(),
            Color::new(1.0, 2.0, 3.0)
        ));
    }

    #[test]
    fn mul() {
        let a = Color::new(1.0, 1.0, 1.0);
        let b = Color::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(Color, a * b, Color::new(0.0, 1.0, 2.0)));
    }
    #[test]
    fn mul_assign() {
        let mut a = Color::new(1.0, 1.0, 1.0);
        let b = Color::new(0.0, 1.0, 2.0);
        a *= b;
        assert!(approx_eq!(Color, a, Color::new(0.0, 1.0, 2.0)));
    }

    #[test]
    fn display() {
        let color = Color::new(0.0, 0.5, 1.0);
        assert_eq!(format!("Color({})", color), "Color(0 181 255)");
    }
}
