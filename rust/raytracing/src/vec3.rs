use float_cmp::{approx_eq, ApproxEq, F64Margin};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(PartialEq, Debug, Clone)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x: x, y: y, z: z }
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn length_squareed(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn dot(&self, rhs: &Vec3) -> Vec3 {
        self * rhs
    }
    pub fn cross(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z,
            y: self.z * rhs.x,
            z: self.x * rhs.y,
        }
    }
    pub fn unit_vector(&self) -> Vec3 {
        let len = self.length();
        self / &Vec3::new(len, len, len)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

impl ApproxEq for Vec3 {
    type Margin = F64Margin;
    fn approx_eq<M: Into<Self::Margin>>(self, other: Vec3, margin: M) -> bool {
        let margin = margin.into();

        self.x.approx_eq(other.x, margin)
            && self.y.approx_eq(other.y, margin)
            && self.z.approx_eq(other.z, margin)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<'a, 'b> Add<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn add(self, other: &'b Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.y + other.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<'a, 'b> Sub<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn sub(self, other: &'b Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.y - other.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl<'a, 'b> Mul<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn mul(self, other: &'b Vec3) -> Vec3 {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.y * other.z,
        }
    }
}

impl Div for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl<'a, 'b> Div<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn div(self, other: &'b Vec3) -> Vec3 {
        Vec3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.y / other.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl<'a> Neg for &'a Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        approx_eq!(Vec3, &a + &b, Vec3::new(1.0, 2.0, 3.0));
        approx_eq!(Vec3, a + b, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn sub() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        approx_eq!(Vec3, &a - &b, Vec3::new(1.0, 0.0, -1.0));
        approx_eq!(Vec3, a - b, Vec3::new(1.0, 0.0, -1.0));
    }

    #[test]
    fn mul() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        approx_eq!(Vec3, &a * &b, Vec3::new(0.0, 1.0, 2.0));
        approx_eq!(Vec3, a * b, Vec3::new(0.0, 1.0, 2.0));
    }

    #[test]
    fn div() {
        let b = Vec3::new(0.0, 1.0, 2.0);
        let a = Vec3::new(1.0, 2.0, 1.0);
        approx_eq!(Vec3, &b / &a, Vec3::new(0.0, 0.5, 2.0));
        approx_eq!(Vec3, b / a, Vec3::new(0.0, 0.5, 2.0));
    }

    #[test]
    fn neg() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        approx_eq!(Vec3, -&a, Vec3::new(0.0, -1.0, -2.0));
        approx_eq!(Vec3, -a, Vec3::new(0.0, -1.0, -2.0));
    }

    #[test]
    fn len() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        approx_eq!(f64, a.length(), 5.0_f64.sqrt());
        approx_eq!(f64, a.length_squareed(), 5.0);
    }

    #[test]
    fn dot() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        let b = Vec3::new(1.0, 1.0, 1.0);
        approx_eq!(Vec3, a.dot(&b), Vec3::new(0.0, 1.0, 2.0));
    }

    #[test]
    fn cross() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, 4.0, 5.0);
        approx_eq!(Vec3, a.cross(&b), Vec3::new(-2.0, 4.0, -2.0));
    }

    #[test]
    fn unit() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        approx_eq!(Vec3, a.unit_vector(), Vec3::new(-2.0, 4.0, -2.0));
    }

    #[test]
    fn display() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(format!("{}", a), "1 2 3");
    }
}
