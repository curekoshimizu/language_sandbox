use float_cmp::{ApproxEq, F64Margin};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(PartialEq, Debug, Clone)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub type Point3 = Vec3;

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
    pub fn add(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
    pub fn sub(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
    fn mul(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
    fn div(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
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
        <Vec3>::add(&self, &rhs)
    }
}

impl<'a> Add<&'a Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &Vec3) -> Self::Output {
        <Vec3>::add(&self, rhs)
    }
}

impl<'a> Add<Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        <Vec3>::add(self, &rhs)
    }
}

impl<'a, 'b> Add<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &'b Vec3) -> Vec3 {
        self.add(rhs)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Self::Output {
        <Vec3>::sub(&self, &rhs)
    }
}

impl<'a, 'b> Sub<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: &'b Vec3) -> Vec3 {
        self.sub(rhs)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        <Vec3>::mul(&self, &rhs)
    }
}

impl<'a, 'b> Mul<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: &'b Vec3) -> Vec3 {
        self.mul(rhs)
    }
}

impl Div for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: Vec3) -> Self::Output {
        <Vec3>::div(&self, &rhs)
    }
}

impl<'a, 'b> Div<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;
    fn div(self, rhs: &'b Vec3) -> Vec3 {
        self.div(rhs)
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
    use float_cmp::approx_eq;

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
}
