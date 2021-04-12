use crate::color::Color;
use float_cmp::{ApproxEq, F64Margin};
use std::ops::Deref;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(PartialEq, Debug, Clone)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<Color> for Vec3 {
    fn from(c: Color) -> Self {
        c.deref().clone()
    }
}

pub type Point3 = Vec3;

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    pub fn length(&self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn length_squared(&self) -> f64 {
        self.dot(self)
    }
    pub fn dot(&self, rhs: &Vec3) -> f64 {
        let v = self * rhs;
        v.x + v.y + v.z
    }
    pub fn cross(&self, rhs: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
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
    pub fn to_xyz(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    pub fn near_zero(&self) -> bool {
        let eps = 1e-8;

        (self.x.abs() < eps) && (self.y.abs() < eps) && (self.z.abs() < eps)
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

macro_rules! implement_binop {
    ($imp:ident, $method:ident) => {
        impl $imp<Vec3> for Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: Vec3) -> Self::Output {
                <Vec3>::$method(&self, &rhs)
            }
        }
        impl<'a> $imp<&'a Vec3> for Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: &Vec3) -> Self::Output {
                <Vec3>::$method(&self, rhs)
            }
        }
        impl<'a> $imp<Vec3> for &'a Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: Vec3) -> Self::Output {
                <Vec3>::$method(self, &rhs)
            }
        }
        impl<'a, 'b> $imp<&'b Vec3> for &'a Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: &'b Vec3) -> Vec3 {
                <Vec3>::$method(self, rhs)
            }
        }

        impl $imp<f64> for Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: f64) -> Self::Output {
                <Vec3>::$method(&self, &Vec3::new(rhs, rhs, rhs))
            }
        }
        impl<'a> $imp<f64> for &'a Vec3 {
            type Output = Vec3;
            fn $method(self, rhs: f64) -> Self::Output {
                <Vec3>::$method(self, &Vec3::new(rhs, rhs, rhs))
            }
        }
        impl $imp<Vec3> for f64 {
            type Output = Vec3;
            fn $method(self, rhs: Vec3) -> Self::Output {
                <Vec3>::$method(&Vec3::new(self, self, self), &rhs)
            }
        }
        impl<'a> $imp<&'a Vec3> for f64 {
            type Output = Vec3;
            fn $method(self, rhs: &Vec3) -> Self::Output {
                <Vec3>::$method(&Vec3::new(self, self, self), rhs)
            }
        }
    };
}

macro_rules! implement_assignop {
    ($imp:ident, $method:ident, $term:tt) => {
        impl $imp<Vec3> for Vec3 {
            fn $method(&mut self, rhs: Self) {
                *self = Self {
                    x: self.x $term rhs.x,
                    y: self.y $term rhs.y,
                    z: self.z $term rhs.z,
                };
            }
        }
        impl<'a> $imp<&'a Vec3> for Vec3 {
            fn $method(&mut self, rhs: &'a Vec3) {
                *self = Self {
                    x: self.x $term rhs.x,
                    y: self.y $term rhs.y,
                    z: self.z $term rhs.z,
                };
            }
        }
        impl $imp<f64> for Vec3 {
            fn $method(&mut self, rhs: f64) {
                *self = Self {
                    x: self.x $term rhs,
                    y: self.y $term rhs,
                    z: self.z $term rhs,
                };
            }
        }
    };
}

implement_binop! {Add, add}
implement_binop! {Sub, sub}
implement_binop! {Mul, mul}
implement_binop! {Div, div}

implement_assignop! {AddAssign, add_assign, +}
implement_assignop! {SubAssign, sub_assign, -}
implement_assignop! {MulAssign, mul_assign, *}
implement_assignop! {DivAssign, div_assign, /}

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
            z: -self.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn approx() {
        assert!(approx_eq!(
            Vec3,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(!approx_eq!(
            Vec3,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 2.0, 3.0)
        ));
    }

    #[test]
    fn add() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(Vec3, &a + &b, Vec3::new(1.0, 2.0, 3.0)));
        assert!(approx_eq!(Vec3, a + b, Vec3::new(1.0, 2.0, 3.0)));

        assert!(approx_eq!(
            Vec3,
            Vec3::new(0.0, 1.0, 2.0) + Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            Vec3::new(0.0, 1.0, 2.0) + &Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            &Vec3::new(0.0, 1.0, 2.0) + Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            &Vec3::new(0.0, 1.0, 2.0) + &Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            Vec3::new(0.0, 1.0, 2.0) + 1.0,
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            &Vec3::new(0.0, 1.0, 2.0) + 1.0,
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            1.0 + Vec3::new(0.0, 1.0, 2.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
        assert!(approx_eq!(
            Vec3,
            1.0 + &Vec3::new(0.0, 1.0, 2.0),
            Vec3::new(1.0, 2.0, 3.0)
        ));
    }

    #[test]
    fn sub() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(Vec3, &a - &b, Vec3::new(1.0, 0.0, -1.0)));
        assert!(approx_eq!(Vec3, a - b, Vec3::new(1.0, 0.0, -1.0)));
    }

    #[test]
    fn mul() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(Vec3, &a * &b, Vec3::new(0.0, 1.0, 2.0)));
        assert!(approx_eq!(Vec3, a * b, Vec3::new(0.0, 1.0, 2.0)));
    }

    #[test]
    fn div() {
        let b = Vec3::new(0.0, 1.0, 2.0);
        let a = Vec3::new(1.0, 2.0, 1.0);
        assert!(approx_eq!(Vec3, &b / &a, Vec3::new(0.0, 0.5, 2.0)));
        assert!(approx_eq!(Vec3, b / a, Vec3::new(0.0, 0.5, 2.0)));
    }

    #[test]
    fn add_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a += Vec3::new(-1.0, -1.0, -1.0);
        let mut b = a.clone();
        assert!(approx_eq!(Vec3, a, Vec3::new(0.0, 1.0, 2.0)));
        b += 3.0;
        assert!(approx_eq!(Vec3, b, Vec3::new(3.0, 4.0, 5.0)));
    }

    #[test]
    fn sub_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a -= Vec3::new(-1.0, -1.0, -1.0);
        let mut b = a.clone();
        assert!(approx_eq!(Vec3, a, Vec3::new(2.0, 3.0, 4.0)));
        b -= 3.0;
        assert!(approx_eq!(Vec3, b, Vec3::new(-1.0, 0.0, 1.0)));
    }

    #[test]
    fn mul_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a *= Vec3::new(-1.0, -1.0, -1.0);
        let mut b = a.clone();
        assert!(approx_eq!(Vec3, a, Vec3::new(-1.0, -2.0, -3.0)));
        b *= 3.0;
        assert!(approx_eq!(Vec3, b, Vec3::new(-3.0, -6.0, -9.0)));
    }

    #[test]
    fn div_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a /= Vec3::new(-1.0, -1.0, -1.0);
        let mut b = a.clone();
        assert!(approx_eq!(Vec3, a, Vec3::new(-1.0, -2.0, -3.0)));
        b /= 3.0;
        assert!(approx_eq!(Vec3, b, Vec3::new(-1.0 / 3.0, -2.0 / 3.0, -1.0)));
    }

    #[test]
    fn neg() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(Vec3, -&a, Vec3::new(0.0, -1.0, -2.0)));
        assert!(approx_eq!(Vec3, -a, Vec3::new(0.0, -1.0, -2.0)));
    }

    #[test]
    fn len() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        assert!(approx_eq!(f64, a.length(), 5.0_f64.sqrt()));
        assert!(approx_eq!(f64, a.length_squared(), 5.0));
    }

    #[test]
    fn dot() {
        let a = Vec3::new(0.0, 1.0, 2.0);
        let b = Vec3::new(1.0, 1.0, 2.0);
        assert!(approx_eq!(f64, a.dot(&b), 5.0));
    }

    #[test]
    fn cross() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, 4.0, 5.0);
        assert!(approx_eq!(Vec3, a.cross(&b), Vec3::new(-2.0, 4.0, -2.0)));
    }

    #[test]
    fn unit() {
        let a = Vec3::new(-3.0, 0.0, 4.0);
        assert!(approx_eq!(
            Vec3,
            a.unit_vector(),
            Vec3::new(-3.0 / 5.0, 0.0, 4.0 / 5.0)
        ));
    }
}
