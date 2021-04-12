use float_cmp::{approx_eq, ApproxEq, F64Margin};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(PartialEq, Debug, Clone)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x: x, y: y, z: z }
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
}
