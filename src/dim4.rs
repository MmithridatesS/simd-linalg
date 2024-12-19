use std::simd::{f32x16, f32x4, num::SimdFloat, simd_swizzle, Simd};
use std::ops::Mul;

trait Dot<Rhs=Self> {
    type Output;
    fn dot(self, rhs: Rhs) -> Self::Output;
}

#[derive(PartialEq, Debug, Clone)]
pub struct Mat4(f32x16);

impl Mat4 {
    pub fn new<T: Into<f32>>(arr: [T; 16]) -> Self {
        let input: [f32; 16] = arr.map(|x| x.into());
        Self(Simd::from_array(input))
    }

    pub fn transpose(other: Mat4) -> Self {
        const TRAPO_IDCS: [usize; 16] = [0, 4, 8, 12,
                          1, 5, 9, 13, 
                          2, 6, 10, 14,
                          3, 7, 11, 15];
        let Mat4(data) = other;
        Self(simd_swizzle!(data, TRAPO_IDCS))
    }

    pub fn identity() -> Self {
        Mat4::new([1., 0., 0., 0.,
                   0., 1., 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.])
    }
    pub fn dot(first: Mat4, other: Mat4) -> Mat4 {
        let Mat4(mat1)= first;
        let mat1 = mat1.to_array();
        let Mat4(mat2) = Mat4::transpose(other);
        let mat2 = mat2.to_array();
        let mut res: Vec<f32> = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                let a: Simd<f32, 4> = Simd::from_slice(&mat1[(i)*4..(i+1)*4]);
                let b: Simd<f32, 4> = Simd::from_slice(&mat2[(j)*4..(j+1)*4]);
                res.push((a*b).reduce_sum());
            }
        }
        let arr: [f32; 16] = res.try_into().expect("this shouldnt fail");
        Self(Simd::from_array(arr))
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Vec4(f32x4);

impl Vec4 {
    pub fn new<T: Into<f32>>(arr: [T; 4]) -> Self {
        let input: [f32; 4] = arr.map(|x| x.into());
        Self(Simd::from_array(input))
    }
    pub fn from_simd(data: f32x4) -> Self {
        Self(data)
    }
    pub fn vec3_as_array(&self) -> [f32; 3] {
        let Vec4(inner) = self;
        [inner[0], inner[1], inner[2]]
    }

    pub fn vec4_as_array(&self) -> [f32; 4] {
        let Vec4(inner) = self;
        inner.to_array()
    }
    pub fn cross(&self, other: &Vec4) -> Self {
        let Vec4(me) = self;
        let Vec4(other) = other;
        let me_yzx = simd_swizzle!(me.clone().to_owned(), [1, 2, 0, 3]);
        let other_yzx = simd_swizzle!(other.clone().to_owned(), [1, 2, 0, 3]);
        let me_zxy = simd_swizzle!(me.clone().to_owned(), [2, 0, 1, 3]);
        let other_zxy = simd_swizzle!(other.clone().to_owned(), [2, 0, 1, 3]);
        Self::from_simd(me_yzx*other_zxy-me_zxy*other_yzx)
    }
}

impl Dot<Mat4> for Mat4 {
    type Output = Mat4;
    fn dot(self, rhs: Mat4) -> Mat4 {
        let Mat4(mat1)= self;
        let mat1 = mat1.to_array();
        let Mat4(mat2) = Mat4::transpose(rhs);
        let mat2 = mat2.to_array();
        let mut res: Vec<f32> = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                let a: Simd<f32, 4> = Simd::from_slice(&mat1[(i)*4..(i+1)*4]);
                let b: Simd<f32, 4> = Simd::from_slice(&mat2[(j)*4..(j+1)*4]);
                res.push((a*b).reduce_sum());
            }
        }
        let arr: [f32; 16] = res.try_into().expect("this shouldnt fail");
        Self(Simd::from_array(arr))
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;
    fn mul(self, rhs: Mat4) -> Self::Output {
        Dot::dot(self, rhs)
    }
}

impl Dot<Vec4> for Mat4 {
    type Output = Vec4;
    fn dot(self, rhs: Vec4) -> Self::Output {
        let Mat4(lhs) = self;
        let Vec4(rhs) = rhs;
        let one: Simd<f32, 4> = Simd::from_slice(&lhs[0..4]);
        let two: Simd<f32, 4> = Simd::from_slice(&lhs[4..8]);
        let three: Simd<f32, 4> = Simd::from_slice(&lhs[8..12]);
        let four: Simd<f32, 4> = Simd::from_slice(&lhs[12..16]);
        Vec4::new([(one*rhs).reduce_sum(), (two*rhs).reduce_sum(), (three*rhs).reduce_sum(), (four*rhs).reduce_sum()])
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        Dot::dot(self, rhs)
    }
}

impl Dot<Vec4> for Vec4 {
    type Output = f32;
    fn dot(self, rhs: Vec4) -> Self::Output {
        let Vec4(lhs) = self;
        let Vec4(rhs) = rhs;
        (lhs*rhs).reduce_sum()
    }
}

impl Mul<Vec4> for Vec4 {
    type Output = f32;
    fn mul(self, rhs: Vec4) -> Self::Output {
        Dot::dot(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let x: [u16; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let x = Mat4::new(x);
        assert_eq!(x, x);
        print!("{:?}", x);
    }

    #[test]
    fn identity_works() {
        let x: [f32; 16] = [1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.];
        let x = Mat4::new(x);
        assert_eq!(x, Mat4::identity());
        print!("{:?}", x);
    }

    #[test]
    fn mat4_dot_identity() {
        let x: [u16; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let x = Mat4::new(x);
        let y = Mat4::dot(x.clone(), Mat4::identity());
        assert_eq!(x, y);
    }

    #[test]
    fn mat4_dot_varied() {
        let x: [u16; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let x = Mat4::new(x);
        let y = Mat4::dot(x.clone(), x.clone());
        let res: [u16; 16] = [90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600];
        let res = Mat4::new(res);
        assert_eq!(y,res);
    }

    #[test]
    fn vec4_dot_zero_elemetns() {
        let x: [u16; 4] = [0, 0, 0, 0];
        let y: [u16; 4] = [1, 1, 1, 1];
        let r  = Dot::dot(Vec4::new(x), Vec4::new(y));
        assert_eq!(r, 0.);
    }

    #[test]
    fn vec4_dot_all_ones() {
        let x: [u16; 4] = [1, 1, 1, 1];
        let y: [u16; 4] = [1, 1, 1, 1];
        let r  = Dot::dot(Vec4::new(x), Vec4::new(y));
        assert_eq!(r, 4.);
    }

    #[test]
    fn two_vec4_dot_varied() {
        let x: [u16; 4] = [1, 2, 3, 4];
        let y: [u16; 4] = [1, 2, 3, 4];
        let r  = Dot::dot(Vec4::new(x), Vec4::new(y));
        assert_eq!(r, 30.);
    }

    #[test]
    fn two_vec4_dot_varied_2() {
        let x: [u16; 4] = [1, 2, 3, 4];
        let y: [u16; 4] = [2, 2, 2, 2];
        let r  = Dot::dot(Vec4::new(x), Vec4::new(y));
        assert_eq!(r, 20.);
    }

    #[test]
    fn mat_vec_dot_e_coord() {
        let x: [u16; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let y: [u16; 4] = [0, 0, 0, 1];
        let r  = Dot::dot(Mat4::new(x), Vec4::new(y));
        assert_eq!(r, Vec4::new([4u16, 8, 12, 16]));
    }

    #[test]
    fn mat4_dot_astrix_varied() {
        let x: [u16; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let x = Mat4::new(x);
        let y = x.clone() * x.clone();
        let res: [u16; 16] = [90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600];
        let res = Mat4::new(res);
        assert_eq!(y,res);
    }

    #[test]
    fn vec4_cross_for_coordinates() {
        let x = Vec4::new([1u16, 0, 0, 0]);
        let y = Vec4::new([0u16, 1, 0, 0]);
        let res = Vec4::new([0u16, 0, 1, 0]);
        assert_eq!(Vec4::cross(&x, &y),res);
    }

    #[test]
    fn vec4_cross_for_123456() {
        let x = Vec4::new([1u16, 2, 3, 0]);
        let y = Vec4::new([4u16, 5, 6, 0]);
        let res = Vec4::new([-3i16, 6, -3, 0]);
        assert_eq!(Vec4::cross(&x, &y),res);
    }
}
