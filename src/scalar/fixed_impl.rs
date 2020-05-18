#![allow(missing_docs)]

//! Implementation of traits form fixed-point numbers.
use crate::scalar::{ComplexField, Field, RealField, SubsetOf, SupersetOf};
use crate::simd::{PrimitiveSimdValue, SimdValue};
use fixed::types::extra::{
    IsLessOrEqual, LeEqU128, LeEqU16, LeEqU32, LeEqU64, LeEqU8, True, Unsigned, U13, U14, U16, U29,
    U30, U32, U5, U6, U61, U62, U64, U8,
};
use num::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use std::cmp::Ordering;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

macro_rules! impl_fixed_type(
    ($($FixedI: ident, $Int: ident, $LeEqDim: ident, $LeEqDim1: ident, $LeEqDim2: ident, $LeEqDim3: ident;)*) => {$(
        #[derive(Copy, Clone)]
        /// Signed fixed-point number with a generic number of bits for the fractional part.
        pub struct $FixedI<Fract: $LeEqDim>(pub fixed::$FixedI<Fract>);

        impl<Fract: $LeEqDim> PartialEq for $FixedI<Fract> {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<Fract: $LeEqDim> Eq for $FixedI<Fract> {}

        impl<Fract: $LeEqDim> PartialOrd for $FixedI<Fract> {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<Fract: $LeEqDim> PrimitiveSimdValue for $FixedI<Fract> {}
        impl<Fract: $LeEqDim> SimdValue for $FixedI<Fract> {
            type Element = Self;
            type SimdBool = bool;

            #[inline(always)]
            fn lanes() -> usize {
                1
            }

            #[inline(always)]
            fn splat(val: Self::Element) -> Self {
                val
            }

            #[inline(always)]
            fn extract(&self, _: usize) -> Self::Element {
                *self
            }

            #[inline(always)]
            unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
                *self
            }

            #[inline(always)]
            fn replace(&mut self, _: usize, val: Self::Element) {
                *self = val
            }

            #[inline(always)]
            unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
                *self = val
            }

            #[inline(always)]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                if cond {
                    self
                } else {
                    other
                }
            }
        }

        impl<Fract: $LeEqDim> Mul for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl<Fract: $LeEqDim> Div for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }

        impl<Fract: $LeEqDim> Rem for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn rem(self, rhs: Self) -> Self {
                Self(self.0 % rhs.0)
            }
        }

        impl<Fract: $LeEqDim> Add for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<Fract: $LeEqDim> Sub for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl<Fract: $LeEqDim> Neg for $FixedI<Fract> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl<Fract: $LeEqDim> MulAssign for $FixedI<Fract> {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0
            }
        }

        impl<Fract: $LeEqDim> DivAssign for $FixedI<Fract> {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0
            }
        }

        impl<Fract: $LeEqDim> RemAssign for $FixedI<Fract> {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: Self) {
                self.0 %= rhs.0
            }
        }

        impl<Fract: $LeEqDim> AddAssign for $FixedI<Fract> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0
            }
        }

        impl<Fract: $LeEqDim> SubAssign for $FixedI<Fract> {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0
            }
        }

        impl<Fract: $LeEqDim> Zero for $FixedI<Fract> {
            #[inline(always)]
            fn zero() -> Self {
                Self(fixed::$FixedI::from_num(0))
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.0 == Self::zero().0
            }
        }

        impl<Fract: $LeEqDim> One for $FixedI<Fract> {
            #[inline(always)]
            fn one() -> Self {
                Self(fixed::$FixedI::from_num(1))
            }
        }

        impl<Fract: $LeEqDim> Num for $FixedI<Fract> {
            type FromStrRadixErr = ();
            fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                unimplemented!()
            }
        }

        impl<Fract: $LeEqDim> Field for $FixedI<Fract> {}

        impl<Fract: $LeEqDim> SubsetOf<$FixedI<Fract>> for f64 {
            #[inline]
            fn to_superset(&self) -> $FixedI<Fract> {
                $FixedI(fixed::$FixedI::from_num(*self))
            }

            #[inline]
            fn from_superset(element: &$FixedI<Fract>) -> Option<Self> {
                Some(Self::from_superset_unchecked(element))
            }

            #[inline]
            fn from_superset_unchecked(element: &$FixedI<Fract>) -> Self {
                element.0.to_num::<f64>()
            }

            #[inline]
            fn is_in_subset(_: &$FixedI<Fract>) -> bool {
                true
            }
        }

        impl<Fract: $LeEqDim> SubsetOf<$FixedI<Fract>> for $FixedI<Fract> {
            #[inline]
            fn to_superset(&self) -> $FixedI<Fract> {
                *self
            }

            #[inline]
            fn from_superset(element: &$FixedI<Fract>) -> Option<Self> {
                Some(*element)
            }

            #[inline]
            fn from_superset_unchecked(element: &$FixedI<Fract>) -> Self {
                *element
            }

            #[inline]
            fn is_in_subset(_: &$FixedI<Fract>) -> bool {
                true
            }
        }

        impl<Fract: $LeEqDim> approx::AbsDiffEq for $FixedI<Fract> {
            type Epsilon = Self;
            fn default_epsilon() -> Self::Epsilon {
                Self(fixed::$FixedI::from_bits(0b01))
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                // This is the impl used in the approx crate.
                if self > other {
                    (*self - *other) <= epsilon
                } else {
                    (*other - *self) <= epsilon
                }
            }
        }

        impl<Fract: $LeEqDim> approx::RelativeEq for $FixedI<Fract> {
            fn default_max_relative() -> Self::Epsilon {
                use approx::AbsDiffEq;
                Self::default_epsilon()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool
            {
                // This is the impl used in the approx crate.
                let abs_diff = (*self - *other).abs();

                if abs_diff <= epsilon {
                    return true;
                }

                let abs_self = self.abs();
                let abs_other = other.abs();

                let largest = if abs_other > abs_self {
                    abs_other
                } else {
                    abs_self
                };

                abs_diff <= largest * max_relative
            }
        }

        impl<Fract: $LeEqDim> approx::UlpsEq for $FixedI<Fract> {
            fn default_max_ulps() -> u32 {
                4
            }

            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                use approx::AbsDiffEq;

                if self.abs_diff_eq(other, epsilon) {
                    return true;
                }

                if self.signum() != other.signum() {
                    return false;
                }

                let bits1 = self.0.to_bits();
                let bits2 = other.0.to_bits();

                if bits1 > bits2 {
                    (bits1 - bits2) <= max_ulps as $Int
                } else {
                    (bits2 - bits1) <= max_ulps as $Int
                }
            }
        }

        impl<Fract: $LeEqDim> std::fmt::Debug for $FixedI<Fract> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }

        impl<Fract: $LeEqDim> std::fmt::Display for $FixedI<Fract> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }

        impl<Fract: $LeEqDim> Bounded for $FixedI<Fract> {
            #[inline]
            fn min_value() -> Self {
                Self(fixed::$FixedI::MIN)
            }

            #[inline]
            fn max_value() -> Self {
                Self(fixed::$FixedI::MAX)
            }
        }

        impl<Fract: $LeEqDim> FromPrimitive for $FixedI<Fract> {
            fn from_i64(n: i64) -> Option<Self> {
                unimplemented!()
            }
            fn from_u64(n: u64) -> Option<Self> {
                unimplemented!()
            }
            fn from_isize(n: isize) -> Option<Self> {
                unimplemented!()
            }
            fn from_i8(n: i8) -> Option<Self> {
                unimplemented!()
            }
            fn from_i16(n: i16) -> Option<Self> {
                unimplemented!()
            }
            fn from_i32(n: i32) -> Option<Self> {
                unimplemented!()
            }
            fn from_usize(n: usize) -> Option<Self> {
                unimplemented!()
            }
            fn from_u8(n: u8) -> Option<Self> {
                unimplemented!()
            }
            fn from_u16(n: u16) -> Option<Self> {
                unimplemented!()
            }
            fn from_u32(n: u32) -> Option<Self> {
                unimplemented!()
            }
            fn from_f32(n: f32) -> Option<Self> {
                unimplemented!()
            }
            fn from_f64(n: f64) -> Option<Self> {
                unimplemented!()
            }
        }

        impl<Fract: $LeEqDim> Signed for $FixedI<Fract> {
            fn abs(&self) -> Self {
                Self(self.0.abs())
            }

            fn abs_sub(&self, other: &Self) -> Self {
                unimplemented!()
            }

            fn signum(&self) -> Self {
                Self(self.0.signum())
            }

            fn is_positive(&self) -> bool {
                self.0 >= Self::zero().0
            }

            fn is_negative(&self) -> bool {
                self.0 <= Self::zero().0
            }
        }

        impl<Fract: Send + Sync + 'static> ComplexField for $FixedI<Fract>
            where Fract: Unsigned
                    + IsLessOrEqual<$LeEqDim1, Output = True>
                    + IsLessOrEqual<$LeEqDim2, Output = True>
                    + IsLessOrEqual<$LeEqDim3, Output = True> {
            type RealField = Self;

            #[inline]
            fn from_real(re: Self::RealField) -> Self {
                re
            }

            #[inline]
            fn real(self) -> Self::RealField {
                self
            }

            #[inline]
            fn imaginary(self) -> Self::RealField {
                Self::zero()
            }

            #[inline]
            fn norm1(self) -> Self::RealField {
                self.abs()
            }

            #[inline]
            fn modulus(self) -> Self::RealField {
                self.abs()
            }

            #[inline]
            fn modulus_squared(self) -> Self::RealField {
                self * self
            }

            #[inline]
            fn argument(self) -> Self::RealField {
                if self >= Self::zero() {
                    Self::zero()
                } else {
                    Self::pi()
                }
            }

            #[inline]
            fn to_exp(self) -> (Self, Self) {
                if self >= Self::zero() {
                    (self, Self::one())
                } else {
                    (-self, -Self::one())
                }
            }

            #[inline]
            fn recip(self) -> Self {
                Self::one() / self
            }

            #[inline]
            fn conjugate(self) -> Self {
                self
            }

            #[inline]
            fn scale(self, factor: Self::RealField) -> Self {
                self * factor
            }

            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self {
                self / factor
            }

            #[inline]
            fn floor(self) -> Self {
                Self(self.0.floor())
            }

            #[inline]
            fn ceil(self) -> Self {
                Self(self.0.ceil())
            }

            #[inline]
            fn round(self) -> Self {
                Self(self.0.round())
            }

            #[inline]
            fn trunc(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn fract(self) -> Self {
                Self(self.0.frac())
            }

            #[inline]
            fn abs(self) -> Self {
                Self(self.0.abs())
            }

            #[inline]
            fn signum(self) -> Self {
                Self(self.0.signum())
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                self * a + b
            }

            #[cfg(feature = "std")]
            #[inline]
            fn powi(self, n: i32) -> Self {
                unimplemented!()
            }

            #[cfg(not(feature = "std"))]
            #[inline]
            fn powi(self, n: i32) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powf(self, n: Self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powc(self, n: Self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn sqrt(self) -> Self {
                Self(fixed_trig::cordic::sqrt(self.0, 64)) // FIXME: let the user choose the number of iterations somehow.
            }

            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                if self >= Self::zero() {
                    Some(self.sqrt())
                } else {
                    None
                }
            }

            #[inline]
            fn exp(self) -> Self {
                Self(fixed_trig::cordic::exp(self.0))
            }

            #[inline]
            fn exp2(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn exp_m1(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn ln_1p(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn ln(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn log(self, base: Self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn log2(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn log10(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn cbrt(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                unimplemented!()
            }

            #[inline]
            fn sin(self) -> Self {
                Self(fixed_trig::cordic::sin(self.0))
            }

            #[inline]
            fn cos(self) -> Self {
                Self(fixed_trig::cordic::cos(self.0))
            }

            #[inline]
            fn tan(self) -> Self {
                Self(fixed_trig::cordic::tan(self.0))
            }

            #[inline]
            fn asin(self) -> Self {
                Self(fixed_trig::cordic::asin(self.0))
            }

            #[inline]
            fn acos(self) -> Self {
                Self(fixed_trig::cordic::acos(self.0))
            }

            #[inline]
            fn atan(self) -> Self {
                Self(fixed_trig::cordic::atan(self.0))
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = fixed_trig::cordic::sin_cos(self.0);
                (Self(sin), Self(cos))
            }

            #[inline]
            fn sinh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn cosh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn tanh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn asinh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn acosh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn atanh(self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn is_finite(&self) -> bool {
                unimplemented!()
            }
        }

        impl<Fract: Send + Sync + 'static> RealField for $FixedI<Fract>
            where Fract: Unsigned
                    + IsLessOrEqual<$LeEqDim1, Output = True>
                    + IsLessOrEqual<$LeEqDim2, Output = True>
                    + IsLessOrEqual<$LeEqDim3, Output = True> {
            #[inline]
            fn is_sign_positive(self) -> bool {
                unimplemented!()
            }

            #[inline]
            fn is_sign_negative(self) -> bool {
                unimplemented!()
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                if self < other {
                    self
                } else {
                    other
                }
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                if self < min {
                    min
                } else if self > max {
                    max
                } else {
                    self
                }
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                Self(fixed_trig::cordic::atan2(self.0, other.0))
            }

            /// Archimedes' constant.
            #[inline]
            fn pi() -> Self {
                Self(fixed::$FixedI::PI)
            }

            /// 2.0 * pi.
            #[inline]
            fn two_pi() -> Self {
                unimplemented!()
            }

            /// pi / 2.0.
            #[inline]
            fn frac_pi_2() -> Self {
                Self(fixed::$FixedI::FRAC_PI_2)
            }

            /// pi / 3.0.
            #[inline]
            fn frac_pi_3() -> Self {
                Self(fixed::$FixedI::FRAC_PI_3)
            }

            /// pi / 4.0.
            #[inline]
            fn frac_pi_4() -> Self {
                unimplemented!()
            }

            /// pi / 6.0.
            #[inline]
            fn frac_pi_6() -> Self {
                unimplemented!()
            }

            /// pi / 8.0.
            #[inline]
            fn frac_pi_8() -> Self {
                Self(fixed::$FixedI::FRAC_PI_8)
            }

            /// 1.0 / pi.
            #[inline]
            fn frac_1_pi() -> Self {
                Self(fixed::$FixedI::FRAC_1_PI)
            }

            /// 2.0 / pi.
            #[inline]
            fn frac_2_pi() -> Self {
                unimplemented!()
            }

            /// 2.0 / sqrt(pi).
            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                unimplemented!()
            }

            /// Euler's number.
            #[inline]
            fn e() -> Self {
                unimplemented!()
            }

            /// log2(e).
            #[inline]
            fn log2_e() -> Self {
                unimplemented!()
            }

            /// log10(e).
            #[inline]
            fn log10_e() -> Self {
                unimplemented!()
            }

            /// ln(2.0).
            #[inline]
            fn ln_2() -> Self {
                unimplemented!()
            }

            /// ln(10.0).
            #[inline]
            fn ln_10() -> Self {
                unimplemented!()
            }
        }
    )*}
);

impl_fixed_type!(
    FixedI8, i8, LeEqU8, U8, U6, U5;
    FixedI16, i16, LeEqU16, U16, U14, U13;
    FixedI32, i32, LeEqU32, U32, U30, U29;
    FixedI64, i64, LeEqU64, U64, U62, U61;
);

pub type FixedI8F24 = FixedI32<fixed::types::extra::U24>;
pub type FixedI16F16 = FixedI32<fixed::types::extra::U16>;
pub type FixedI32F32 = FixedI64<fixed::types::extra::U32>;
pub type FixedI40F24 = FixedI64<fixed::types::extra::U24>;
pub type FixedI48F16 = FixedI64<fixed::types::extra::U16>;
