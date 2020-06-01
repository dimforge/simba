#![allow(missing_docs)]

//! Implementation of traits form fixed-point numbers.
use crate::scalar::{ComplexField, Field, RealField, SubsetOf};
use crate::simd::{PrimitiveSimdValue, SimdValue};
use fixed::types::extra::{
    IsLessOrEqual, LeEqU16, LeEqU32, LeEqU64, LeEqU8, True, Unsigned, U13, U14, U16, U29, U30, U32,
    U5, U6, U61, U62, U64, U8,
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

        #[cfg(feature = "rand")]
        impl<Fract: $LeEqDim> rand::distributions::Distribution<$FixedI<Fract>> for rand::distributions::Standard {
            #[inline]
            fn sample<'a, G: rand::Rng + ?Sized>(&self, rng: &mut G) -> $FixedI<Fract> {
                let bits = rng.gen();
                $FixedI(fixed::$FixedI::from_bits(bits))
            }
        }

        #[cfg(feature = "rand")]
        impl<Fract: $LeEqDim> rand::distributions::Distribution<$FixedI<Fract>> for rand::distributions::OpenClosed01 {
            #[inline]
            fn sample<'a, G: rand::Rng + ?Sized>(&self, rng: &mut G) -> $FixedI<Fract> {
                let val: f64 = rng.gen();
                $FixedI(fixed::$FixedI::from_num(val))
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
            fn from_i64(_n: i64) -> Option<Self> {
                unimplemented!()
            }
            fn from_u64(_n: u64) -> Option<Self> {
                unimplemented!()
            }
            fn from_isize(_n: isize) -> Option<Self> {
                unimplemented!()
            }
            fn from_i8(_n: i8) -> Option<Self> {
                unimplemented!()
            }
            fn from_i16(_n: i16) -> Option<Self> {
                unimplemented!()
            }
            fn from_i32(_n: i32) -> Option<Self> {
                unimplemented!()
            }
            fn from_usize(_n: usize) -> Option<Self> {
                unimplemented!()
            }
            fn from_u8(_n: u8) -> Option<Self> {
                unimplemented!()
            }
            fn from_u16(_n: u16) -> Option<Self> {
                unimplemented!()
            }
            fn from_u32(_n: u32) -> Option<Self> {
                unimplemented!()
            }
            fn from_f32(_n: f32) -> Option<Self> {
                unimplemented!()
            }
            fn from_f64(_n: f64) -> Option<Self> {
                unimplemented!()
            }
        }

        impl<Fract: $LeEqDim> Signed for $FixedI<Fract> {
            fn abs(&self) -> Self {
                Self(self.0.abs())
            }

            fn abs_sub(&self, other: &Self) -> Self {
                self.abs() - *other
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
            fn powi(self, _n: i32) -> Self {
                unimplemented!()
            }

            #[cfg(not(feature = "std"))]
            #[inline]
            fn powi(self, n: i32) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powf(self, _n: Self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powc(self, _n: Self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn sqrt(self) -> Self {
                Self(cordic::sqrt(self.0))
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
                Self(cordic::exp(self.0))
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
            fn log(self, _base: Self) -> Self {
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
            fn hypot(self, _other: Self) -> Self::RealField {
                unimplemented!()
            }

            #[inline]
            fn sin(self) -> Self {
                Self(cordic::sin(self.0))
            }

            #[inline]
            fn cos(self) -> Self {
                Self(cordic::cos(self.0))
            }

            #[inline]
            fn tan(self) -> Self {
                Self(cordic::tan(self.0))
            }

            #[inline]
            fn asin(self) -> Self {
                Self(cordic::asin(self.0))
            }

            #[inline]
            fn acos(self) -> Self {
                Self(cordic::acos(self.0))
            }

            #[inline]
            fn atan(self) -> Self {
                Self(cordic::atan(self.0))
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = cordic::sin_cos(self.0);
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
                Self(cordic::atan2(self.0, other.0))
            }

            /// Archimedes' constant.
            #[inline]
            fn pi() -> Self {
                Self(fixed::$FixedI::PI)
            }

            /// 2.0 * pi.
            #[inline]
            fn two_pi() -> Self {
                Self::pi() + Self::pi()
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

pub type FixedI8F0 = FixedI8<fixed::types::extra::U0>;
pub type FixedI7F1 = FixedI8<fixed::types::extra::U1>;
pub type FixedI6F2 = FixedI8<fixed::types::extra::U2>;
pub type FixedI5F3 = FixedI8<fixed::types::extra::U3>;
pub type FixedI4F4 = FixedI8<fixed::types::extra::U4>;
pub type FixedI3F5 = FixedI8<fixed::types::extra::U5>;

pub type FixedI16F0 = FixedI16<fixed::types::extra::U0>;
pub type FixedI15F1 = FixedI16<fixed::types::extra::U1>;
pub type FixedI14F2 = FixedI16<fixed::types::extra::U2>;
pub type FixedI13F3 = FixedI16<fixed::types::extra::U3>;
pub type FixedI12F4 = FixedI16<fixed::types::extra::U4>;
pub type FixedI11F5 = FixedI16<fixed::types::extra::U5>;
pub type FixedI10F6 = FixedI16<fixed::types::extra::U6>;
pub type FixedI9F7 = FixedI16<fixed::types::extra::U7>;
pub type FixedI8F8 = FixedI16<fixed::types::extra::U8>;
pub type FixedI7F9 = FixedI16<fixed::types::extra::U9>;
pub type FixedI6F10 = FixedI16<fixed::types::extra::U10>;
pub type FixedI5F11 = FixedI16<fixed::types::extra::U11>;
pub type FixedI4F12 = FixedI16<fixed::types::extra::U12>;
pub type FixedI3F13 = FixedI16<fixed::types::extra::U13>;

pub type FixedI32F0 = FixedI32<fixed::types::extra::U0>;
pub type FixedI31F1 = FixedI32<fixed::types::extra::U1>;
pub type FixedI30F2 = FixedI32<fixed::types::extra::U2>;
pub type FixedI29F3 = FixedI32<fixed::types::extra::U3>;
pub type FixedI28F4 = FixedI32<fixed::types::extra::U4>;
pub type FixedI27F5 = FixedI32<fixed::types::extra::U5>;
pub type FixedI26F6 = FixedI32<fixed::types::extra::U6>;
pub type FixedI25F7 = FixedI32<fixed::types::extra::U7>;
pub type FixedI24F8 = FixedI32<fixed::types::extra::U8>;
pub type FixedI23F9 = FixedI32<fixed::types::extra::U9>;
pub type FixedI22F10 = FixedI32<fixed::types::extra::U10>;
pub type FixedI21F11 = FixedI32<fixed::types::extra::U11>;
pub type FixedI20F12 = FixedI32<fixed::types::extra::U12>;
pub type FixedI19F13 = FixedI32<fixed::types::extra::U13>;
pub type FixedI18F14 = FixedI32<fixed::types::extra::U14>;
pub type FixedI17F15 = FixedI32<fixed::types::extra::U15>;
pub type FixedI16F16 = FixedI32<fixed::types::extra::U16>;
pub type FixedI15F17 = FixedI32<fixed::types::extra::U17>;
pub type FixedI14F18 = FixedI32<fixed::types::extra::U18>;
pub type FixedI13F19 = FixedI32<fixed::types::extra::U19>;
pub type FixedI12F20 = FixedI32<fixed::types::extra::U20>;
pub type FixedI11F21 = FixedI32<fixed::types::extra::U21>;
pub type FixedI10F22 = FixedI32<fixed::types::extra::U22>;
pub type FixedI9F23 = FixedI32<fixed::types::extra::U23>;
pub type FixedI8F24 = FixedI32<fixed::types::extra::U24>;
pub type FixedI7F25 = FixedI32<fixed::types::extra::U25>;
pub type FixedI6F26 = FixedI32<fixed::types::extra::U26>;
pub type FixedI5F27 = FixedI32<fixed::types::extra::U27>;
pub type FixedI4F28 = FixedI32<fixed::types::extra::U28>;
pub type FixedI3F29 = FixedI32<fixed::types::extra::U29>;

pub type FixedI64F0 = FixedI64<fixed::types::extra::U0>;
pub type FixedI63F1 = FixedI64<fixed::types::extra::U1>;
pub type FixedI62F2 = FixedI64<fixed::types::extra::U2>;
pub type FixedI61F3 = FixedI64<fixed::types::extra::U3>;
pub type FixedI60F4 = FixedI64<fixed::types::extra::U4>;
pub type FixedI59F5 = FixedI64<fixed::types::extra::U5>;
pub type FixedI58F6 = FixedI64<fixed::types::extra::U6>;
pub type FixedI57F7 = FixedI64<fixed::types::extra::U7>;
pub type FixedI56F8 = FixedI64<fixed::types::extra::U8>;
pub type FixedI55F9 = FixedI64<fixed::types::extra::U9>;
pub type FixedI54F10 = FixedI64<fixed::types::extra::U10>;
pub type FixedI53F11 = FixedI64<fixed::types::extra::U11>;
pub type FixedI52F12 = FixedI64<fixed::types::extra::U12>;
pub type FixedI51F13 = FixedI64<fixed::types::extra::U13>;
pub type FixedI50F14 = FixedI64<fixed::types::extra::U14>;
pub type FixedI49F15 = FixedI64<fixed::types::extra::U15>;
pub type FixedI48F16 = FixedI64<fixed::types::extra::U16>;
pub type FixedI47F17 = FixedI64<fixed::types::extra::U17>;
pub type FixedI46F18 = FixedI64<fixed::types::extra::U18>;
pub type FixedI45F19 = FixedI64<fixed::types::extra::U19>;
pub type FixedI44F20 = FixedI64<fixed::types::extra::U20>;
pub type FixedI43F21 = FixedI64<fixed::types::extra::U21>;
pub type FixedI42F22 = FixedI64<fixed::types::extra::U22>;
pub type FixedI41F23 = FixedI64<fixed::types::extra::U23>;
pub type FixedI40F24 = FixedI64<fixed::types::extra::U24>;
pub type FixedI39F25 = FixedI64<fixed::types::extra::U25>;
pub type FixedI38F26 = FixedI64<fixed::types::extra::U26>;
pub type FixedI37F27 = FixedI64<fixed::types::extra::U27>;
pub type FixedI36F28 = FixedI64<fixed::types::extra::U28>;
pub type FixedI35F29 = FixedI64<fixed::types::extra::U29>;
pub type FixedI34F30 = FixedI64<fixed::types::extra::U30>;
pub type FixedI33F31 = FixedI64<fixed::types::extra::U31>;
pub type FixedI32F32 = FixedI64<fixed::types::extra::U32>;
pub type FixedI31F33 = FixedI64<fixed::types::extra::U33>;
pub type FixedI30F34 = FixedI64<fixed::types::extra::U34>;
pub type FixedI29F35 = FixedI64<fixed::types::extra::U35>;
pub type FixedI28F36 = FixedI64<fixed::types::extra::U36>;
pub type FixedI27F37 = FixedI64<fixed::types::extra::U37>;
pub type FixedI26F38 = FixedI64<fixed::types::extra::U38>;
pub type FixedI25F39 = FixedI64<fixed::types::extra::U39>;
pub type FixedI24F40 = FixedI64<fixed::types::extra::U40>;
pub type FixedI23F41 = FixedI64<fixed::types::extra::U41>;
pub type FixedI22F42 = FixedI64<fixed::types::extra::U42>;
pub type FixedI21F43 = FixedI64<fixed::types::extra::U43>;
pub type FixedI20F44 = FixedI64<fixed::types::extra::U44>;
pub type FixedI19F45 = FixedI64<fixed::types::extra::U45>;
pub type FixedI18F46 = FixedI64<fixed::types::extra::U46>;
pub type FixedI17F47 = FixedI64<fixed::types::extra::U47>;
pub type FixedI16F48 = FixedI64<fixed::types::extra::U48>;
pub type FixedI15F49 = FixedI64<fixed::types::extra::U49>;
pub type FixedI14F50 = FixedI64<fixed::types::extra::U50>;
pub type FixedI13F51 = FixedI64<fixed::types::extra::U51>;
pub type FixedI12F52 = FixedI64<fixed::types::extra::U52>;
pub type FixedI11F53 = FixedI64<fixed::types::extra::U53>;
pub type FixedI10F54 = FixedI64<fixed::types::extra::U54>;
pub type FixedI9F55 = FixedI64<fixed::types::extra::U55>;
pub type FixedI8F56 = FixedI64<fixed::types::extra::U56>;
pub type FixedI7F57 = FixedI64<fixed::types::extra::U57>;
pub type FixedI6F58 = FixedI64<fixed::types::extra::U58>;
pub type FixedI5F59 = FixedI64<fixed::types::extra::U59>;
pub type FixedI4F60 = FixedI64<fixed::types::extra::U60>;
