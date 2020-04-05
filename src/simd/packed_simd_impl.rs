#![allow(missing_docs)]
#![allow(non_camel_case_types)] // For the simd type aliases.

//! Traits for SIMD values.

use crate::scalar::{ComplexField, Field, SubsetOf, SupersetOf};
use crate::simd::{
    PrimitiveSimdValue, SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdSigned,
    SimdValue,
};
use approx::AbsDiffEq;
#[cfg(feature = "decimal")]
use decimal::d128;
use num::{FromPrimitive, Num, One, Zero};
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

/// An Simd structure that implements all the relevant traits from `num` an `simba`.
///
/// This is needed to overcome the orphan rules.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Simd<N: SimdValue>(pub N);

impl<N: SimdValue + Copy> fmt::Display for Simd<N>
where
    N::Element: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if N::lanes() == 1 {
            return self.extract(0).fmt(f);
        }

        write!(f, "({}", self.extract(0))?;

        for i in 1..N::lanes() {
            write!(f, ", {}", self.extract(i))?;
        }

        write!(f, ")")
    }
}

impl<N: PrimitiveSimdValue> PrimitiveSimdValue for Simd<N> {}

impl<N: SimdValue> SimdValue for Simd<N> {
    type Element = N::Element;
    type SimdBool = N::SimdBool;

    #[inline(always)]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        Simd(N::splat(val))
    }

    #[inline(always)]
    fn extract(&self, i: usize) -> Self::Element {
        self.0.extract(i)
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        self.0.extract_unchecked(i)
    }

    #[inline(always)]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.0.replace(i, val);
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.0.replace_unchecked(i, val);
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Simd(self.0.select(cond, other.0))
    }
}

macro_rules! impl_simd_bool(
    ($($t: ty;)*) => {$(
        impl SimdBool for $t {
            #[inline(always)]
            fn and(self) -> bool {
                self.and()
            }

            #[inline(always)]
            fn or(self) -> bool {
                self.or()
            }

            #[inline(always)]
            fn xor(self) -> bool {
                self.xor()
            }

            #[inline(always)]
            fn all(self) -> bool {
                self.all()
            }

            #[inline(always)]
            fn any(self) -> bool {
                self.any()
            }

            #[inline(always)]
            fn none(self) -> bool {
                self.none()
            }

            #[inline(always)]
            fn if_else<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_value();
                a.select(self, b)
            }

            #[inline(always)]
            fn if_else2<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_if.1();
                let c = else_value();

                let cond_a = self;
                let cond_b = else_if.0();

                a.select(cond_a, b.select(cond_b, c))
            }

            #[inline(always)]
            fn if_else3<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_if.1();
                let c = else_else_if.1();
                let d = else_value();

                let cond_a = self;
                let cond_b = else_if.0();
                let cond_c = else_else_if.0();

                a.select(cond_a, b.select(cond_b, c.select(cond_c, d)))
            }
        }
    )*}
);

macro_rules! impl_scalar_subset_of_simd(
    ($($t: ty),*) => {$(
        impl<N2: SimdValue + Copy> SubsetOf<Simd<N2>> for $t
            where N2::Element: SupersetOf<$t> + PartialEq {
            #[inline(always)]
            fn to_superset(&self) -> Simd<N2> {
                Simd(N2::splat(N2::Element::from_subset(self)))
            }

            #[inline(always)]
            fn from_superset_unchecked(element: &Simd<N2>) -> $t {
                element.extract(0).to_subset_unchecked()
            }

            #[inline(always)]
            fn is_in_subset(c: &Simd<N2>) -> bool {
                let elt0 = c.extract(0);
                elt0.is_in_subset() &&
                (1..N2::lanes()).all(|i| c.extract(i) == elt0)
            }
        }
    )*}
);

impl_scalar_subset_of_simd!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
#[cfg(feature = "decimal")]
impl_scalar_subset_of_simd!(d128);

macro_rules! impl_simd_value(
    ($($t: ty, $elt: ty, $bool: ty;)*) => ($(
        impl PrimitiveSimdValue for $t {}

        impl SimdValue for $t {
            type Element = $elt;
            type SimdBool = $bool;

            #[inline(always)]
            fn lanes() -> usize {
                <$t>::lanes()
            }

            #[inline(always)]
            fn splat(val: Self::Element) -> Self {
                <$t>::splat(val)
            }

            #[inline(always)]
            fn extract(&self, i: usize) -> Self::Element {
                <$t>::extract(*self, i)
            }

            #[inline(always)]
            unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
                <$t>::extract_unchecked(*self, i)
            }

            #[inline(always)]
            fn replace(&mut self, i: usize, val: Self::Element) {
                *self = <$t>::replace(*self, i, val)
            }

            #[inline(always)]
            unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
                *self = <$t>::replace_unchecked(*self, i, val)
            }

            #[inline(always)]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                cond.select(self, other)
            }
        }
    )*)
);

macro_rules! impl_uint_simd(
    ($($t: ty, $elt: ty, $bool: ty;)*) => ($(
        impl_simd_value!($t, $elt, $bool;);

        impl From<[$elt; <$t>::lanes()]> for Simd<$t> {
            #[inline(always)]
            fn from(vals: [$elt; <$t>::lanes()]) -> Self {
                Simd(<$t>::from(vals))
            }
        }

        impl SubsetOf<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn to_superset(&self) -> Self {
                *self
            }

            #[inline(always)]
            fn from_superset(element: &Self) -> Option<Self> {
                Some(*element)
            }

            #[inline(always)]
            fn from_superset_unchecked(element: &Self) -> Self {
                *element
            }

            #[inline(always)]
            fn is_in_subset(_: &Self) -> bool {
                true
            }
        }

        impl Num for Simd<$t> {
            type FromStrRadixErr = <$elt as Num>::FromStrRadixErr;

            #[inline(always)]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$elt>::from_str_radix(str, radix).map(Self::splat)
            }
        }

        impl FromPrimitive for Simd<$t> {
            #[inline(always)]
            fn from_i64(n: i64) -> Option<Self> {
                <$elt>::from_i64(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u64(n: u64) -> Option<Self> {
                <$elt>::from_u64(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_isize(n: isize) -> Option<Self>  {
                <$elt>::from_isize(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i8(n: i8) -> Option<Self>  {
                <$elt>::from_i8(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i16(n: i16) -> Option<Self>  {
                <$elt>::from_i16(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i32(n: i32) -> Option<Self>  {
                <$elt>::from_i32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_usize(n: usize) -> Option<Self>  {
                <$elt>::from_usize(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u8(n: u8) -> Option<Self>  {
                <$elt>::from_u8(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u16(n: u16) -> Option<Self>  {
                <$elt>::from_u16(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u32(n: u32) -> Option<Self>  {
                <$elt>::from_u32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_f32(n: f32) -> Option<Self>  {
                <$elt>::from_f32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_f64(n: f64) -> Option<Self>  {
                <$elt>::from_f64(n).map(Self::splat)
            }
        }


        impl Zero for Simd<$t> {
            #[inline(always)]
            fn zero() -> Self {
                Simd(<$t>::splat(<$elt>::zero()))
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl One for Simd<$t> {
            #[inline(always)]
            fn one() -> Self {
                Simd(<$t>::splat(<$elt>::one()))
            }
        }

        impl Add<Simd<$t>> for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub<Simd<$t>> for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl Mul<Simd<$t>> for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl Div<Simd<$t>> for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }

        impl Rem<Simd<$t>> for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn rem(self, rhs: Self) -> Self {
                Self(self.0 % rhs.0)
            }
        }

        impl AddAssign<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0
            }
        }

        impl SubAssign<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0
            }
        }

        impl DivAssign<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0
            }
        }

        impl MulAssign<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0
            }
        }

        impl RemAssign<Simd<$t>> for Simd<$t> {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: Self) {
                self.0 %= rhs.0
            }
        }

        impl SimdPartialOrd for Simd<$t> {
            #[inline(always)]
            fn simd_gt(self, other: Self) -> Self::SimdBool {
                self.0.gt(other.0)
            }

            #[inline(always)]
            fn simd_lt(self, other: Self) -> Self::SimdBool {
                self.0.lt(other.0)
            }

            #[inline(always)]
            fn simd_ge(self, other: Self) -> Self::SimdBool {
                self.0.ge(other.0)
            }

            #[inline(always)]
            fn simd_le(self, other: Self) -> Self::SimdBool {
                self.0.le(other.0)
            }

            #[inline(always)]
            fn simd_eq(self, other: Self) -> Self::SimdBool {
                self.0.eq(other.0)
            }

            #[inline(always)]
            fn simd_ne(self, other: Self) -> Self::SimdBool {
                self.0.ne(other.0)
            }

            #[inline(always)]
            fn simd_max(self, other: Self) -> Self {
                Simd(self.0.max(other.0))
            }
            #[inline(always)]
            fn simd_min(self, other: Self) -> Self {
                Simd(self.0.min(other.0))
            }

            #[inline(always)]
            fn simd_clamp(self, min: Self, max: Self) -> Self {
                self.simd_max(min).simd_min(max)
            }
        }

//        impl MeetSemilattice for Simd<$t> {
//            #[inline(always)]
//            fn meet(&self, other: &Self) -> Self {
//                Simd(self.0.min(other.0))
//            }
//        }
//
//        impl JoinSemilattice for Simd<$t> {
//            #[inline(always)]
//            fn join(&self, other: &Self) -> Self {
//                Simd(self.0.max(other.0))
//            }
//        }
    )*)
);

macro_rules! impl_int_simd(
    ($($t: ty, $elt: ty, $bool: ty;)*) => ($(
        impl_uint_simd!($t, $elt, $bool;);

        impl Neg for Simd<$t> {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
    )*)
);

macro_rules! impl_float_simd(
    ($($t: ty, $elt: ty, $bool: ty;)*) => ($(
        impl_int_simd!($t, $elt, $bool;);

        // FIXME: this should be part of impl_int_simd
        // but those methods do not seem to be implemented
        // by packed_simd for integers.
        impl SimdSigned for Simd<$t> {
            #[inline(always)]
            fn simd_abs(&self) -> Self {
                Simd(self.0.abs())
            }

            #[inline(always)]
            fn simd_abs_sub(&self, other: &Self) -> Self {
                Simd((self.0 - other.0).max(Self::zero().0))
            }

            #[inline(always)]
            fn simd_signum(&self) -> Self {
                // NOTE: is there a more efficient way of doing this?
                let zero = Self::zero().0;
                let one = Self::one().0;
                let gt = self.0.gt(zero);
                let lt = self.0.lt(zero);
                Simd(lt.select(-one, gt.select(one, zero)))
            }

            #[inline(always)]
            fn is_simd_positive(&self) -> Self::SimdBool {
                self.simd_gt(Self::zero())
            }

            #[inline(always)]
            fn is_simd_negative(&self) -> Self::SimdBool {
                self.simd_lt(Self::zero())
            }
        }

        impl Field for Simd<$t> {}

        impl SimdRealField for Simd<$t> {
            #[inline(always)]
            fn simd_atan2(self, other: Self) -> Self {
                self.zip_map_lanes(other, |a, b| a.atan2(b))
            }

            #[inline(always)]
            fn simd_default_epsilon() -> Self {
                Self::splat(<$elt>::default_epsilon())
            }

            #[inline(always)]
            fn simd_pi() -> Self {
                Simd(<$t>::PI)
            }

            #[inline(always)]
            fn simd_two_pi() -> Self {
                Simd(<$t>::PI + <$t>::PI)
            }

            #[inline(always)]
            fn simd_frac_pi_2() -> Self {
                Simd(<$t>::FRAC_PI_2)
            }

            #[inline(always)]
            fn simd_frac_pi_3() -> Self {
                Simd(<$t>::FRAC_PI_3)
            }

            #[inline(always)]
            fn simd_frac_pi_4() -> Self {
                Simd(<$t>::FRAC_PI_4)
            }

            #[inline(always)]
            fn simd_frac_pi_6() -> Self {
                Simd(<$t>::FRAC_PI_6)
            }

            #[inline(always)]
            fn simd_frac_pi_8() -> Self {
                Simd(<$t>::FRAC_PI_8)
            }

            #[inline(always)]
            fn simd_frac_1_pi() -> Self {
                Simd(<$t>::FRAC_1_PI)
            }

            #[inline(always)]
            fn simd_frac_2_pi() -> Self {
                Simd(<$t>::FRAC_2_PI)
            }

            #[inline(always)]
            fn simd_frac_2_sqrt_pi() -> Self {
                Simd(<$t>::FRAC_2_SQRT_PI)
            }


            #[inline(always)]
            fn simd_e() -> Self {
                Simd(<$t>::E)
            }

            #[inline(always)]
            fn simd_log2_e() -> Self {
                Simd(<$t>::LOG2_E)
            }

            #[inline(always)]
            fn simd_log10_e() -> Self {
                Simd(<$t>::LOG10_E)
            }

            #[inline(always)]
            fn simd_ln_2() -> Self {
                Simd(<$t>::LN_2)
            }

            #[inline(always)]
            fn simd_ln_10() -> Self {
                Simd(<$t>::LN_10)
            }
        }

        impl SimdComplexField for Simd<$t> {
            type SimdRealField = Self;

            #[inline(always)]
            fn from_simd_real(re: Self::SimdRealField) -> Self {
                re
            }

            #[inline(always)]
            fn simd_real(self) -> Self::SimdRealField {
                self
            }

            #[inline(always)]
            fn simd_imaginary(self) -> Self::SimdRealField {
                Self::zero()
            }

            #[inline(always)]
            fn simd_norm1(self) -> Self::SimdRealField {
                Simd(self.0.abs())
            }

            #[inline(always)]
            fn simd_modulus(self) -> Self::SimdRealField {
                Simd(self.0.abs())
            }

            #[inline(always)]
            fn simd_modulus_squared(self) -> Self::SimdRealField {
                self * self
            }

            #[inline(always)]
            fn simd_argument(self) -> Self::SimdRealField {
                self.map_lanes(|e| e.argument())
            }

            #[inline(always)]
            fn simd_to_exp(self) -> (Self::SimdRealField, Self) {
                let ge = self.0.ge(Self::one().0);
                let exp = ge.select(Self::one().0, -Self::one().0);
                (Simd(self.0 * exp), Simd(exp))
            }

            #[inline(always)]
            fn simd_recip(self) -> Self {
                Self::one() / self
            }

            #[inline(always)]
            fn simd_conjugate(self) -> Self {
                self
            }

            #[inline(always)]
            fn simd_scale(self, factor: Self::SimdRealField) -> Self {
                Simd(self.0 * factor.0)
            }

            #[inline(always)]
            fn simd_unscale(self, factor: Self::SimdRealField) -> Self {
                Simd(self.0 / factor.0)
            }

            #[inline(always)]
            fn simd_floor(self) -> Self {
                Simd(self.0.map_lanes(|e| e.floor()))
            }

            #[inline(always)]
            fn simd_ceil(self) -> Self {
                Simd(self.0.map_lanes(|e| e.ceil()))
            }

            #[inline(always)]
            fn simd_round(self) -> Self {
                Simd(self.0.map_lanes(|e| e.round()))
            }

            #[inline(always)]
            fn simd_trunc(self) -> Self {
                Simd(self.0.map_lanes(|e| e.trunc()))
            }

            #[inline(always)]
            fn simd_fract(self) -> Self {
                Simd(self.0.map_lanes(|e| e.fract()))
            }

            #[inline(always)]
            fn simd_abs(self) -> Self {
                Simd(self.0.abs())
            }

            #[inline(always)]
            fn simd_signum(self) -> Self {
                Simd(self.0.map_lanes(|e| e.signum()))
            }

            #[inline(always)]
            fn simd_mul_add(self, a: Self, b: Self) -> Self {
                Simd(self.0.mul_add(a.0, b.0))
            }

            #[inline(always)]
            fn simd_powi(self, n: i32) -> Self {
               Simd(self.0.powf(<$t>::splat(n as $elt)))
            }

            #[inline(always)]
            fn simd_powf(self, n: Self) -> Self {
                Simd(self.0.powf(n.0))
            }

            #[inline(always)]
            fn simd_powc(self, n: Self) -> Self {
               Simd(self.0.powf(n.0))
            }

            #[inline(always)]
            fn simd_sqrt(self) -> Self {
                Simd(self.0.sqrt())
            }

            #[inline(always)]
            fn simd_exp(self) -> Self {
                Simd(self.0.exp())
            }

            #[inline(always)]
            fn simd_exp2(self) -> Self {
                Simd(self.0.map_lanes(|e| e.exp2()))
            }


            #[inline(always)]
            fn simd_exp_m1(self) -> Self {
                Simd(self.0.map_lanes(|e| e.exp_m1()))
            }

            #[inline(always)]
            fn simd_ln_1p(self) -> Self {
                Simd(self.0.map_lanes(|e| e.ln_1p()))
            }

            #[inline(always)]
            fn simd_ln(self) -> Self {
                Simd(self.0.ln())
            }

            #[inline(always)]
            fn simd_log(self, base: Self) -> Self {
                Simd(self.0.zip_map_lanes(base.0, |e, b| e.log(b)))
            }

            #[inline(always)]
            fn simd_log2(self) -> Self {
                Simd(self.0.map_lanes(|e| e.log2()))
            }

            #[inline(always)]
            fn simd_log10(self) -> Self {
                Simd(self.0.map_lanes(|e| e.log10()))
            }

            #[inline(always)]
            fn simd_cbrt(self) -> Self {
                Simd(self.0.map_lanes(|e| e.cbrt()))
            }

            #[inline(always)]
            fn simd_hypot(self, other: Self) -> Self::SimdRealField {
                Simd(self.0.zip_map_lanes(other.0, |e, o| e.hypot(o)))
            }

            #[inline(always)]
            fn simd_sin(self) -> Self {
                Simd(self.0.sin())
            }

            #[inline(always)]
            fn simd_cos(self) -> Self {
                Simd(self.0.cos())
            }

            #[inline(always)]
            fn simd_tan(self) -> Self {
                Simd(self.0.map_lanes(|e| e.tan()))
            }

            #[inline(always)]
            fn simd_asin(self) -> Self {
                Simd(self.0.map_lanes(|e| e.asin()))
            }

            #[inline(always)]
            fn simd_acos(self) -> Self {
                Simd(self.0.map_lanes(|e| e.acos()))
            }

            #[inline(always)]
            fn simd_atan(self) -> Self {
                Simd(self.0.map_lanes(|e| e.atan()))
            }

            #[inline(always)]
            fn simd_sin_cos(self) -> (Self, Self) {
                (self.simd_sin(), self.simd_cos())
            }

//            #[inline(always]
//            fn simd_exp_m1(self) -> Self {
//                $libm::exp_m1(self)
//            }
//
//            #[inline(always]
//            fn simd_ln_1p(self) -> Self {
//                $libm::ln_1p(self)
//            }
//
            #[inline(always)]
            fn simd_sinh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.sinh()))
            }

            #[inline(always)]
            fn simd_cosh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.cosh()))
            }

            #[inline(always)]
            fn simd_tanh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.tanh()))
            }

            #[inline(always)]
            fn simd_asinh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.asinh()))
            }

            #[inline(always)]
            fn simd_acosh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.acosh()))
            }

            #[inline(always)]
            fn simd_atanh(self) -> Self {
                Simd(self.0.map_lanes(|e| e.atanh()))
            }
        }

        // NOTE: most of the impls in there are copy-paste from the implementation of
        // ComplexField for num_complex::Complex. Unfortunately, we can't reuse the implementations
        // so easily.
        impl SimdComplexField for num_complex::Complex<Simd<$t>> {
            type SimdRealField = Simd<$t>;

            #[inline]
            fn from_simd_real(re: Self::SimdRealField) -> Self {
                Self::new(re, Self::SimdRealField::zero())
            }

            #[inline]
            fn simd_real(self) -> Self::SimdRealField {
                self.re
            }

            #[inline]
            fn simd_imaginary(self) -> Self::SimdRealField {
                self.im
            }

            #[inline]
            fn simd_argument(self) -> Self::SimdRealField {
                self.im.simd_atan2(self.re)
            }

            #[inline]
            fn simd_modulus(self) -> Self::SimdRealField {
                self.re.simd_hypot(self.im)
            }

            #[inline]
            fn simd_modulus_squared(self) -> Self::SimdRealField {
                self.re * self.re + self.im * self.im
            }

            #[inline]
            fn simd_norm1(self) -> Self::SimdRealField {
                self.re.simd_abs() + self.im.simd_abs()
            }

            #[inline]
            fn simd_recip(self) -> Self {
                Self::one() / self
            }

            #[inline]
            fn simd_conjugate(self) -> Self {
                self.conj()
            }

            #[inline]
            fn simd_scale(self, factor: Self::SimdRealField) -> Self {
                self * factor
            }

            #[inline]
            fn simd_unscale(self, factor: Self::SimdRealField) -> Self {
                self / factor
            }

            #[inline]
            fn simd_floor(self) -> Self {
                Self::new(self.re.simd_floor(), self.im.simd_floor())
            }

            #[inline]
            fn simd_ceil(self) -> Self {
                Self::new(self.re.simd_ceil(), self.im.simd_ceil())
            }

            #[inline]
            fn simd_round(self) -> Self {
                Self::new(self.re.simd_round(), self.im.simd_round())
            }

            #[inline]
            fn simd_trunc(self) -> Self {
                Self::new(self.re.simd_trunc(), self.im.simd_trunc())
            }

            #[inline]
            fn simd_fract(self) -> Self {
                Self::new(self.re.simd_fract(), self.im.simd_fract())
            }

            #[inline]
            fn simd_mul_add(self, a: Self, b: Self) -> Self {
                self * a + b
            }

            #[inline]
            fn simd_abs(self) -> Self::SimdRealField {
                self.simd_modulus()
            }

            #[inline]
            fn simd_exp2(self) -> Self {
                let _2 = Simd::<$t>::one() + Simd::<$t>::one();
                num_complex::Complex::new(_2, Simd::<$t>::zero()).simd_powc(self)
            }

            #[inline]
            fn simd_exp_m1(self) -> Self {
                self.simd_exp() - Self::one()
            }

            #[inline]
            fn simd_ln_1p(self) -> Self {
                (Self::one() + self).simd_ln()
            }

            #[inline]
            fn simd_log2(self) -> Self {
                let _2 = Simd::<$t>::one() + Simd::<$t>::one();
                self.simd_log(_2)
            }

            #[inline]
            fn simd_log10(self) -> Self {
                let _10 = Simd::<$t>::from_subset(&10.0f64);
                self.simd_log(_10)
            }

            #[inline]
            fn simd_cbrt(self) -> Self {
                let one_third = Simd::<$t>::from_subset(&(1.0 / 3.0));
                self.simd_powf(one_third)
            }

            #[inline]
            fn simd_powi(self, n: i32) -> Self {
                // FIXME: is there a more accurate solution?
                let n = Simd::<$t>::from_subset(&(n as f64));
                self.simd_powf(n)
            }

            /*
             *
             *
             * Unfortunately we are forced to copy-paste all
             * those impls from https://github.com/rust-num/num-complex/blob/master/src/lib.rs
             * to avoid requiring `std`.
             *
             *
             */
            /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
            #[inline]
            fn simd_exp(self) -> Self {
                // formula: e^(a + bi) = e^a (cos(b) + i*sin(b))
                // = from_polar(e^a, b)
                simd_complex_from_polar(self.re.simd_exp(), self.im)
            }

            /// Computes the principal value of natural logarithm of `self`.
            ///
            /// This function has one branch cut:
            ///
            /// * `(-∞, 0]`, continuous from above.
            ///
            /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
            #[inline]
            fn simd_ln(self) -> Self {
                // formula: ln(z) = ln|z| + i*arg(z)
                let (r, theta) = self.simd_to_polar();
                Self::new(r.simd_ln(), theta)
            }

            /// Computes the principal value of the square root of `self`.
            ///
            /// This function has one branch cut:
            ///
            /// * `(-∞, 0)`, continuous from above.
            ///
            /// The branch satisfies `-π/2 ≤ arg(sqrt(z)) ≤ π/2`.
            #[inline]
            fn simd_sqrt(self) -> Self {
                // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
                let two = Simd::<$t>::one() + Simd::<$t>::one();
                let (r, theta) = self.simd_to_polar();
                simd_complex_from_polar(r.simd_sqrt(), theta / two)
            }

            #[inline]
            fn simd_hypot(self, b: Self) -> Self::SimdRealField {
                (self.simd_modulus_squared() + b.simd_modulus_squared()).simd_sqrt()
            }

            /// Raises `self` to a floating point power.
            #[inline]
            fn simd_powf(self, exp: Self::SimdRealField) -> Self {
                // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
                // = from_polar(ρ^y, θ y)
                let (r, theta) = self.simd_to_polar();
                simd_complex_from_polar(r.simd_powf(exp), theta * exp)
            }

            /// Returns the logarithm of `self` with respect to an arbitrary base.
            #[inline]
            fn simd_log(self, base: Simd<$t>) -> Self {
                // formula: log_y(x) = log_y(ρ e^(i θ))
                // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
                // = log_y(ρ) + i θ / ln(y)
                let (r, theta) = self.simd_to_polar();
                Self::new(r.simd_log(base), theta / base.simd_ln())
            }

            /// Raises `self` to a complex power.
            #[inline]
            fn simd_powc(self, exp: Self) -> Self {
                // formula: x^y = (a + i b)^(c + i d)
                // = (ρ e^(i θ))^c (ρ e^(i θ))^(i d)
                //    where ρ=|x| and θ=arg(x)
                // = ρ^c e^(−d θ) e^(i c θ) ρ^(i d)
                // = p^c e^(−d θ) (cos(c θ)
                //   + i sin(c θ)) (cos(d ln(ρ)) + i sin(d ln(ρ)))
                // = p^c e^(−d θ) (
                //   cos(c θ) cos(d ln(ρ)) − sin(c θ) sin(d ln(ρ))
                //   + i(cos(c θ) sin(d ln(ρ)) + sin(c θ) cos(d ln(ρ))))
                // = p^c e^(−d θ) (cos(c θ + d ln(ρ)) + i sin(c θ + d ln(ρ)))
                // = from_polar(p^c e^(−d θ), c θ + d ln(ρ))
                let (r, theta) = self.simd_to_polar();
                simd_complex_from_polar(
                    r.simd_powf(exp.re) * (-exp.im * theta).simd_exp(),
                    exp.re * theta + exp.im * r.simd_ln(),
                )
            }

            /*
            /// Raises a floating point number to the complex power `self`.
            #[inline]
            fn simd_expf(&self, base: T) -> Self {
                // formula: x^(a+bi) = x^a x^bi = x^a e^(b ln(x) i)
                // = from_polar(x^a, b ln(x))
                Self::from_polar(&base.powf(self.re), &(self.im * base.ln()))
            }
            */

            /// Computes the sine of `self`.
            #[inline]
            fn simd_sin(self) -> Self {
                // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
                Self::new(
                    self.re.simd_sin() * self.im.simd_cosh(),
                    self.re.simd_cos() * self.im.simd_sinh(),
                )
            }

            /// Computes the cosine of `self`.
            #[inline]
            fn simd_cos(self) -> Self {
                // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
                Self::new(
                    self.re.simd_cos() * self.im.simd_cosh(),
                    -self.re.simd_sin() * self.im.simd_sinh(),
                )
            }

            #[inline]
            fn simd_sin_cos(self) -> (Self, Self) {
                let (rsin, rcos) = self.re.simd_sin_cos();
                let (isinh, icosh) = self.im.simd_sinh_cosh();
                let sin = Self::new(rsin * icosh, rcos * isinh);
                let cos = Self::new(rcos * icosh, -rsin * isinh);

                (sin, cos)
            }

            /// Computes the tangent of `self`.
            #[inline]
            fn simd_tan(self) -> Self {
                // formula: tan(a + bi) = (sin(2a) + i*sinh(2b))/(cos(2a) + cosh(2b))
                let (two_re, two_im) = (self.re + self.re, self.im + self.im);
                Self::new(two_re.simd_sin(), two_im.simd_sinh()).unscale(two_re.simd_cos() + two_im.simd_cosh())
            }

            /// Computes the principal value of the inverse sine of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞, -1)`, continuous from above.
            /// * `(1, ∞)`, continuous from below.
            ///
            /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
            #[inline]
            fn simd_asin(self) -> Self {
                // formula: arcsin(z) = -i ln(sqrt(1-z^2) + iz)
                let i = Self::i();
                -i * ((Self::one() - self * self).simd_sqrt() + i * self).simd_ln()
            }

            /// Computes the principal value of the inverse cosine of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞, -1)`, continuous from above.
            /// * `(1, ∞)`, continuous from below.
            ///
            /// The branch satisfies `0 ≤ Re(acos(z)) ≤ π`.
            #[inline]
            fn simd_acos(self) -> Self {
                // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
                let i = Self::i();
                -i * (i * (Self::one() - self * self).simd_sqrt() + self).simd_ln()
            }

            /// Computes the principal value of the inverse tangent of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞i, -i]`, continuous from the left.
            /// * `[i, ∞i)`, continuous from the right.
            ///
            /// The branch satisfies `-π/2 ≤ Re(atan(z)) ≤ π/2`.
            #[inline]
            fn simd_atan(self) -> Self {
                // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
                let i = Self::i();
                let one = Self::one();
                let two = one + one;

                if self == i {
                    return Self::new(Simd::<$t>::zero(), Simd::<$t>::one() / Simd::<$t>::zero());
                } else if self == -i {
                    return Self::new(Simd::<$t>::zero(), -Simd::<$t>::one() / Simd::<$t>::zero());
                }

                ((one + i * self).simd_ln() - (one - i * self).simd_ln()) / (two * i)
            }

            /// Computes the hyperbolic sine of `self`.
            #[inline]
            fn simd_sinh(self) -> Self {
                // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
                Self::new(
                    self.re.simd_sinh() * self.im.simd_cos(),
                    self.re.simd_cosh() * self.im.simd_sin(),
                )
            }

            /// Computes the hyperbolic cosine of `self`.
            #[inline]
            fn simd_cosh(self) -> Self {
                // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
                Self::new(
                    self.re.simd_cosh() * self.im.simd_cos(),
                    self.re.simd_sinh() * self.im.simd_sin(),
                )
            }

            #[inline]
            fn simd_sinh_cosh(self) -> (Self, Self) {
                let (rsinh, rcosh) = self.re.simd_sinh_cosh();
                let (isin, icos) = self.im.simd_sin_cos();
                let sin = Self::new(rsinh * icos, rcosh * isin);
                let cos = Self::new(rcosh * icos, rsinh * isin);

                (sin, cos)
            }

            /// Computes the hyperbolic tangent of `self`.
            #[inline]
            fn simd_tanh(self) -> Self {
                // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
                let (two_re, two_im) = (self.re + self.re, self.im + self.im);
                Self::new(two_re.simd_sinh(), two_im.simd_sin()).unscale(two_re.simd_cosh() + two_im.simd_cos())
            }

            /// Computes the principal value of inverse hyperbolic sine of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞i, -i)`, continuous from the left.
            /// * `(i, ∞i)`, continuous from the right.
            ///
            /// The branch satisfies `-π/2 ≤ Im(asinh(z)) ≤ π/2`.
            #[inline]
            fn simd_asinh(self) -> Self {
                // formula: arcsinh(z) = ln(z + sqrt(1+z^2))
                let one = Self::one();
                (self + (one + self * self).simd_sqrt()).simd_ln()
            }

            /// Computes the principal value of inverse hyperbolic cosine of `self`.
            ///
            /// This function has one branch cut:
            ///
            /// * `(-∞, 1)`, continuous from above.
            ///
            /// The branch satisfies `-π ≤ Im(acosh(z)) ≤ π` and `0 ≤ Re(acosh(z)) < ∞`.
            #[inline]
            fn simd_acosh(self) -> Self {
                // formula: arccosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))
                let one = Self::one();
                let two = one + one;
                two * (((self + one) / two).simd_sqrt() + ((self - one) / two).simd_sqrt()).simd_ln()
            }

            /// Computes the principal value of inverse hyperbolic tangent of `self`.
            ///
            /// This function has two branch cuts:
            ///
            /// * `(-∞, -1]`, continuous from above.
            /// * `[1, ∞)`, continuous from below.
            ///
            /// The branch satisfies `-π/2 ≤ Im(atanh(z)) ≤ π/2`.
            #[inline]
            fn simd_atanh(self) -> Self {
                // formula: arctanh(z) = (ln(1+z) - ln(1-z))/2
                let one = Self::one();
                let two = one + one;
                if self == one {
                    return Self::new(Simd::<$t>::one() / Simd::<$t>::zero(), Simd::<$t>::zero());
                } else if self == -one {
                    return Self::new(-Simd::<$t>::one() / Simd::<$t>::zero(), Simd::<$t>::zero());
                }
                ((one + self).simd_ln() - (one - self).simd_ln()) / two
            }
        }
    )*)
);

#[inline]
fn simd_complex_from_polar<N: SimdRealField>(r: N, theta: N) -> num_complex::Complex<N> {
    num_complex::Complex::new(r * theta.simd_cos(), r * theta.simd_sin())
}

impl_float_simd!(
    packed_simd::f32x2, f32, m32x2;
    packed_simd::f32x4, f32, m32x4;
    packed_simd::f32x8, f32, m32x8;
    packed_simd::f32x16, f32, m32x16;
    packed_simd::f64x2, f64, m64x2;
    packed_simd::f64x4, f64, m64x4;
    packed_simd::f64x8, f64, m64x8;
);

impl_int_simd!(
    packed_simd::i128x1, i128, m128x1;
    packed_simd::i128x2, i128, m128x2;
    packed_simd::i128x4, i128, m128x4;
    packed_simd::i16x2, i16, m16x2;
    packed_simd::i16x4, i16, m16x4;
    packed_simd::i16x8, i16, m16x8;
    packed_simd::i16x16, i16, m16x16;
    packed_simd::i16x32, i16, m16x32;
    packed_simd::i32x2, i32, m32x2;
    packed_simd::i32x4, i32, m32x4;
    packed_simd::i32x8, i32, m32x8;
    packed_simd::i32x16, i32, m32x16;
    packed_simd::i64x2, i64, m64x2;
    packed_simd::i64x4, i64, m64x4;
    packed_simd::i64x8, i64, m64x8;
    packed_simd::i8x2, i8, m8x2;
    packed_simd::i8x4, i8, m8x4;
    packed_simd::i8x8, i8, m8x8;
    packed_simd::i8x16, i8, m8x16;
    packed_simd::i8x32, i8, m8x32;
    packed_simd::i8x64, i8, m8x64;
    packed_simd::isizex2, isize, msizex2;
    packed_simd::isizex4, isize, msizex4;
    packed_simd::isizex8, isize, msizex8;
);

impl_uint_simd!(
    packed_simd::u128x1, u128, m128x1;
    packed_simd::u128x2, u128, m128x2;
    packed_simd::u128x4, u128, m128x4;
    packed_simd::u16x2, u16, m16x2;
    packed_simd::u16x4, u16, m16x4;
    packed_simd::u16x8, u16, m16x8;
    packed_simd::u16x16, u16, m16x16;
    packed_simd::u16x32, u16, m16x32;
    packed_simd::u32x2, u32, m32x2;
    packed_simd::u32x4, u32, m32x4;
    packed_simd::u32x8, u32, m32x8;
    packed_simd::u32x16, u32, m32x16;
    packed_simd::u64x2, u64, m64x2;
    packed_simd::u64x4, u64, m64x4;
    packed_simd::u64x8, u64, m64x8;
    packed_simd::u8x2, u8, m8x2;
    packed_simd::u8x4, u8, m8x4;
    packed_simd::u8x8, u8, m8x8;
    packed_simd::u8x16, u8, m8x16;
    packed_simd::u8x32, u8, m8x32;
    packed_simd::u8x64, u8, m8x64;
    packed_simd::usizex2, usize, msizex2;
    packed_simd::usizex4, usize, msizex4;
    packed_simd::usizex8, usize, msizex8;
);

impl_simd_value!(
    m128x1, bool, m128x1;
    m128x2, bool, m128x2;
    m128x4, bool, m128x4;
    m16x2, bool, m16x2;
    m16x4, bool, m16x4;
    m16x8, bool, m16x8;
    m16x16, bool, m16x16;
    m16x32, bool, m16x32;
    m32x2, bool, m32x2;
    m32x4, bool, m32x4;
    m32x8, bool, m32x8;
    m32x16, bool, m32x16;
    m64x2, bool, m64x2;
    m64x4, bool, m64x4;
    m64x8, bool, m64x8;
    m8x2, bool, m8x2;
    m8x4, bool, m8x4;
    m8x8, bool, m8x8;
    m8x16, bool, m8x16;
    m8x32, bool, m8x32;
    m8x64, bool, m8x64;
    msizex2, bool, msizex2;
    msizex4, bool, msizex4;
    msizex8, bool, msizex8;
);

impl_simd_bool!(
    packed_simd::m128x1;
    packed_simd::m128x2;
    packed_simd::m128x4;
    packed_simd::m16x2;
    packed_simd::m16x4;
    packed_simd::m16x8;
    packed_simd::m16x16;
    packed_simd::m16x32;
    packed_simd::m32x2;
    packed_simd::m32x4;
    packed_simd::m32x8;
    packed_simd::m32x16;
    packed_simd::m64x2;
    packed_simd::m64x4;
    packed_simd::m64x8;
    packed_simd::m8x2;
    packed_simd::m8x4;
    packed_simd::m8x8;
    packed_simd::m8x16;
    packed_simd::m8x32;
    packed_simd::m8x64;
    packed_simd::msizex2;
    packed_simd::msizex4;
    packed_simd::msizex8;
);

//
// NOTE: the following does not work because of the orphan rules.
//
//macro_rules! impl_simd_complex_from(
//    ($($t: ty, $elt: ty $(, $i: expr)*;)*) => ($(
//        impl From<[num_complex::Complex<$elt>; <$t>::lanes()]> for num_complex::Complex<Simd<$t>> {
//            #[inline(always)]
//            fn from(vals: [num_complex::Complex<$elt>; <$t>::lanes()]) -> Self {
//                num_complex::Complex {
//                    re: <$t>::from([$(vals[$i].re),*]),
//                    im: <$t>::from([$(vals[$i].im),*]),
//                }
//            }
//        }
//    )*)
//);
//
//impl_simd_complex_from!(
//    packed_simd::f32x2, f32, 0, 1;
//    packed_simd::f32x4, f32, 0, 1, 2, 3;
//    packed_simd::f32x8, f32, 0, 1, 2, 3, 4, 5, 6, 7;
//    packed_simd::f32x16, f32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
//);

//////////////////////////////////////////
//               Aliases                //
//////////////////////////////////////////

pub type f32x2 = Simd<packed_simd::f32x2>;
pub type f32x4 = Simd<packed_simd::f32x4>;
pub type f32x8 = Simd<packed_simd::f32x8>;
pub type f32x16 = Simd<packed_simd::f32x16>;
pub type f64x2 = Simd<packed_simd::f64x2>;
pub type f64x4 = Simd<packed_simd::f64x4>;
pub type f64x8 = Simd<packed_simd::f64x8>;
pub type i128x1 = Simd<packed_simd::i128x1>;
pub type i128x2 = Simd<packed_simd::i128x2>;
pub type i128x4 = Simd<packed_simd::i128x4>;
pub type i16x2 = Simd<packed_simd::i16x2>;
pub type i16x4 = Simd<packed_simd::i16x4>;
pub type i16x8 = Simd<packed_simd::i16x8>;
pub type i16x16 = Simd<packed_simd::i16x16>;
pub type i16x32 = Simd<packed_simd::i16x32>;
pub type i32x2 = Simd<packed_simd::i32x2>;
pub type i32x4 = Simd<packed_simd::i32x4>;
pub type i32x8 = Simd<packed_simd::i32x8>;
pub type i32x16 = Simd<packed_simd::i32x16>;
pub type i64x2 = Simd<packed_simd::i64x2>;
pub type i64x4 = Simd<packed_simd::i64x4>;
pub type i64x8 = Simd<packed_simd::i64x8>;
pub type i8x2 = Simd<packed_simd::i8x2>;
pub type i8x4 = Simd<packed_simd::i8x4>;
pub type i8x8 = Simd<packed_simd::i8x8>;
pub type i8x16 = Simd<packed_simd::i8x16>;
pub type i8x32 = Simd<packed_simd::i8x32>;
pub type i8x64 = Simd<packed_simd::i8x64>;
pub type isizex2 = Simd<packed_simd::isizex2>;
pub type isizex4 = Simd<packed_simd::isizex4>;
pub type isizex8 = Simd<packed_simd::isizex8>;
pub type u128x1 = Simd<packed_simd::u128x1>;
pub type u128x2 = Simd<packed_simd::u128x2>;
pub type u128x4 = Simd<packed_simd::u128x4>;
pub type u16x2 = Simd<packed_simd::u16x2>;
pub type u16x4 = Simd<packed_simd::u16x4>;
pub type u16x8 = Simd<packed_simd::u16x8>;
pub type u16x16 = Simd<packed_simd::u16x16>;
pub type u16x32 = Simd<packed_simd::u16x32>;
pub type u32x2 = Simd<packed_simd::u32x2>;
pub type u32x4 = Simd<packed_simd::u32x4>;
pub type u32x8 = Simd<packed_simd::u32x8>;
pub type u32x16 = Simd<packed_simd::u32x16>;
pub type u64x2 = Simd<packed_simd::u64x2>;
pub type u64x4 = Simd<packed_simd::u64x4>;
pub type u64x8 = Simd<packed_simd::u64x8>;
pub type u8x2 = Simd<packed_simd::u8x2>;
pub type u8x4 = Simd<packed_simd::u8x4>;
pub type u8x8 = Simd<packed_simd::u8x8>;
pub type u8x16 = Simd<packed_simd::u8x16>;
pub type u8x32 = Simd<packed_simd::u8x32>;
pub type u8x64 = Simd<packed_simd::u8x64>;
pub type usizex2 = Simd<packed_simd::usizex2>;
pub type usizex4 = Simd<packed_simd::usizex4>;
pub type usizex8 = Simd<packed_simd::usizex8>;

pub use packed_simd::m128x1;
pub use packed_simd::m128x2;
pub use packed_simd::m128x4;
pub use packed_simd::m16x16;
pub use packed_simd::m16x2;
pub use packed_simd::m16x32;
pub use packed_simd::m16x4;
pub use packed_simd::m16x8;
pub use packed_simd::m32x16;
pub use packed_simd::m32x2;
pub use packed_simd::m32x4;
pub use packed_simd::m32x8;
pub use packed_simd::m64x2;
pub use packed_simd::m64x4;
pub use packed_simd::m64x8;
pub use packed_simd::m8x16;
pub use packed_simd::m8x2;
pub use packed_simd::m8x32;
pub use packed_simd::m8x4;
pub use packed_simd::m8x64;
pub use packed_simd::m8x8;
pub use packed_simd::msizex2;
pub use packed_simd::msizex4;
pub use packed_simd::msizex8;
