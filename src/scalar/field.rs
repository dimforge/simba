use crate::simd::SimdValue;
use num::NumAssign;
pub use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

/// Trait __alias__ for `Add` and `AddAssign` with result of type `Self`.
pub trait ClosedAdd<Right = Self>: Sized + Add<Right, Output = Self> + AddAssign<Right> {}

/// Trait __alias__ for `Sub` and `SubAssign` with result of type `Self`.
pub trait ClosedSub<Right = Self>: Sized + Sub<Right, Output = Self> + SubAssign<Right> {}

/// Trait __alias__ for `Mul` and `MulAssign` with result of type `Self`.
pub trait ClosedMul<Right = Self>: Sized + Mul<Right, Output = Self> + MulAssign<Right> {}

/// Trait __alias__ for `Div` and `DivAssign` with result of type `Self`.
pub trait ClosedDiv<Right = Self>: Sized + Div<Right, Output = Self> + DivAssign<Right> {}

/// Trait __alias__ for `Neg` with result of type `Self`.
pub trait ClosedNeg: Sized + Neg<Output = Self> {}

impl<T, Right> ClosedAdd<Right> for T where T: Add<Right, Output = T> + AddAssign<Right> {}
impl<T, Right> ClosedSub<Right> for T where T: Sub<Right, Output = T> + SubAssign<Right> {}
impl<T, Right> ClosedMul<Right> for T where T: Mul<Right, Output = T> + MulAssign<Right> {}
impl<T, Right> ClosedDiv<Right> for T where T: Div<Right, Output = T> + DivAssign<Right> {}
impl<T> ClosedNeg for T where T: Neg<Output = T> {}

/// Trait implemented by fields, i.e., complex numbers and floats.
pub trait Field: SimdValue + NumAssign + ClosedNeg {}

impl<N: SimdValue + Clone + NumAssign + ClosedNeg> Field for num_complex::Complex<N> {}

macro_rules! impl_field(
    ($($t: ty),*) => {$(
        impl Field for $t {}
    )*}
);

impl_field!(f32, f64);
//#[cfg(feature = "decimal")]
//impl_field!(decimal::d128);
