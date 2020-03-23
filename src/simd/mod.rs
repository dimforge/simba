//! Traits for SIMD types (also implemented by scalar types).

pub use self::simd_bool::SimdBool;
pub use self::simd_complex::SimdComplexField;
#[cfg(feature = "simd")]
pub use self::simd_impl::*;
pub use self::simd_option::SimdOption;
pub use self::simd_partial_ord::SimdPartialOrd;
pub use self::simd_real::SimdRealField;
pub use self::simd_signed::SimdSigned;
pub use self::simd_value::{PrimitiveSimdValue, SimdValue};

mod simd_bool;
mod simd_complex;
#[cfg(feature = "simd")]
mod simd_impl;
mod simd_option;
mod simd_partial_ord;
mod simd_real;
mod simd_signed;
mod simd_value;
