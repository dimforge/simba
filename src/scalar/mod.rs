//! Traits implemented by scalar, non-SIMD, types.

pub use self::complex::ComplexField;
pub use self::field::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, Field};
#[cfg(feature = "partial_fixed_point_support")]
pub use self::fixed::{FixedI16, FixedI32, FixedI64, FixedI8};
pub use self::real::RealField;
pub use self::subset::{SubsetOf, SupersetOf};

mod real;
#[macro_use]
mod complex;
mod field;
#[cfg(feature = "partial_fixed_point_support")]
mod fixed;
mod subset;
