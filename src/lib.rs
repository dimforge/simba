/*!
__Simba__ is a crate defining a set of trait for writing code that can be generic with regard to the
number of lanes of the numeric input value. Those traits are implemented by `f32`, `u32`, `i16`,
`bool` as well as SIMD types like `f32x4, u32x8, i16x2`, etc.

One example of use-case applied by the [nalgebra crate](https://nalgebra.org) is to define generic methods
like vector normalization that will work for `Vector3<f32>` as well as `Vector3<f32x4>`.

This makes it easier leverage the power of [SIMD Array-of-Struct-of-Array (AoSoA)](https://www.rustsim.org/blog/2020/03/23/simd-aosoa-in-nalgebra/)
with less code duplication.


## Cargo features

Two cargo features can be optionally enabled:
- With the __`packed_simd`__ feature enabled, the `simba::simd` module will export several SIMD types like `f32x2`,
 `f64x4`, `i32i8`, `u16i16`, etc. There types are wrappers around the SIMD types from the [__packed_simd__
 crate](https://docs.rs/packed_simd). This requires a nightly compiler.
- With the __`wide`__ feature enabled, the `simba::simd` module will export the `WideF32x4` and `WideBoolF32x4`
  types. They types ure wrapper around the `wide::f32x4` type from the [__wide__ crate](https://docs.rs/wide).
  This will work with both a stable or nightly compiler.

If none of those features are enabled, __simba__ will still define all the scalar and SIMD traits.
However, the SIMD traits won't be implemented for any SIMD types. Therefore it is recommended to:
- Use the `packed_simd` feature if you want more features, and can afford to use a nightly compiler.
- Use the `wide` feature if you only need 4-lanes 32-bits floats, and can't afford to use a nightly compiler.
*/

#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style,
    unused_parens,
    unused_results,
    missing_docs,
    clippy::all
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;
extern crate num_traits as num;

#[macro_use]
pub mod scalar;
pub mod simd;
