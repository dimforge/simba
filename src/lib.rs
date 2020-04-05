/*!
__Simba__ is a crate defining a set of trait for writing code that can be generic with regard to the
number of lanes of the numeric input value. Those traits are implemented by `f32`, `u32`, `i16`,
`bool` as well as SIMD types like `f32x4, u32x8, i16x2`, etc.

One example of use-case applied by the [nalgebra crate](https://nalgebra.org) is to define generic methods
like vector normalization that will work for `Vector3<f32>` as well as `Vector3<f32x4>`.

This makes it easier leverage the power of [SIMD Array-of-Struct-of-Array (AoSoA)](https://www.rustsim.org/blog/2020/03/23/simd-aosoa-in-nalgebra/)
with less code duplication.
*/

extern crate num_traits as num;

#[macro_use]
pub mod scalar;
pub mod simd;
