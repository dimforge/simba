use crate::simd::SimdBool;

/// Trait implemented by Simd types as well as scalar types (f32, u32, etc.).
pub trait SimdValue: Sized {
    /// The type of the elements of each lane of this SIMD value.
    type Element: SimdValue<Element = Self::Element, SimdBool = bool>;
    type SimdBool: SimdBool;

    fn lanes() -> usize;
    fn splat(val: Self::Element) -> Self;
    fn extract(&self, i: usize) -> Self::Element;
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element;
    fn replace(&mut self, i: usize, val: Self::Element);
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element);

    fn select(self, cond: Self::SimdBool, other: Self) -> Self;

    #[inline(always)]
    fn map_lanes(self, f: impl Fn(Self::Element) -> Self::Element) -> Self
    where
        Self: Clone,
    {
        let mut result = self.clone();

        for i in 0..Self::lanes() {
            unsafe { result.replace_unchecked(i, f(self.extract_unchecked(i))) }
        }

        result
    }

    #[inline(always)]
    fn zip_map_lanes(
        self,
        b: Self,
        f: impl Fn(Self::Element, Self::Element) -> Self::Element,
    ) -> Self
    where
        Self: Clone,
    {
        let mut result = self.clone();

        for i in 0..Self::lanes() {
            unsafe {
                let a = self.extract_unchecked(i);
                let b = b.extract_unchecked(i);
                result.replace_unchecked(i, f(a, b))
            }
        }

        result
    }
}

impl<N: SimdValue> SimdValue for num_complex::Complex<N> {
    type Element = num_complex::Complex<N::Element>;
    type SimdBool = N::SimdBool;

    #[inline(always)]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        num_complex::Complex {
            re: N::splat(val.re),
            im: N::splat(val.im),
        }
    }

    #[inline(always)]
    fn extract(&self, i: usize) -> Self::Element {
        num_complex::Complex {
            re: self.re.extract(i),
            im: self.im.extract(i),
        }
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        num_complex::Complex {
            re: self.re.extract_unchecked(i),
            im: self.im.extract_unchecked(i),
        }
    }

    #[inline(always)]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.re.replace(i, val.re);
        self.im.replace(i, val.im);
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.re.replace_unchecked(i, val.re);
        self.im.replace_unchecked(i, val.im);
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        num_complex::Complex {
            re: self.re.select(cond, other.re),
            im: self.im.select(cond, other.im),
        }
    }
}

macro_rules! impl_simd_value_for_scalar(
    ($($t: ty),*) => {$(
        impl SimdValue for $t {
            type Element = $t;
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
    )*}
);

impl_simd_value_for_scalar!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);
#[cfg(feature = "decimal")]
impl_simd_value_for_scalar!(decimal::d128);
