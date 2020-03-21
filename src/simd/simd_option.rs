use crate::simd::{SimdBool, SimdValue};

//pub trait SimdOption {
//    type SimdValue: SimdValue;
//
//    fn simd_unwrap(self) -> Self::SimdValue;
//    fn simd_unwrap_or(self, other: impl FnOnce() -> Self::SimdValue) -> Self::SimdValue;
//}

//impl<T: SimdValue> SimdOption for Option<T> {
//    type SimdValue = T;
//
//    #[inline(always)]
//    fn simd_unwrap(self) -> T {
//        self.unwrap()
//    }
//
//    #[inline(always)]
//    fn simd_unwrap_or(self, other: impl FnOnce() -> Self::SimdValue) -> Self::SimdValue {
//        self.unwrap_or_else(other)
//    }
//}

pub struct SimdOption<V: SimdValue> {
    val: V,
    mask: V::SimdBool,
}

impl<V: SimdValue> SimdOption<V> {
    pub fn new(val: V, mask: V::SimdBool) -> Self {
        SimdOption { val, mask }
    }

    pub fn option(self) -> Option<V> {
        if self.mask.all() {
            Some(self.val)
        } else {
            None
        }
    }

    #[inline]
    pub fn simd_unwrap(self) -> V {
        assert!(
            self.mask.all(),
            "Attempt to unwrap an SIMD value with at least one false lane."
        );
        self.val
    }

    #[inline(always)]
    pub fn simd_unwrap_or(self, other: impl FnOnce() -> V) -> V {
        self.val.select(self.mask, other())
    }
}
