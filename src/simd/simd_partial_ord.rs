use crate::simd::SimdValue;

pub trait SimdPartialOrd: SimdValue {
    fn simd_gt(self, other: Self) -> Self::SimdBool;
    fn simd_lt(self, other: Self) -> Self::SimdBool;
    fn simd_ge(self, other: Self) -> Self::SimdBool;
    fn simd_le(self, other: Self) -> Self::SimdBool;
    fn simd_eq(self, other: Self) -> Self::SimdBool;
    fn simd_ne(self, other: Self) -> Self::SimdBool;

    fn simd_max(self, other: Self) -> Self;
    fn simd_min(self, other: Self) -> Self;
    fn simd_clamp(self, min: Self, max: Self) -> Self;
}

impl<T: PartialOrd + SimdValue<SimdBool = bool>> SimdPartialOrd for T {
    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdBool {
        self > other
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdBool {
        self < other
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdBool {
        self >= other
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdBool {
        self <= other
    }

    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdBool {
        self == other
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdBool {
        self != other
    }

    #[inline(always)]
    fn simd_max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    fn simd_min(self, other: Self) -> Self {
        if self <= other {
            self
        } else {
            other
        }
    }

    #[inline(always)]
    fn simd_clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
}
