use crate::simd::SimdValue;
use std::ops::{BitAnd, BitOr, BitXor};

pub trait SimdBool:
    Copy + BitAnd<Self, Output = Self> + BitOr<Self, Output = Self> + BitXor<Self, Output = Self>
{
    fn and(self) -> bool;
    fn or(self) -> bool;
    fn xor(self) -> bool;
    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;
    fn if_else<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_value: impl FnOnce() -> Res,
    ) -> Res;
    fn if_else2<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_value: impl FnOnce() -> Res,
    ) -> Res;
    fn if_else3<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_value: impl FnOnce() -> Res,
    ) -> Res;
}

impl SimdBool for bool {
    #[inline(always)]
    fn and(self) -> bool {
        self
    }

    #[inline(always)]
    fn or(self) -> bool {
        self
    }

    #[inline(always)]
    fn xor(self) -> bool {
        self
    }

    #[inline(always)]
    fn all(self) -> bool {
        self
    }

    #[inline(always)]
    fn any(self) -> bool {
        self
    }

    #[inline(always)]
    fn none(self) -> bool {
        !self
    }

    #[inline(always)]
    fn if_else<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_value: impl FnOnce() -> Res,
    ) -> Res {
        if self {
            if_value()
        } else {
            else_value()
        }
    }

    #[inline(always)]
    fn if_else2<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_value: impl FnOnce() -> Res,
    ) -> Res {
        if self {
            if_value()
        } else if else_if.0() {
            else_if.1()
        } else {
            else_value()
        }
    }

    #[inline(always)]
    fn if_else3<Res: SimdValue<SimdBool = Self>>(
        self,
        if_value: impl FnOnce() -> Res,
        else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
        else_value: impl FnOnce() -> Res,
    ) -> Res {
        if self {
            if_value()
        } else if else_if.0() {
            else_if.1()
        } else if else_else_if.0() {
            else_else_if.1()
        } else {
            else_value()
        }
    }
}
