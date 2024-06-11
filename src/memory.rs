use num::{NumCast, Zero, Float};
use std::{
    fmt::{Debug, Display, LowerExp}, iter::Sum, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign}, simd::{num::SimdFloat, LaneCount, SupportedLaneCount}
};
use rand::distributions::uniform::SampleUniform;
use std::simd::{Simd, SimdElement};

// TODO: Remove this and use const_generics, as soon as they are stable and the compiler stops crashing :)
pub(crate) const LANES: usize = 8;

pub trait Primitive: Add + AddAssign + Sum + Sub + SubAssign + Zero + Float + NumCast + SampleUniform
                + PartialOrd + Copy + Default + Display + Debug + Sync + Send + LowerExp + 'static
                + for<'a> AddAssign<&'a Self> + for<'a> Sub<&'a Self> {}
impl Primitive for f32 {}
impl Primitive for f64 {}

pub trait SimdWrapper<T, const LANES: usize>: Sized + Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign
    + Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign + Sum
where
    T: SimdElement,
{
    unsafe fn from_slice_aligned_unchecked(src: &[T]) -> Self;
    unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [T]);
    fn splat(single: T) -> Self;
    fn sum(self) -> T;
}

impl<const LANES: usize> SimdWrapper<f64, LANES> for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    unsafe fn from_slice_aligned_unchecked(src: &[f64]) -> Self {
        Simd::from_slice_aligned_unchecked(src)
    }

    #[inline(always)]
    unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [f64]) {
        self.write_to_slice_aligned_unchecked(slice);
    }

    #[inline(always)]
    fn splat(single: f64) -> Self {
        Simd::splat(single)
    }

    #[inline(always)]
    fn sum(self) -> f64 {
        self.reduce_sum()
    }
}

impl<const LANES: usize> SimdWrapper<f32, LANES> for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    unsafe fn from_slice_aligned_unchecked(src: &[f32]) -> Self {
        Simd::from_slice_aligned_unchecked(src)
    }

    #[inline(always)]
    unsafe fn write_to_slice_aligned_unchecked(self, slice: &mut [f32]) {
        self.write_to_slice_aligned_unchecked(slice);
    }

    #[inline(always)]
    fn splat(single: f32) -> Self {
        Simd::splat(single)
    }

    #[inline(always)]
    fn sum(self) -> f32 {
        self.reduce_sum()
    }
}


pub(crate) struct AlignedFloatVec<const LANES: usize>;
impl <const LANES: usize> AlignedFloatVec<LANES> {
    pub fn new<T: Primitive>(size: usize) -> Vec<T> {
        use std::alloc::{alloc_zeroed, Layout};

        assert_eq!(size % LANES, 0);
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), LANES * std::mem::size_of::<T>())
            .expect("Illegal aligned allocation");
        unsafe {
            let aligned_ptr = alloc_zeroed(layout) as *mut T;
            let resvec = Vec::from_raw_parts(aligned_ptr, size, size);
            debug_assert_eq!((resvec.get_unchecked(0) as *const T).align_offset(LANES * std::mem::size_of::<T>()), 0);
            resvec
        }
    }
    pub fn new_uninitialized<T: Primitive>(size: usize) -> Vec<T> {
        use std::alloc::{alloc, Layout};

        assert_eq!(size % LANES, 0);
        let layout = Layout::from_size_align(size * std::mem::size_of::<T>(), LANES * std::mem::size_of::<T>())
            .expect("Illegal aligned allocation");
        unsafe {
            let aligned_ptr = alloc(layout) as *mut T;
            let resvec = Vec::from_raw_parts(aligned_ptr, size, size);
            debug_assert_eq!((resvec.get_unchecked(0) as *const T).align_offset(LANES * std::mem::size_of::<T>()), 0);
            resvec
        }
    }
}