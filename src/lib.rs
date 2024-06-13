#![feature(test)]
#![feature(portable_simd)]

#![doc = include_str!("../README.md")]

extern crate test;
#[macro_use]
mod helpers;
mod abort_strategy;
mod api;
mod inits;
mod memory;
mod variants;

pub use abort_strategy::AbortStrategy;
pub use api::{KMeans, KMeansConfig, KMeansConfigBuilder, KMeansState};
pub use memory::Primitive;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::*;
    use rand::prelude::*;
    use std::iter::Sum;
    use std::ops::{Add, Div, Mul, Sub};
    use std::simd::{num::SimdFloat, LaneCount, Simd, SimdElement, SupportedLaneCount};
    use test::Bencher;

    #[bench]
    fn complete_benchmark_lloyd_small_f64(b: &mut Bencher) {
        complete_benchmark_lloyd::<f64, LANES>(b, 200, 2000, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_mid_f64(b: &mut Bencher) {
        complete_benchmark_lloyd::<f64, LANES>(b, 2000, 200, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_big_f64(b: &mut Bencher) {
        complete_benchmark_lloyd::<f64, LANES>(b, 10000, 8, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_huge_f64(b: &mut Bencher) {
        complete_benchmark_lloyd::<f64, LANES>(b, 20000, 256, 1, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_small_f32(b: &mut Bencher) {
        complete_benchmark_lloyd::<f32, LANES>(b, 200, 2000, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_mid_f32(b: &mut Bencher) {
        complete_benchmark_lloyd::<f32, LANES>(b, 2000, 200, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_big_f32(b: &mut Bencher) {
        complete_benchmark_lloyd::<f32, LANES>(b, 10000, 8, 10, 32);
    }
    #[bench]
    fn complete_benchmark_lloyd_huge_f32(b: &mut Bencher) {
        complete_benchmark_lloyd::<f32, LANES>(b, 20000, 256, 1, 32);
    }
    fn complete_benchmark_lloyd<T: Primitive, const LANES: usize>(
        b: &mut Bencher,
        sample_cnt: usize,
        sample_dims: usize,
        max_iter: usize,
        k: usize,
    ) where
        T: SimdElement
            + Copy
            + Default
            + Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sub<Output = T>
            + Sum
            + Primitive
            + num::Zero
            + num::One,
        Simd<T, LANES>: Sub<Output = Simd<T, LANES>>
            + Add<Output = Simd<T, LANES>>
            + Mul<Output = Simd<T, LANES>>
            + Div<Output = Simd<T, LANES>>
            + Sum
            + SimdFloat<Scalar = T>,
        LaneCount<LANES>: SupportedLaneCount,
    {
        let mut rnd = StdRng::seed_from_u64(1337);
        let mut samples = vec![T::zero(); sample_cnt * sample_dims];
        samples
            .iter_mut()
            .for_each(|v| *v = rnd.gen_range(T::zero()..T::one()));
        let kmean: KMeans<T, LANES> = KMeans::new(samples, sample_cnt, sample_dims);
        let conf = KMeansConfig::build().random_generator(rnd).build();
        b.iter(|| kmean.kmeans_lloyd(k, max_iter, KMeans::init_kmeanplusplus, &conf));
    }

    #[bench]
    fn complete_benchmark_minibatch_small_f64(b: &mut Bencher) {
        complete_benchmark_minibatch::<f64, LANES>(b, 30, 200, 2000, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_mid_f64(b: &mut Bencher) {
        complete_benchmark_minibatch::<f64, LANES>(b, 200, 2000, 200, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_big_f64(b: &mut Bencher) {
        complete_benchmark_minibatch::<f64, LANES>(b, 1000, 10000, 8, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_huge_f64(b: &mut Bencher) {
        complete_benchmark_minibatch::<f64, LANES>(b, 2000, 20000, 256, 30, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_small_f32(b: &mut Bencher) {
        complete_benchmark_minibatch::<f32, LANES>(b, 30, 200, 2000, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_mid_f32(b: &mut Bencher) {
        complete_benchmark_minibatch::<f32, LANES>(b, 200, 2000, 200, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_big_f32(b: &mut Bencher) {
        complete_benchmark_minibatch::<f32, LANES>(b, 1000, 10000, 8, 100, 32);
    }
    #[bench]
    fn complete_benchmark_minibatch_huge_f32(b: &mut Bencher) {
        complete_benchmark_minibatch::<f32, LANES>(b, 2000, 20000, 256, 30, 32);
    }
    fn complete_benchmark_minibatch<T: Primitive, const LANES: usize>(
        b: &mut Bencher,
        batch_size: usize,
        sample_cnt: usize,
        sample_dims: usize,
        max_iter: usize,
        k: usize,
    ) where
        T: SimdElement
            + Copy
            + Default
            + Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Sub<Output = T>
            + Sum
            + Primitive
            + num::Zero
            + num::One,
        Simd<T, LANES>: Sub<Output = Simd<T, LANES>>
            + Add<Output = Simd<T, LANES>>
            + Mul<Output = Simd<T, LANES>>
            + Div<Output = Simd<T, LANES>>
            + Sum
            + SimdFloat<Scalar = T>,
        LaneCount<LANES>: SupportedLaneCount,
    {
        let mut rnd = StdRng::seed_from_u64(1337);
        let mut samples = vec![T::zero(); sample_cnt * sample_dims];
        samples
            .iter_mut()
            .for_each(|v| *v = rnd.gen_range(T::zero()..T::one()));
        let kmean = KMeans::new(samples, sample_cnt, sample_dims);
        let conf = KMeansConfig::build().random_generator(rnd).build();
        b.iter(|| {
            kmean.kmeans_minibatch(batch_size, k, max_iter, KMeans::init_random_sample, &conf)
        });
    }
}
