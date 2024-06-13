# kmeans-smid
[![Current Crates.io Version](https://img.shields.io/crates/v/kmeans-smid.svg)](https://crates.io/crates/kmeans-smid)
[![docs](https://docs.rs/kmeans-smid/badge.svg)](https://docs.rs/kmeans-smid/latest/kmeans/)

kmeans-smid is a small and fast library for k-means clustering calculations. It fixes smid problem from kmeans crate.
Here is a small example, using kmean++ as initialization method and lloyd as k-means variant:

```rust
use kmeans::*;

fn main() {
    let (sample_cnt, sample_dims, k, max_iter) = (20000, 200, 4, 100);

    // Generate some random data
    let mut samples = vec![0.0f64;sample_cnt * sample_dims];
    samples.iter_mut().for_each(|v| *v = rand::random());

    // Calculate kmeans, using kmean++ as initialization-method
    let kmean = KMeans<f64, 8>::new(samples, sample_cnt, sample_dims);
    let result = kmean.kmeans_lloyd(k, max_iter, KMeans::init_kmeanplusplus, &KMeansConfig::default());

    println!("Centroids: {:?}", result.centroids);
    println!("Cluster-Assignments: {:?}", result.assignments);
    println!("Error: {}", result.distsum);
}
```


## Datastructures
For performance-reasons, all calculations are done on bare vectors, using hand-written SIMD intrinsics from the `packed_simd` crate. All vectors are stored row-major, so each sample is stored in a consecutive block of memory.

## Supported variants / algorithms
- lloyd (standard kmeans)
- minibatch

## Supported centroid initialization methods
- KMean++
- random partition
- random sample
