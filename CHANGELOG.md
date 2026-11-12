# Changelog

## 0.2.20

### Changed

- Transforms on device-memory arrays enqueue asynchronously on the sphere's
  CUDA stream with CUDA.jl 5.8.3 or later. Synchronization is retained for
  older CUDA.jl releases and other memory types.
- Closing a CUDA sphere waits for its stream before releasing the plans.
- The Enzyme extension requires Enzyme 0.13.24 or later in the 0.13 series.

## 0.2.19

### Added

- CUDA spheres infer the batch size from the input and cache one transform
  plan per size. The `batch` keyword accepts an integer or a nonempty tuple or
  vector of initial sizes. Additional sizes are planned on demand, and
  `close(sphere)` releases all plans.

## 0.2.18

### Added

- Native CUDA support for scalar and vector spherical harmonic transforms in
  Float32 and Float64, including batched fields.
- Reverse-mode differentiation of CUDA transforms with Enzyme and Zygote.

### Fixed

- Reverse rules now compute adjoints for the unweighted array dot product and
  apply spectral normalization to every field in a batch.
