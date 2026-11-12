# Changelog

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
