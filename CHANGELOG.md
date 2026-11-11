# Changelog

## 0.2.18

### Added

- Native CUDA support for scalar and vector spherical harmonic transforms in
  Float32 and Float64, including batched fields.
- Reverse-mode differentiation of CUDA transforms with Enzyme and Zygote.

### Fixed

- Reverse rules now compute adjoints for the unweighted array dot product and
  apply spectral normalization to every field in a batch.
