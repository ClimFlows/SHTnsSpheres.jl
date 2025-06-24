# SHTnsSpheres

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/SHTnsSpheres.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/SHTnsSpheres.jl/dev/)
[![Build Status](https://github.com/ClimFlows/SHTnsSpheres.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/SHTnsSpheres.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/SHTnsSpheres.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/SHTnsSpheres.jl)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ClimFlows/SHTnsSpheres.jl)

Interface to the [SHTns](https://nschaeff.bitbucket.io/shtns/) spherical harmonics library.
Compatible with forward differentiation by [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) and backward differentiation with [Zygote](https://github.com/FluxML/Zygote.jl) and [Enzyme](https://github.com/EnzymeAD/Enzyme.jl).

## Installation

`SHTnsSpheres` is registered in the ClimFlows [repository](https://github.com/ClimFlows/JuliaRegistry). Follow instructions there, then:
```julia
] add SHTnsSpheres
```

## Using `SHTnsSpheres`

Following the principle that *explicit is better than implicit*, `SHTnsSpheres` does not export symbols. You can discover available symbols with TAB completion in the REPL.

```julia
using SHTnsSpheres
sphere = SHTnsSpheres.SHTnsSphere(21)
```
You may also explicitly list the objects you use:
```julia
using SHTnsSpheres: SHTnsSphere, sample_scalar!, analysis_scalar!, synthesis_scalar!
sphere = SHTnsSphere(21)
```
See also the [unit tests](test/runtests.jl).

## Terminology and data structures

The spectral representation of a scalar field is a `Vector` of complex spectral coefficients. The grid representation of a scalar field is a `Matrix` of real values at grid points.

The spectral representation of a vector field is a named tuple `(; toroidal, spheroidal)` of complex `Vector`s corresponding to a vector potential and streamfunction. The grid representation of a vector field is a named tuple `(; ucolat, ulon)` of real matrices storing grid-point values of vector components in colatitude, longitude coordinates.

`synthesis_XX!` takes spectral coefficients and returns grid-point values. `analysis_XX!` goes in the reverse direction.

## Mutating and non-mutating variants

Following julia convention, the first argument of functions whose name ends with `!` is an output argument, as in `copy!` or `mul!`. If an existing field is passed as this argument, the function call should not allocate. This is the default, mutating variant. However automatic differentiation in reverse mode prefers pure, non-mutating functions. The non-mutating variant is obtained by passing `void` as the first argument.

```julia
using SHTnsSpheres: void
f(x,y,z,lon,lat) = x*z
g(x,y,z,lon,lat) = x*y
xy_spat = sample_scalar!(void, f, sphere)         # non-mutating, allocates xy_spat
xy_spec = analysis_scalar!(void, xy_spat, sphere) # non-mutating, allocates xy_spec
xy_spat = sample_scalar!(xy_spat, g, sphere)      # mutating, does not allocate
```

**`SHTns` may modify input argument(s)** and fill them with garbage. To protect your data, a copy is made
where necessary, but this allocates. If it is acceptable for you to lose your input data, you can avoid this allocation by passing `erase(data)` as argument. `erase(data)` returns a thin wrapper around `data` that tells `SHTnsSpheres` that a copy is not required.

```julia
using SHTnsSphere: erase
# non-mutating, allocates xy_spec, internally allocates a copy of xy_spat
xy_spec = analysis_scalar!(void, xy_spat, sphere)
# mutating, non-allocating but fills xy_spat with garbage
xy_spec = analysis_scalar!(xy_spec, erase(xy_spat), sphere)
```

## Automatic differentiation (AD)

`SHTnsSpheres` works with ForwardDiff (forward AD) and `Zygote` (reverse AD).

For reverse AD, `SHTnsSpheres` guarantees that non-mutating variants do not modify their input arguments, even if passed as `erase(data)`. Thus a copy may be made internally. This hopefully prevents silent AD errors, at the price of extra allocations.

## Thread safety

At this moment `SHTnsSpheres` makes no promises regarding thread safety. Use a lock if you use the same `SHTnsSphere` object from different threads.

## Credits

If you use this package for research, you are using SHTns and are kindly invited to cite:

        @article {shtns,
          author = {Schaeffer, Nathanael},
          title = {Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations},
          journal = {Geochemistry, Geophysics, Geosystems},
          doi = {10.1002/ggge.20071}, volume = {14}, number = {3}, pages = {751--758},
          year = {2013},
        }

If you use Ishioka's recurrence (the default since SHTns v3.4), you may also want to cite his paper:

        @article {ishioka2018,
          author={Ishioka, Keiichi},
          title={A New Recurrence Formula for Efficient Computation of Spherical Harmonic Transform},
          journal={Journal of the Meteorological Society of Japan},
          doi = {10.2151/jmsj.2018-019}, volume={96}, number={2}, pages={241--249},
          year={2018},
        }
