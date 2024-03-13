# SHTnsSpheres

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/SHTnsSpheres.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/SHTnsSpheres.jl/dev/)
[![Build Status](https://github.com/ClimFlows/SHTnsSpheres.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/SHTnsSpheres.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/SHTnsSpheres.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/SHTnsSpheres.jl)

Interface to the [SHTns](https://nschaeff.bitbucket.io/shtns/) spherical harmonics library.
Provides [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) tangents and [Zygote](https://github.com/FluxML/Zygote.jl) adjoints.

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
using SHTnsSpheres: SHTnsSphere, analysis_scalar, synthesis_scalar
sphere = SHTnsSphere(21)
```
See also the [unit tests](test/runtests.jl).

## Terminology and data structures

The spectral representation of a scalar field is a `Vector` of complex spectral coefficients. The grid representation of a scalar field is a `Matrix` of real values at grid points.

The spectral representation of a vector field is a named tuple `(; toroidal, spheroidal)` of complex `Vector`s corresponding to a vector potential and streamfunction. The grid representation of a vector field is a named tuple `(; ucolat, ulon)` of real matrices storing grid-point values of vector components in colatitude, longitude coordinates.

`synthesis_XX` takes spectral coefficients and returns grid-point values. `analysis_XX` goes in the reverse direction.

## Warning

Following julia convention, `analysis_XX` returns a newly allocated object while `analysis!_XX` is an non-allocating transform whose first argument is caller-allocated storage for the output.

**`SHTns` may modify input argument(s)** and fill them with garbage. *If you want to protect your `data`, you should pass `copy(data)` as input.*

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
