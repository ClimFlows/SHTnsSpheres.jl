module Zygote_Ext
using Zygote: @adjoint

using SHTnsSpheres: void, Void, Writable, Adjoints

import SHTnsSpheres: 
    analysis_scalar!,
    analysis_vector!,
    synthesis_scalar!,
    synthesis_vector!,
    synthesis_spheroidal!

"""
    y = protect(x)
Makes sure x is not modified even if x::Writable. Used internally for reverse AD.
"""
protect(x::NamedTuple) = map(protect, x)
protect(x::Writable) = x.data
protect(x) = x

#=================== scalar ==============#

isvoid(::Void, fun) = void
isvoid(storage, fun) = throw(ArgumentError("""
    For reverse AD of SHTnsSpheres with Zygote, only non-mutating functions are supported.
    `SHTnsSpheres.$(fun)` has been called with non-void first argument of type `$(typeof(storage))`. Please make sure the first argument is `void` or <:Void`.
    """))

@adjoint analysis_scalar!(out, spat, sph) =
    analysis_scalar!(isvoid(out, analysis_scalar!), protect(spat), sph),
    (spec -> Adjoints.adjoint_analysis_scalar(protect(spec), sph))

@adjoint synthesis_scalar!(out, spec, sph) =
    synthesis_scalar!(isvoid(out, synthesis_scalar!), protect(spec), sph),
    (spat -> Adjoints.adjoint_synthesis_scalar(protect(spat), sph))

#================= vector =================#

@adjoint synthesis_vector!(out, spec, sph) =
    synthesis_vector!(isvoid(out, synthesis_vector!), protect(spec), sph),
    (uv_spat) -> Adjoints.adjoint_synthesis_vector(protect(uv_spat), sph)

@adjoint analysis_vector!(out, spat, sph) =
    analysis_vector!(isvoid(out, analysis_vector!), protect(spat), sph),
    (uv_spec) -> Adjoints.adjoint_analysis_vector(protect(uv_spec), sph)

@adjoint synthesis_spheroidal!(out, phi_spec, sph) =
    synthesis_spheroidal!(isvoid(out, synthesis_spheroidal!), protect(phi_spec), sph),
    (uv_spat) -> Adjoints.adjoint_synthesis_spheroidal(protect(uv_spat), sph)

end
