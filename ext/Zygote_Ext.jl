module Zygote_Ext
using Zygote: @adjoint

import SHTnsSpheres: void, Void, InOut,
    analysis_scalar!,
    analysis_vector!, # TODO
    synthesis_scalar!,
    synthesis_vector!, # TODO
    synthesis_spheroidal!

function scale_m0!(spec, sph, fac)
    for i in 1:sph.mmax+1
        spec[i]*=fac
    end
    return spec
end

"""
    y = protect(x)
Makes sure x is not modified even if x::InOut. Used internally for reverse AD.
"""
protect(x::NamedTuple) = map(protect, x)
protect(x::InOut) = x.data
protect(x) = x

#=================== scalar ==============#

isvoid(::Void, fun) = void
isvoid(storage, fun) = throw(ArgumentError("""
    For reverse AD of SHTnsSpheres with Zygote, only non-mutating functions are supported.
    `SHTnsSpheres.$(fun)` has been called with non-void first argument of type `$(typeof(storage))`. Please make sure the first argument is `void` or <:Void`.
    """))

@adjoint analysis_scalar!(out, spat, sph) =
    analysis_scalar!(isvoid(out, analysis_scalar!), protect(spat), sph),
    (spec -> adjoint_analysis_scalar(protect(spec), sph))

function adjoint_analysis_scalar(spec, sph)
    scale_m0!(spec, sph, 2.0)
    return nothing, synthesis_scalar!(void, spec, sph), nothing, nothing
end

@adjoint synthesis_scalar!(out, spec, sph) =
    synthesis_scalar!(isvoid(out, synthesis_scalar!), protect(spec), sph),
    (spat -> adjoint_synthesis_scalar(protect(spat), sph))

function adjoint_synthesis_scalar(spat, sph)
    spec = analysis_scalar!(void, spat, sph)
    scale_m0!(spec, sph, 0.5)
    return nothing, spec, nothing, nothing
end

#================= vector =================#

@adjoint synthesis_vector!(out, spec, sph) =
    synthesis_vector!(isvoid(out, synthesis_vector!), protect(spec), sph),
    (uv_spat) -> adjoint_synthesis_vector(protect(uv_spat), sph)

function adjoint_synthesis_vector(uv_spat, sph)
    spec = analysis_vector!(void, uv_spat, sph)
    scale_m0!(spec, sph, 0.5)
    @. spec *= -sph.laplace
    return nothing, spec, nothing, nothing
end

@adjoint analysis_vector!(out, spat, sph) =
    analysis_vector!(isvoid(out, analysis_vector!), protect(spat), sph),
    (uv_spec) -> adjoint_analysis_vector(protect(uv_spec), sph)

function adjoint_analysis_vector((; toroidal, spheroidal), sph)
    if !isnothing(spheroidal)
        scale_m0!(spheroidal, sph, 2)
        @. spheroidal *= -sph.poisson
    end
    if !isnothing(toroidal)
        scale_m0!(toroidal, sph, 2)
        @. toroidal *= -sph.poisson
    end
    if isnothing(toroidal)
        spat = synthesis_spheroidal!(void, spheroidal, sph)
    elseif isnothing(spheroidal)
        spat = synthesis_spheroidal!(void, toroidal, sph)
        spat = (ucolat=spat.ulon, ulon=-spat.ucolat)
    else
        spat = synthesis_vector!(void, spec, sph)
    end
    return nothing, spat, nothing, nothing
end

@adjoint synthesis_spheroidal!(out, phi_spec, sph) =
    synthesis_spheroidal!(isvoid(out, synthesis_spheroidal!), protect(phi_spec), sph),
    (uv_spat) -> adjoint_synthesis_spheroidal(protect(uv_spat), sph)

function adjoint_synthesis_spheroidal(uv_spat, sph)
    spec = analysis_vector!(void, uv_spat, sph).spheroidal
    scale_m0!(spec, sph, 0.5)
    @. spec *= -sph.laplace
    return nothing, spec, nothing, nothing
end

end
