module Zygote_Ext
using Zygote: @adjoint

import SHTnsSpheres: void, Void,
    analysis_scalar!,
    analysis_vector!, # TODO
    analysis_div,
    synthesis_scalar!,
    synthesis_vector!, # TODO
    synthesis_spheroidal

function scale_m0!(spec, sph, fac)
    for i in 1:sph.mmax+1
        spec[i]*=fac
    end
    return spec
end

"""
SHTns usually overwrites the input array.
This may confuse Zygote which assumes pure functions.
So we make copies of input arguments.
"""
protect(x::AbstractArray) = copy(x)
protect(x::NamedTuple) = map(protect, x)

#=================== scalar ==============#

@adjoint analysis_scalar!(::Void, spat, sph) =
    analysis_scalar!(void, protect(spat), sph),
    (spec -> adjoint_analysis_scalar(protect(spec), sph))

@adjoint synthesis_scalar!(::Void, spec, sph) =
    synthesis_scalar!(void, protect(spec), sph),
    (spat -> adjoint_synthesis_scalar(protect(spat), sph))

function adjoint_analysis_scalar(spec, sph)
    scale_m0!(spec, sph, 2.0)
    return nothing, synthesis_scalar!(void, spec, sph), nothing, nothing
end

function adjoint_synthesis_scalar(spat, sph)
    spec = analysis_scalar!(void, spat, sph)
    scale_m0!(spec, sph, 0.5)
    return nothing, spec, nothing, nothing
end

#================= vector =================#

@adjoint analysis_div(uv, sph) = analysis_div(protect(uv), sph),
(div_spec) -> adjoint_analysis_div(protect(div_spec), sph)

@adjoint synthesis_spheroidal(phi_spec, sph) = synthesis_spheroidal(protect(phi_spec), sph),
(uv_spat) -> adjoint_synthesis_spheroidal(protect(uv_spat), sph)

function adjoint_analysis_div(spec, sph)
    scale_m0!(spec, sph, 2.0)
    ucolat, ulon = synthesis_spheroidal(spec, sph)
    return (ucolat = -ucolat, ulon = -ulon), nothing, nothing
end

function adjoint_synthesis_spheroidal(uv_spat, sph)
    spec = analysis_div(uv_spat, sph)
    spec = scale_m0!(spec, sph, 0.5)
    return -spec, nothing, nothing
end

end
