module Zygote_Ext
using Zygote: @adjoint

import GFDomains:
    analysis_scalar,
    analysis_vector, # TODO
    analysis_div,
    synthesis_scalar,
    synthesis_vector, # TODO
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

@adjoint analysis_scalar(spat, sph, backend) = analysis_scalar(protect(spat), sph, backend),
(spec) -> adjoint_analysis_scalar(protect(spec), sph, backend)

@adjoint synthesis_scalar(spec, sph, backend) = synthesis_scalar(protect(spec), sph, backend),
(spat) -> adjoint_synthesis_scalar(protect(spat), sph, backend)

function adjoint_analysis_scalar(spec, sph, backend)
    scale_m0!(spec, sph, 2.0)
    return synthesis_scalar(spec, sph, backend), nothing, nothing
end

function adjoint_synthesis_scalar(spat, sph, backend)
    spec = analysis_scalar(spat, sph, backend)
    scale_m0!(spec, sph, 0.5)
    return spec, nothing, nothing
end

#================= vector =================#

@adjoint analysis_div(uv, sph, backend) = analysis_div(protect(uv), sph, backend),
(div_spec) -> adjoint_analysis_div(protect(div_spec), sph, backend)

@adjoint synthesis_spheroidal(phi_spec, sph, backend) = synthesis_spheroidal(protect(phi_spec), sph, backend),
(uv_spat) -> adjoint_synthesis_spheroidal(protect(uv_spat), sph, backend)

function adjoint_analysis_div(spec, sph, backend)
    scale_m0!(spec, sph, 2.0)
    ucolat, ulon = synthesis_spheroidal(spec, sph, backend)
    return (ucolat = -ucolat, ulon = -ulon), nothing, nothing
end

function adjoint_synthesis_spheroidal(uv_spat, sph, backend)
    spec = analysis_div(uv_spat, sph, backend)
    spec = scale_m0!(spec, sph, 0.5)
    return -spec, nothing, nothing
end

end
