module Adjoints

using SHTnsSpheres: void, Void, Writable

using SHTnsSpheres: 
    analysis_scalar!,
    analysis_vector!,
    synthesis_scalar!,
    synthesis_vector!,
    synthesis_spheroidal!

function scale_m0!(spec, sph, fac)
    for i in 1:sph.mmax+1
        spec[i]*=fac
    end
    return spec
end

#=================== scalar ==============#

function adjoint_analysis_scalar(spec, sph)
    spec = copy(spec)
    scale_m0!(spec, sph, 2.0)
    return nothing, synthesis_scalar!(void, spec, sph), nothing, nothing
end

function adjoint_synthesis_scalar(spat, sph)
    spec = analysis_scalar!(void, spat, sph)
    scale_m0!(spec, sph, 0.5)
    return nothing, spec, nothing, nothing
end

#================= vector =================#

function adjoint_synthesis_vector(uv_spat, sph)
    spec = analysis_vector!(void, uv_spat, sph)
    scale_m0!(spec, sph, 0.5)
    @. spec *= -sph.laplace
    return nothing, spec, nothing, nothing
end

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

function adjoint_synthesis_spheroidal(uv_spat, sph)
    spec = analysis_vector!(void, uv_spat, sph).spheroidal
    scale_m0!(spec, sph, 0.5)
    @. spec *= -sph.laplace
    return nothing, spec, nothing, nothing
end

end # module
