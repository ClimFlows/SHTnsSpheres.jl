module Adjoints

using SHTnsSpheres: void, erase

using SHTnsSpheres: 
    analysis_scalar!,
    analysis_vector!,
    synthesis_scalar!,
    synthesis_vector!,
    synthesis_spheroidal!

# Positive orders represent both +m and -m in the real transforms.
function scale_positive_orders!(spec, sph, fac)
    columns = reshape(spec, size(spec, 1), :)
    modes = @view columns[sph.lmax+2:end, :]
    modes .*= fac
    return spec
end

# The adjoints use unweighted array dot products. Analysis already
# applies Gauss quadrature, so its adjoint multiplies by the weights
# after synthesis, and the adjoint of synthesis divides before analysis.
weight_spat!(spat, weights) = (spat .*= weights)
weight_spat!(spat::NamedTuple, weights) = map(a -> weight_spat!(a, weights), spat)
unweight_spat(spat, weights) = spat ./ weights
unweight_spat(spat::NamedTuple, weights) = map(a -> unweight_spat(a, weights), spat)

rcopy(x) = copy(x)
rcopy(::Nothing) = nothing
rcopy(x::NamedTuple) = map(rcopy, x)

#=================== scalar ==============#

function adjoint_analysis_scalar(spec, sph)
    spec = rcopy(spec)
    scale_positive_orders!(spec, sph, 0.5)
    spat = synthesis_scalar!(void, spec, sph)
    return nothing, weight_spat!(spat, sph.weights), nothing, nothing
end

function adjoint_synthesis_scalar(spat, sph)
    spec = analysis_scalar!(void, erase(unweight_spat(spat, sph.weights)), sph)
    scale_positive_orders!(spec, sph, 2)
    return nothing, spec, nothing, nothing
end

#================= vector =================#

function adjoint_analysis_vector(spec, sph)
    spec = rcopy(spec)
    (; toroidal, spheroidal) = spec
    if !isnothing(spheroidal)
        scale_positive_orders!(spheroidal, sph, 0.5)
        @. spheroidal *= -sph.poisson
    end
    if !isnothing(toroidal)
        scale_positive_orders!(toroidal, sph, 0.5)
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
    return nothing, weight_spat!(spat, sph.weights), nothing, nothing
end

function adjoint_synthesis_vector(uv_spat, sph)
    spec = analysis_vector!(void, erase(unweight_spat(uv_spat, sph.weights)), sph)
    (; toroidal, spheroidal) = spec
    scale_positive_orders!(toroidal, sph, 2)
    scale_positive_orders!(spheroidal, sph, 2)
    @. toroidal *= -sph.laplace
    @. spheroidal *= -sph.laplace
    return nothing, spec, nothing, nothing
end

function adjoint_synthesis_spheroidal(uv_spat, sph)
    spec = analysis_vector!(void, erase(unweight_spat(uv_spat, sph.weights)), sph).spheroidal
    scale_positive_orders!(spec, sph, 2)
    @. spec *= -sph.laplace
    return nothing, spec, nothing, nothing
end

end # module
