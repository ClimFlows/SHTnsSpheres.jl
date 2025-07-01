#============ higher-dimensional transforms ===========#

# analysis_XXX!  -> analysis!  -> transform! ( -> batch ) -> low-level function
# synthesis_XXX! -> synthesis! -> transform! ( -> batch ) -> low-level function

# spectral coefficients are not erased by SHTns
# => always pass them as is, even if marked as writeable
function transform!(fun, sph, spec, spat) 
    op(::Writable) = erase  
    op(::Any) = identity
    data(x::Writable) = x.data
    data(x) = x
    return transform!(fun, sph, data(spec), data(spat), op(spat))
end

function analysis!(fun::Fun, sph, spec, spat) where Fun
    spat = writable(spat)
    spec = similar_spec!(spec, spat, sph)
    @assert extra_axes(spec) == extra_axes(spat) "$(extra_axes(spec)) != $(extra_axes(spat))"
    transform!(fun, sph, spec, spat)
    return spec
end

function synthesis!(fun::Fun, sph, spec, spat) where Fun
    spat = similar_spat!(spat, spec, sph)
    @assert extra_axes(spec) == extra_axes(spat)
    transform!(fun, sph, spec, spat)
    return spat
end

function batch(fun, sph, nk, nl)
    nthreads = length(sph.ptrs)
    @batch per=core for thread in 1:nthreads
        ptr = sph.ptrs[thread]
        start, stop = div(nk*(thread-1), nthreads), div(nk*thread, nthreads)
        for k in start+1:stop, l=1:nl
            fun(ptr, thread, k, l)
        end
    end
end

extra_axes(spec::SHTVectorSpec) = extra_axes(spec.toroidal, spec.spheroidal)
extra_axes(spat::SHTVectorSpat) = extra_axes(spat.ucolat, spat.ulon)
extra_axes(x::Writable) = extra_axes(x.data)

function extra_axes(a, b)
    @assert extra_axes(a) == extra_axes(b) "$(extra_axes(a)) != $(extra_axes(b))"
    return extra_axes(a)
end

function extra_axes(spec::AbstractArray{ComplexF64})
    @assert ndims(spec) <= 3 "$(ndims(spec))>3"
    return axes(spec,2), axes(spec, 3)
end

function extra_axes(spat::AbstractArray{Float64})
    @assert ndims(spat) <= 4 "$(ndims(spat))>4"
    return axes(spat,3), axes(spat, 4)
end

get_ptr(sph::SHTnsSphere) = sph.ptr
get_ptr(ptr::Ptr{priv.shtns_info}) = ptr

const SPtr = Union{SHTnsSphere, Ptr{priv.shtns_info}}

#========= scalar synthesis / analysis ========#

function transform!(fun, sph, spec::AbstractArray{ComplexF64}, spat::AbstractArray{Float64}, op::Fun) where Fun
    if length(spec) == size(spec, 1)
        fun(get_ptr(sph), spec, op(spat))
    else # batched transform
        batch(sph, size(spec,2), size(spec,3)) do ptr, _, k, l
            @views fun(ptr, spec[:,k,l], op(spat[:,:,k,l]))
        end
    end
    return nothing
end

synthesis_scalar!(spat, spec::In{<:AbstractArray{ComplexF64}}, sph::SPtr) =
    synthesis!(priv.SH_to_spat, sph, spec, spat)

analysis_scalar!(spec, spat::In{<:AbstractArray{Float64}}, sph::SPtr) =
    analysis!(priv.spat_to_SH, sph, spec, spat)

#========= vector synthesis / analysis ========#

function transform!(fun, sph, spec::SHTVectorSpec{Float64}, spat::SHTVectorSpat{Float64}, op::Fun) where Fun
    @assert axes(spec.toroidal) == axes(spec.spheroidal)
    if length(spec.toroidal) == size(spec.toroidal, 1)
        fun(get_ptr(sph), spec.spheroidal, spec.toroidal, op(spat.ucolat), op(spat.ulon))
    else
        batch(sph, size(spec.toroidal,2), size(spec.toroidal,3)) do ptr, _, k, l
            @views fun(ptr, 
                spec.spheroidal[:,k,l], spec.toroidal[:,k,l], 
                op_spat(spat.ucolat[:,:,k,l]), op_spat(spat.ulon[:,:,k,l]))
        end
    end
    return nothing
end

synthesis_vector!(spat, spec::In{<:SHTVectorSpec{Float64}}, sph::SPtr) =
    synthesis!(priv.SHsphtor_to_spat, sph, spec, spat)

analysis_vector!(spec, spat::In{<:SHTVectorSpat{Float64}}, sph::SPtr) =
    analysis!(priv.spat_to_SHsphtor, sph, spec, spat)

#===================== gradient synthesis ==========================#

function transform!(fun, sph, spec::AbstractArray{ComplexF64}, spat::SHTVectorSpat)
    if length(spec) == size(spec, 1)
        fun(get_ptr(sph), spec, spat.ucolat, spat.ulon)
    else
        batch(sph, size(spec,2), size(spec,3)) do ptr, _, k, l
            @views fun(ptr, spec[:,k,l], spat.ucolat[:,:,k,l], spat.ulon[:,:,k,l])
        end
    end
    return nothing
end

function synthesis_spheroidal!(spat, spec::AbstractArray{ComplexF64}, sph::SPtr)
    if spat isa Void
        spat = (ucolat = similar_spat(spec, sph), ulon=similar_spat(spec, sph))
    end
    synthesis!(priv.SHsph_to_spat, sph, spec, spat)
end

#========= curl, div ========#

# due to SHTns sign convention, ζ=-ΔT with T the toroidal component
# see https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/vsh.html
curl!(spec_out, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. spec_out = spec_in.toroidal * (-sph.laplace)
curl!(::Void, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. -spec_in.toroidal * sph.laplace

divergence!(spec_out, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. spec_out = spec_in.spheroidal * sph.laplace
divergence!(::Void, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. spec_in.spheroidal * sph.laplace
