module SHTnsSpheres

using MutatingOrNot: void, Void
using Polyester: @batch

module priv
using SHTns_jll
include("julia/SHTns.jl")
export AF64, AC64, MF64, VC64
end

using .priv

"""
    num_threads = shtns_use_threads(num_threads)

Call *before* any initialization of shtns to use mutliple threads. Returns the actual number of threads.
    If num_threads > 0, specifies the maximum number of threads that should be used.
    If num_threads <= 0, maximum number of threads is automatically set to the number of processors.
    If num_threads == 1, openmp will be disabled.
"""
shtns_use_threads(n::Int=0) = ccall((:shtns_use_threads, :libshtns), Cint, (Cint,), n)

include("julia/util.jl")

#==================================================================#

struct SHTnsSphere
    ptrs     :: Vector{priv.SHTConfig} # multi-thread
    ptr      :: priv.SHTConfig
    info     :: priv.shtns_info
    nml      :: Int     # total number of (l,m) spherical harmonics components.
    nml_cplx :: Int   # number of complex coefficients to represent a complex-valued spatial field
    lmax     :: Int     # maximum degree (lmax) of spherical harmonics.
    mmax     :: Int     # maximum order (mmax*mres) of spherical harmonics.
    nlat     :: Int     # number of spatial points in Theta direction (latitude) ...
    nlat_padded :: Int  # number of spatial points in Theta direction, including padding.
    nlon     :: Int     # number of spatial points in Phi direction (longitude)
    nspat    :: Int     # number of real numbers that must be allocated in a spatial field.
    li       :: Vector{Int} # degree l for given mode index (size nlm) : li[lm]
    mi       :: Vector{Int} # order m for given mode index (size nlm) : li[lm]
    x        :: Matrix{Float64} # cos(lon)*sin(theta) array (size nlat*2nlat)
    y        :: Matrix{Float64} # sin(lon)*sin(theta) ""
    z        :: Matrix{Float64} # cos(theta)          ""
    lon      :: Matrix{Float64} # longitude           ""
    lat      :: Matrix{Float64} # latitude            ""

    laplace  :: Vector{Float64} # eigenvalues of Laplace operator
    poisson  :: Vector{Float64} # eigenvalues of Poisson operator

    function SHTnsSphere(nlat::Int, nthreads::Int=Threads.nthreads())
        lmax = div(2nlat, 3)
        ptr = priv.shtns_init(priv.sht_gauss, lmax, lmax, 1, nlat, 2nlat)
        info = unsafe_load(ptr,1)
        costheta = [unsafe_load(info.ct, i) for i in 1:nlat]
        sintheta = [unsafe_load(info.st, i) for i in 1:nlat]
        lon    = [ pi*j/nlat   for i=1:nlat, j=0:(2nlat-1) ] # longitudes
        coslat = [ sintheta[i] for i=1:nlat, j=1:2nlat]
        sinlat = [ costheta[i] for i=1:nlat, j=1:2nlat] # sin(lat)
        x   = @. cos(lon)*coslat
        y   = @. sin(lon)*coslat
        lat = @. asin(sinlat)
        li = [Int(unsafe_load(info.li, i)) for i in 1:info.nml]
        mi = [Int(unsafe_load(info.mi, i)) for i in 1:info.nml]
        lap = [Float64(-l*(l+1)) for l in li]

        poisson = map( x-> (x==0) ? 0 : inv(x), lap)

        # one replica per thread
        ptrs = [priv.shtns_create_with_grid(ptr, lmax, false) for _ in 1:nthreads]

        return new(ptrs, ptr, info,
            info.nml, info.nml_cplx,
            info.lmax, info.mmax,
            info.nlat, info.nlat_padded,
            info.nphi, info.nspat,
            li, mi, x, y, sinlat, lon, lat, lap, poisson)
    end

    function SHTnsSphere(sph::SHTnsSphere, nthreads::Int)
        @assert nthreads <= length(sph.ptrs)
        ptrs = sph.ptrs[1:nthreads]
        info = sph.info
        return new(ptrs, sph.ptr, info,
                   info.nml, info.nml_cplx,
                   info.lmax, info.mmax,
                   info.nlat, info.nlat_padded,
                   info.nphi, info.nspat,
                   sph.li, sph.mi,
                   sph.x, sph.y, sph.z, sph.lon, sph.lat,
                   sph.laplace, sph.poisson)
    end

end

Base.show(io::IO, sph::SHTnsSphere) =
    print(io, "SHTns_sphere(T$(sph.lmax), nlon=$(sph.nlon), nlat=$(sph.nlat))")

#========= allocate ========#

const SHTVectorSpat{F<:Real, N, A<:StridedArray{F,N}} = @NamedTuple{ucolat::A, ulon::A}
const SHTVectorSpec{F<:Real, N, A<:StridedArray{Complex{F},N}} = @NamedTuple{spheroidal::A, toroidal::A}

Base.show(io::IO, ::Type{SHTVectorSpat{F,N}}) where {F,N} =
    print(io, "SHTVectorSpat{$F,$N}")
Base.show(io::IO, ::Type{SHTVectorSpec{F,N}}) where {F,N} =
    print(io, "SHTVectorSpec{$F,$N}")

similar_spec(x::InOut, sph) = similar_spec(readable(x), sph)
similar_spec(::Matrix{F}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph)
similar_spec(a::Array{F, 3}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph, size(a,3))
similar_spec(a::Array{F, 4}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph, size(a,3), size(a,4))
similar_spec(::SHTVectorSpat{F,2}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph)
similar_spec(a::SHTVectorSpat{F,3}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph, size(a.ucolat, 3))
similar_spec(a::SHTVectorSpat{F,4}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph, size(a.ucolat, 3), size(a.ucolat,4))

similar_spat(x::InOut, sph) = similar_spat(readable(x), sph)
similar_spat(::Vector{Complex{F}}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph)
similar_spat(a::Matrix{Complex{F}}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph, size(a,2))
similar_spat(a::Array{Complex{F},3}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph, size(a,2), size(a,3))
similar_spat(::SHTVectorSpec{F,1}, sph) where F = shtns_alloc(F, Val(:vector_spat), sph)
similar_spat(a::SHTVectorSpec{F,2}, sph) where F = shtns_alloc(F, Val(:vector_spat), sph, size(a.spheroidal, 2))

shtns_alloc(F, ::Val{:scalar_spec}, sph, args...) = shtns_alloc_spec(F, sph, args...)
shtns_alloc(F, ::Val{:scalar_spat}, sph, args...) = shtns_alloc_spat(F, sph, args...)

shtns_alloc(F, ::Val{:vector_spat}, sph::SHTnsSphere, args...) = (
    ucolat = shtns_alloc_spat(F, sph, args...),
    ulon = shtns_alloc_spat(F, sph, args...) )

shtns_alloc(F, ::Val{:vector_spec}, sph::SHTnsSphere, args...) = (
    spheroidal = shtns_alloc_spec(F, sph, args...),
    toroidal = shtns_alloc_spec(F, sph, args...) )

shtns_alloc_spat(F, sph, dims...) = Array{F}(undef, sph.nlat, 2*sph.nlat, dims...)
shtns_alloc_spec(F, sph, dims...) = Array{Complex{F}}(undef, sph.nml, dims...)

#========= sample ========#

sample_scalar!(spat, f, sph::SHTnsSphere) = @. spat = f(sph.x, sph.y, sph.z, sph.lon, sph.lat)
sample_scalar!(::Void, f, sph::SHTnsSphere) = @. f(sph.x, sph.y, sph.z, sph.lon, sph.lat)

function sample_vector!(::Void, ulonlat, sph::SHTnsSphere)
    (; x, y, z, lon, lat) = sph
    ucolat = @. -getindex(ulonlat(x, y, z, lon, lat), 2)
    ulon = @. getindex(ulonlat(x, y, z, lon, lat), 1)
    return (; ucolat, ulon)
end

#============ higher-dimensional transforms ===========#

function analysis!(fun::Fun, sph, spec, spat) where Fun
    @assert extra_axes(spec) == extra_axes(spat) "$(extra_axes(spec)) != $(extra_axes(spat))"
    transform!(fun, sph, spec, spat)
    return spec
end

function synthesis!(fun::Fun, sph, spec, spat) where Fun
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

function extra_axes(a, b)
    @assert extra_axes(a) == extra_axes(b) "$(extra_axes(a)) != $(extra_axes(b))"
    return extra_axes(a)
end

function extra_axes(spec::Array{ComplexF64})
    @assert ndims(spec) <= 3 "$(ndims(spec))>3"
    return axes(spec,2), axes(spec, 3)
end

function extra_axes(spat::Array{Float64})
    @assert ndims(spat) <= 4 "$(ndims(spat))>4"
    return axes(spat,3), axes(spat, 4)
end

#========= scalar synthesis / analysis ========#

function transform!(fun, sph, spec::Array{ComplexF64}, spat::Array{Float64})
    batch(sph, size(spec,2), size(spec,3)) do ptr, _, k, l
        @views fun(ptr, spec[:,k,l], spat[:,:,k,l])
    end
end

synthesis_scalar!(spat::Array{Float64}, spec::Array{ComplexF64}, sph::SHTnsSphere) =
    synthesis!(priv.SH_to_spat, sph, spec, spat)

synthesis_scalar!(::Void, spec, sph) = synthesis_scalar!(similar_spat(spec, sph), spec, sph)

function synthesis_scalar!(spat, spec, ptr::Ptr)
    priv.SH_to_spat(ptr, spec, spat)
    return spat
end

analysis_scalar!(spec::Array{ComplexF64}, spat::In{<:Array{Float64}}, sph::SHTnsSphere) =
    analysis!(priv.spat_to_SH, sph, spec, writable(spat))

analysis_scalar!(::Void, spat::In{<:Array{Float64}}, sph::SHTnsSphere) = analysis_scalar!(similar_spec(spat, sph), spat, sph)

function analysis_scalar!(spec, spat::In{<:AbstractMatrix{Float64}}, ptr::Ptr)
    spat = writable(spat)
    priv.spat_to_SH(ptr, spec, spat)
    return spec
end

#========= vector synthesis / analysis ========#

vector_spat((vt, vp)) = (ucolat=vt, ulon=vp)
vector_spec((slm, tlm)) = (spheroidal=slm, toroidal=tlm)

function transform!(fun, sph, spec::SHTVectorSpec, spat::SHTVectorSpat)
    batch(sph, size(spec.toroidal,2), size(spec.toroidal,3)) do ptr, _, k, l
        @views fun(ptr, spec.spheroidal[:,k,l], spec.toroidal[:,k,l], spat.ucolat[:,:,k,l], spat.ulon[:,:,k,l])
    end
end

function analysis_vector!(spec, spat::In{<:SHTVectorSpat}, ptr::Ptr)
    spat = writable(spat)
    priv.spat_to_SHsphtor(ptr, spec.spheroidal, spec.toroidal, spat.ucolat, spat.ulon)
    return spec
end

analysis_vector!(::Void, spat, sph) = analysis_vector!(similar_spec(spat, sph), spat, sph)
analysis_vector!(spec::SHTVectorSpec, spat::InOut, sph::SHTnsSphere) = analysis_vector!(spec, writable(spat), sph)
analysis_vector!(spec::SHTVectorSpec, spat::SHTVectorSpat, sph::SHTnsSphere) =
    analysis!(priv.spat_to_SHsphtor, sph, spec, spat)

function synthesis_vector!(spat, spec, ptr::Ptr)
    priv.SHsphtor_to_spat(ptr, spec.spheroidal, spec.toroidal, spat.ucolat, spat.ulon)
    return spat
end

synthesis_vector!(::Void, spec, sph) = synthesis_vector!(similar_spat(spec, sph), spec, sph)
synthesis_vector!(spat::SHTVectorSpat, spec::InOut, sph) = synthesis_vector!(spat, spec, sph)
synthesis_vector!(spat::SHTVectorSpat, spec::SHTVectorSpec, sph::SHTnsSphere) =
    synthesis!(priv.SHsphtor_to_spat, sph, spec, spat)

# spheroidal synthesis (gradient)
function transform!(fun, sph, spec::Array{ComplexF64}, spat::SHTVectorSpat)
    batch(sph, size(spec,2), size(spec,3)) do ptr, _, k, l
        @views fun(ptr, spec[:,k,l], spat.ucolat[:,:,k,l], spat.ulon[:,:,k,l])
    end
end

synthesis_spheroidal!(::Void, spec::VC64, sph::SHTnsSphere) =
    synthesis_spheroidal!(shtns_alloc(Float64, Val(:vector_spat), sph), spec, sph)
synthesis_spheroidal!(::Void, spec::Matrix{ComplexF64}, sph::SHTnsSphere) =
    synthesis_spheroidal!(shtns_alloc(Float64, Val(:vector_spat), sph, size(spec,2)), spec, sph)

synthesis_spheroidal!(spat::SHTVectorSpat, spec::Array{ComplexF64}, sph::SHTnsSphere) =
    synthesis!(priv.SHsph_to_spat, sph, spec, spat)

function synthesis_spheroidal!(spat, spec, sph::Ptr)
    priv.SHsph_to_spat(sph, spec, spat.ucolat, spat.ulon)
    return spat
end

#========= curl, div ========#

# due to SHTns sign convention, ζ=-ΔT with T the toroidal component
# see https://www2.atmos.umd.edu/~dkleist/docs/shtns/doc/html/vsh.html
curl!(spec_out, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. spec_out = spec_in.toroidal * (-sph.laplace)

divergence!(spec_out, spec_in::NamedTuple{(:spheroidal, :toroidal)}, sph::SHTnsSphere) =
    @. spec_out = spec_in.spheroidal * sph.laplace

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
