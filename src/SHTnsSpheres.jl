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
shtns_use_threads(n::Int=0) = ccall((:shtns_use_threads, priv.libshtns), Cint, (Cint,), n)

include("julia/util.jl")

#==================================================================#

"""
  sph = SHTnsSphere(nlat, nthreadsThreads.nthreads())

Returns a SHTnsSphere corresponding to a lat-lon mesh of size nlat×2nlat
and spectral truncation 2nlat/3 (i.e. T21 and 32×64 for nlat=32)
"""
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

const SHTVectorSpat{F<:Real, N, A<:AbstractArray{F,N}} = @NamedTuple{ucolat::A, ulon::A}
const SHTVectorSpec{F<:Real, N, A<:AbstractArray{Complex{F},N}} = @NamedTuple{spheroidal::A, toroidal::A}

Base.show(io::IO, ::Type{SHTVectorSpat{F,N}}) where {F,N} =
    print(io, "SHTVectorSpat{$F,$N}")
Base.show(io::IO, ::Type{SHTVectorSpec{F,N}}) where {F,N} =
    print(io, "SHTVectorSpec{$F,$N}")

similar_spec!(y, x, sph) = y
similar_spec!(::Void, x, sph) = similar_spec(x, sph)
similar_spat!(y, x, sph) = y
similar_spat!(::Void, x, sph) = similar_spat(x, sph)

similar_spec(x::Writable, sph) = similar_spec(readable(x), sph)
similar_spec(::AbstractMatrix{F}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph)
similar_spec(a::AbstractArray{F, 3}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph, size(a,3))
similar_spec(a::AbstractArray{F, 4}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph, size(a,3), size(a,4))
similar_spec(::SHTVectorSpat{F,2}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph)
similar_spec(a::SHTVectorSpat{F,3}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph, size(a.ucolat, 3))
similar_spec(a::SHTVectorSpat{F,4}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph, size(a.ucolat, 3), size(a.ucolat,4))

similar_spat(x::Writable, sph) = similar_spat(readable(x), sph)
similar_spat(::AbstractVector{Complex{F}}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph)
similar_spat(a::AbstractMatrix{Complex{F}}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph, size(a,2))
similar_spat(a::AbstractArray{Complex{F},3}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph, size(a,2), size(a,3))
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

include("julia/transforms.jl")
include("julia/adjoints.jl")

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
