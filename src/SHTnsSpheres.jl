module SHTnsSpheres

using MutatingOrNot: void, Void

module priv
using SHTns_jll
include("julia/SHTns.jl")
export AF64, AC64, MF64, VC64
end

using .priv
using .priv: shtns_use_threads

include("julia/util.jl")

#==================================================================#

struct SHTnsSphere
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

    function SHTnsSphere(nlat)
        lmax = div(2nlat, 3)
        ptr = priv.shtns_init(priv.sht_gauss, lmax, lmax, 1, nlat, 2nlat)
        info = unsafe_load(ptr,1)
        costheta = [unsafe_load(info.ct, i) for i in 1:nlat]
        sintheta = [unsafe_load(info.st, i) for i in 1:nlat]
        lon    = [ pi*j/nlat   for i=1:nlat, j=1:2nlat] # longitudes
        coslat = [ sintheta[i] for i=1:nlat, j=1:2nlat]
        sinlat = [ costheta[i] for i=1:nlat, j=1:2nlat] # sin(lat)
        x   = @. cos(lon)*coslat
        y   = @. sin(lon)*coslat
        lat = @. asin(sinlat)
        li = [Int(unsafe_load(info.li, i)) for i in 1:info.nml]
        mi = [Int(unsafe_load(info.mi, i)) for i in 1:info.nml]
        lap = [Float64(-l*(l+1)) for l in li]

        poisson = map( x-> (x==0) ? 0 : inv(x), lap)

        return new(ptr, info,
            info.nml, info.nml_cplx,
            info.lmax, info.mmax,
            info.nlat, info.nlat_padded,
            info.nphi, info.nspat,
            li, mi, x, y, sinlat, lon, lat, lap, poisson)
    end
end

Base.show(io::IO, sph::SHTnsSphere) =
    print(io, "SHTns_sphere(T$(sph.lmax), nlon=$(sph.nlon), nlat=$(sph.nlat))")

#========= allocate ========#

const SHTVectorSpat{F<:Real, N} = @NamedTuple{ucolat::Array{F,N}, ulon::Array{F,N}}
const SHTVectorSpec{F<:Real, N} = @NamedTuple{spheroidal::Array{Complex{F},N}, toroidal::Array{Complex{F},N}}

Base.show(io::IO, ::Type{SHTVectorSpat{F,N}}) where {F,N} =
    print(io, "SHTVectorSpat{$F,$N}")
Base.show(io::IO, ::Type{SHTVectorSpec{F,N}}) where {F,N} =
    print(io, "SHTVectorSpec{$F,$N}")

similar_spec(x::InOut, sph) = similar_spec(readable(x), sph)
similar_spec(::Matrix{F}, sph) where F = shtns_alloc(F, Val(:scalar_spec), sph)
similar_spec(::SHTVectorSpat{F}, sph) where F = shtns_alloc(F, Val(:vector_spec), sph)

similar_spat(x::InOut, sph) = similar_spat(readable(x), sph)
similar_spat(::Vector{Complex{F}}, sph) where F = shtns_alloc(F, Val(:scalar_spat), sph)
similar_spat(::SHTVectorSpec{F}, sph) where F = shtns_alloc(F, Val(:vector_spat), sph)

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

#========= scalar synthesis / analysis ========#

function synthesis_scalar!(spat::MF64, spec::VC64, sph::SHTnsSphere)
    priv.SH_to_spat(sph.ptr, spec, spat)
    return spat
end
synthesis_scalar!(::Void, spec, sph) = synthesis_scalar!(similar_spat(spec, sph), spec, sph)

function analysis_scalar!(spec::VC64, spat::MF64, sph::SHTnsSphere)
    priv.spat_to_SH(sph.ptr, spat, spec)
    return spec
end

analysis_scalar!(::Void, spat::AF64, sph::SHTnsSphere) = analysis_scalar!(similar_spec(spat, sph), spat, sph)

#========= vector synthesis / analysis ========#

analysis_vector!(::Void, spat, sph) = analysis_vector!(similar_spec(spat, sph), spat, sph)
analysis_vector!(spec::SHTVectorSpec, spat::InOut, sph) = analysis_vector!(spec, readable(spat), sph)
function analysis_vector!(spec::SHTVectorSpec{Float64,1}, spat::SHTVectorSpat{Float64,2}, sph::SHTnsSphere)
    priv.spat_to_SHsphtor(sph.ptr, spat.ucolat, spat.ulon, spec.spheroidal, spec.toroidal)
    return spec
end

synthesis_vector!(::Void, spec, sph) = synthesis_vector!(similar_spat(spec, sph), spec, sph)
synthesis_vector!(spat::SHTVectorSpat, spec::InOut, sph) = synthesis_vector!(spat, writable(spec), sph)
function synthesis_vector!(spat::SHTVectorSpat, spec::SHTVectorSpec, sph::SHTnsSphere)
    priv.SHsphtor_to_spat(sph.ptr, spec.spheroidal, spec.toroidal, spat.ucolat, spat.ulon)
    return spat
end

# spheroidal synthesis (gradient)
function synthesis_spheroidal!(spat::SHTVectorSpat, spec::VC64, sph::SHTnsSphere)
    priv.SHsph_to_spat(sph.ptr, spec, spat.ucolat, spat.ulon)
    return spat
end

synthesis_spheroidal!(::Void, spec::VC64, sph::SHTnsSphere) =
    synthesis_spheroidal!(shtns_alloc(Float64, Val(:vector_spat), sph), spec, sph)

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
