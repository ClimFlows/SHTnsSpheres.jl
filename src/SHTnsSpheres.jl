module SHTnsSpheres

module priv
using SHTns_jll
include("julia/SHTns.jl")
export AF64, AC64, MF64, VC64
end

using .priv

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

#========= allocate and sample scalar fields ========#

Base.show(io::IO, sph::SHTnsSphere) =
    print(io, "SHTns_sphere(T$(sph.lmax), nlon=$(sph.nlon), nlat=$(sph.nlat))")

shtns_alloc_spat(sph, dims...)         = Array{Float64}(undef, sph.nlat, 2*sph.nlat, dims...)
shtns_alloc_spec(sph, dims...)         = Array{ComplexF64}(undef, sph.nml, dims...)

similar_spec(spat::MF64, sph) = allocate_shtns(Val(:scalar_spec), sph,)
similar_spat(spec::VC64, sph) = allocate_shtns(Val(:scalar_spat), sph)

allocate_shtns(::Val{:scalar_spec}, sph, args...) = shtns_alloc_spec(sph, args...)
allocate_shtns(::Val{:scalar_spat}, sph, args...) = shtns_alloc_spat(sph, args...)

sample_scalar!(spat, f, sph::SHTnsSphere) = @. spat = f(sph.x, sph.y, sph.z, sph.lon, sph.lat)
sample_scalar!(::Void, f, sph::SHTnsSphere) = @. f(sph.x, sph.y, sph.z, sph.lon, sph.lat)

#========= allocate and sample vector fields ========#

const SHTVectorSpat{N} = NamedTuple{(:ucolat, :ulon),         <:Tuple{<:AF64{N}, <:AF64{N}}}
const SHTVectorSpec{M} = NamedTuple{(:spheroidal, :toroidal), <:Tuple{<:AC64{M}, <:AC64{M}}}

similar_spec(spat::SHTVectorSpat, sph) = allocate_shtns(Val(:vector_spec), sph)
similar_spat(spec::SHTVectorSpec, sph) = allocate_shtns(Val(:vector_spat), sph)

allocate_shtns(::Val{:vector_spat}, sph::SHTnsSphere, args...) = (
    ucolat = shtns_alloc_spat(sph, args...),
    ulon = shtns_alloc_spat(sph, args...) )

allocate_shtns(::Val{:vector_spec}, sph::SHTnsSphere, args...) = (
    spheroidal = shtns_alloc_spec(sph, args...),
    toroidal = shtns_alloc_spec(sph, args...) )

function sample_vector(ulonlat, sph::SHTnsSphere)
    (; x, y, z, lon, lat) = sph
    ucolat = @. -getindex(ulonlat(x, y, z, lon, lat), 2)
    ulon = @. getindex(ulonlat(x, y, z, lon, lat), 1)
    return (; ucolat, ulon)
end

#========= scalar synthesis ========#

function synthesis_scalar!(spat::MF64, spec::VC64, sph::SHTnsSphere)
    priv.SH_to_spat(sph.ptr, spec, spat)
    return spat
end
synthesis_scalar!(::Void, spec, sph) = synthesis_scalar!(similar_spat(spec, sph), spec, sph)

#========= scalar analysis ========#

function analysis_scalar!(spec::VC64, spat::MF64, sph::SHTnsSphere)
    priv.spat_to_SH(sph.ptr, spat, spec)
    return spec
end

analysis_scalar!(::Void, spat::AF64, sph::SHTnsSphere) = analysis_scalar!(similar_spec(spat, sph), spat, sph)

#========= vector analysis ========#

function analysis_vector!(vec::SHTVectorSpec, (ucolat,ulon)::SHTVectorSpat, sph::SHTnsSphere)
    priv.spat_to_SHsphtor(sph.ptr, ucolat, ulon, vec.spheroidal, vec.toroidal)
    return vec
end

analysis_vector(spat::SHTVectorSpat, sph::SHTnsSphere) = analysis_vector!(similar_spec(spat, sph), spat, sph)

#========= spheroidal synthesis (gradient) =========#

function synthesis_spheroidal!(vec::SHTVectorSpat, spec::VC64, sph::SHTnsSphere)
    priv.SHsph_to_spat(sph.ptr, spec, vec.ucolat, vec.ulon)
    return vec
end

synthesis_spheroidal(spec::VC64, sph::SHTnsSphere) =
    synthesis_spheroidal!(allocate_shtns(Val(:vector_spat), sph), spec, sph)

#========= divergence ========#

function divergence!(spec::VC64, vec::SHTVectorSpec, sph::SHTnsSphere)
    spheroidal, laplace = vec.spheroidal, sph.laplace
    @. spec = spheroidal*laplace
    return spec
end

divergence(vec::SHTVectorSpec, sph::SHTnsSphere) = vec.spheroidal .* sph.laplace

function analysis_div(spat::SHTVectorSpat, sph::SHTnsSphere)
    (; spheroidal) = analysis_vector(spat, sph)
    return spheroidal .* sph.laplace
end

#========= curl ========#

curl(vec::SHTVectorSpec, sph::SHTnsSphere) = vec.toroidal .* sph.laplace

#========== for Julia <1.9 ==========#

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
