module shtns_sphere

using SHTns_jll
using GFDomains: GFDomains, @offload

# GFDomains functions extended here
import GFDomains: allocate_field

# functions declared in GFDomains and implemented here
import GFDomains.SHTns: allocate_shtns
import GFDomains: sample_scalar, sample_vector, truncate_scalar!,
    analysis_scalar, analysis_scalar!,
    analysis_vector, analysis_vector!,
    analysis_div,
    synthesis_scalar, synthesis_scalar!,
    synthesis_vector, synthesis_vector!,
    synthesis_spheroidal, synthesis_spheroidal!

#===================== Low-level interface to SHTns ====================#

@enum shtns_type begin
    sht_gauss=0
    sht_auto
    sht_reg_fast
    sht_reg_dct
    sht_quick_init
    sht_reg_poles
    sht_gauss_fly
end

struct shtns_info
    nml    :: Cuint     # total number of (l,m) spherical harmonics components.
    lmax   :: Cushort   # maximum degree (lmax) of spherical harmonics.
    mmax   :: Cushort   # maximum order (mmax*mres) of spherical harmonics.
    mres   :: Cushort   # the periodicity along the phi axis.
    nlat_2 :: Cushort   # half of spatial points in Theta direction (using (shtns.nlat+1)/2 allows odd shtns.nlat.)
    nlat   :: Cuint     # number of spatial points in Theta direction (latitude) ...
    nphi   :: Cuint     # number of spatial points in Phi direction (longitude)
    nspat  :: Cuint     # number of real numbers that must be allocated in a spatial field.
    li     :: Ptr{Cushort} # degree l for given mode index (size nlm) : li[lm]
    mi     :: Ptr{Cushort} # order m for given mode index (size nlm) : li[lm]
    ct     :: Ptr{Float64} # cos(theta) array (size nlat)
    st     :: Ptr{Float64} # sin(theta) array (size nlat)
    nlat_padded :: Cuint   # number of spatial points in Theta direction, including padding.
    nml_cplx    :: Cuint   # number of complex coefficients to represent a complex-valued spatial field.
end

const ArrayOrSub{N,F} = StridedArray{F,N}
const AF64{N} = ArrayOrSub{N,Float64}
const AC64{N} = ArrayOrSub{N,ComplexF64}

const MF64 = AF64{2}
const VC64 = AC64{1}
const PF64 = Ptr{Float64}
const PC64 = Ptr{ComplexF64}
const SHTConfig = Ptr{shtns_info}

function shtns_init(flags::shtns_type, lmax::Int, mmax::Int, mres::Int, nlat::Int, nphi::Int)
    output_ptr = ccall(
        (:shtns_init, :libshtns),              # name of C function and library
        Ptr{shtns_info},                       # output type
        (Cint, Cint, Cint, Cint, Cint, Cint),  # tuple of input types
        flags, lmax, mmax, mres, nlat, nphi)   # arguments

    if output_ptr == C_NULL # Could not allocate memory
        throw(OutOfMemoryError())
    end
    return output_ptr
end

function shtns_create(lmax::Int, mmax::Int, mres::Int=1, norm::Int=0)
    output_ptr = ccall(
        (:shtns_create, :libshtns),    # name of C function and library
        Ptr{shtns_info},               # output type
        (Cint, Cint, Cint, Cint),      # tuple of input types
        lmax, mmax, mres, norm)        # arguments

    if output_ptr == C_NULL # Could not allocate memory
        throw(OutOfMemoryError())
    end
    return output_ptr
end

function SH_to_spat(ptr::SHTConfig, qlm::VC64, vr::MF64)
    ccall(
        (:SH_to_spat, :libshtns),  # name of C function and library
        Cvoid,                     # output type
        (SHTConfig, PC64, PF64),   # tuple of input types
        ptr, qlm, vr)              # arguments
end

function spat_to_SH(ptr::SHTConfig, vr::MF64, qlm::VC64)
    # check that arrays are contiguous
    @assert stride(vr,1)==1
    @assert stride(vr,2)==size(vr,1)
    @assert stride(qlm,1)==1
    ccall(
        (:spat_to_SH, :libshtns),  # name of C function and library
        Cvoid,                     # output type
        (SHTConfig, PF64, PC64),   # tuple of input types
        ptr, vr, qlm)              # arguments
end

function SHsph_to_spat(ptr::SHTConfig, slm::VC64, vt::MF64, vp::MF64)
    # check that arrays are contiguous
    @assert stride(vt,1)==1
    @assert stride(vt,2)==size(vt,1)
    @assert stride(slm,1)==1
    @assert stride(vp,1)==1
    @assert stride(vp,2)==size(vp,1)
    ccall(
        (:SHsph_to_spat, :libshtns),   # name of C function and library
        Cvoid,                         # output type
        (SHTConfig, PC64, PF64, PF64), # tuple of input types
        ptr, slm, vt, vp)              # arguments
end

function SHsphtor_to_spat(ptr::SHTConfig, slm::VC64, tlm::VC64, vt::MF64, vp::MF64)
    # check that arrays are contiguous
    @assert stride(vt,1)==1
    @assert stride(vt,2)==size(vt,1)
    @assert stride(slm,1)==1
    @assert stride(vp,1)==1
    @assert stride(vp,2)==size(vp,1)
    @assert stride(tlm,1)==1
    ccall(
        (:SHsphtor_to_spat, :libshtns),       # name of C function and library
        Cvoid,                                # output type
        (SHTConfig, PC64, PC64, PF64, PF64),  # tuple of input types
        ptr, slm, tlm, vt, vp)                # arguments
end

function spat_to_SHsphtor(ptr::SHTConfig, vt::MF64, vp::MF64, slm::VC64, tlm::VC64)
    ccall(
        (:spat_to_SHsphtor, :libshtns),       # name of C function and library
        Cvoid,                                # output type
        (SHTConfig, PF64, PF64, PC64, PC64),  # tuple of input types
        ptr, vt, vp, slm, tlm)                # arguments
end

#==================================================================#

struct SHTns_sphere <: GFDomains.SHTns_sphere
    ptr      :: Ptr{shtns_info}
    info     :: shtns_info
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

    function SHTns_sphere(nlat)
        lmax = div(2nlat, 3)
        ptr = shtns_init(sht_gauss, lmax, lmax, 1, nlat, 2nlat)
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
GFDomains.SHTns_sphere(nlat) = SHTns_sphere(nlat)

Base.show(io::IO, sph::SHTns_sphere) =
    print(io, "SHTns_sphere(T$(sph.lmax), nlon=$(sph.nlon), nlat=$(sph.nlat))")

shtns_alloc_spat(sph, dims...)         = Array{Float64}(undef, sph.nlat, 2*sph.nlat, dims...)
shtns_alloc_spec(sph, dims...)         = Array{ComplexF64}(undef, sph.nml, dims...)

#========= allocate (generic) ========#

allocate_field(::Val, ::SHTns_sphere, ::Type{F}) where F = only_F64(:allocate_field, F)
allocate_shell(::Val, ::SHTns_sphere, nz, ::Type{F}) where F = only_F64(:allocate_shell, F)
allocate_shell(::Val, ::SHTns_sphere, nz, nq, ::Type{F}) where F = only_F64(:allocate_shell, F)

allocate_field(val::Val, sph::SHTns_sphere, ::Type{Float64}) = allocate_shtns(val, sph)
allocate_shell(val::Val, sph::SHTns_sphere, nz, ::Type{Float64}) = allocate_shtns(val, sph, nz)
allocate_shell(val::Val, sph::SHTns_sphere, nz, nq, ::Type{Float64}) = allocate_shtns(val, sph, nz, nq)

function only_F64(fun, ::Type{F}) where F
    @error "SHTns supports only Float64"
    throw(TypeError(fun, "SHTnsSphere", Type{Float64}, F))
end

extra_dims(spat::AF64) = size(spat)[3:end]
extra_dims(spec::AC64) = size(spec)[2:end]

#========= allocate and sample scalar fields ========#

const SHTScalar{M,N} = NamedTuple{(:spat,:spec), <:Tuple{<:AF64{N}, <:AC64{M}}}

Base.similar((spat,spec)::SHTScalar) = (; spat = similar(spat), spec = similar(spec))

similar_spec(spat::AF64, sph) = allocate_shtns(Val(:scalar_spec), sph, extra_dims(spat)...)
similar_spat(spec::AC64, sph) = allocate_shtns(Val(:scalar_spat), sph, extra_dims(spec)...)

allocate_shtns(::Val{:scalar}, sph, args...) = (; spat=shtns_alloc_spat(sph, args...), spec=shtns_alloc_spec(sph, args...) )

allocate_shtns(::Val{:scalar_spec}, sph, args...) = shtns_alloc_spec(sph, args...)
allocate_shtns(::Val{:scalar_spat}, sph, args...) = shtns_alloc_spat(sph, args...)

sample_scalar(f, sph::SHTns_sphere) = @. f(sph.x, sph.y, sph.z)

omit_halo((spat,spec)::SHTScalar, sph::SHTns_sphere) = spat
omit_halo(spat::MF64, sph::SHTns_sphere) = spat

#========= allocate and sample vector fields ========#

const SHTVector{M,N}   = NamedTuple{(:ucolat, :ulon, :spheroidal, :toroidal), <:Tuple{<:AF64{N}, <:AF64{N}, <:AC64{M}, <:AC64{M}}} # N==M+1
const SHTVectorSpat{N} = NamedTuple{(:ucolat, :ulon),                         <:Tuple{<:AF64{N}, <:AF64{N}}}
const SHTVectorSpec{M} = NamedTuple{(:spheroidal, :toroidal),                 <:Tuple{<:AC64{M}, <:AC64{M}}}

similar_spec(spat::SHTVectorSpat, sph) = allocate_shtns(Val(:vector_spec), sph, extra_dims(spat.ulon)...)
similar_spat(spec::SHTVectorSpec, sph) = allocate_shtns(Val(:vector_spat), sph, extra_dims(spec.spheroidal)...)

allocate_shtns(::Val{:vector}, sph::SHTns_sphere, args...) = (
    ucolat = shtns_alloc_spat(sph, args...),
    ulon = shtns_alloc_spat(sph, args...),
    spheroidal = shtns_alloc_spec(sph, args...),
    toroidal = shtns_alloc_spec(sph, args...))

allocate_shtns(::Val{:vector_spat}, sph::SHTns_sphere, args...) = (
    ucolat = shtns_alloc_spat(sph, args...),
    ulon = shtns_alloc_spat(sph, args...) )

allocate_shtns(::Val{:vector_spec}, sph::SHTns_sphere, args...) = (
    spheroidal = shtns_alloc_spec(sph, args...),
    toroidal = shtns_alloc_spec(sph, args...) )

function sample_vector(ulonlat, sph::SHTns_sphere)
    uv = @. ulonlat(sph.x, sph.y, sph.z)
    return (
        ucolat = [-v for (u,v) in uv],
        ulon = [u for (u,v) in uv] )
end

#========= scalar spectral transforms ========#

function analysis_scalar!(f::SHTScalar, sph::SHTns_sphere, backend)
    analysis_scalar!(f.spec, f.spat, sph, backend)
    return f
end

function analysis_scalar!(spec::VC64, spat::MF64, sph::SHTns_sphere, backend)
    # Quirk to ensure that spat_to_SH is called only once
    # when `backend` is multithread.
    @offload let range=backend(1:1 ; ptr=sph.ptr, spec, spat)
        spat_to_SH(sph.ptr, spat, spec)
    end
    return spec
end

function analysis_scalar!(spec::AC64{3}, spat::AF64{4}, sph::SHTns_sphere, backend)
    for iq in axes(spat, 4)
        @offload let range=backend(axes(spat,3) ; iq, sph, spat, spec)
            for k in range
                spat_to_SH(sph.ptr, (@view spat[:,:,k,iq]), (@view spec[:,k,iq]))
            end
        end
    end
    return spec
end

analysis_scalar(spat::AF64, sph::SHTns_sphere, backend) = analysis_scalar!(similar_spec(spat, sph), spat, sph, backend)

truncate_scalar!((spat, spec)::SHTScalar, sph::SHTns_sphere) = (; spat, spec)
truncate_scalar!(spec:: AC64, sph::SHTns_sphere) = spec

#synthesis_scalar((spat, spec)::SHTScalar, sph::SHTns_sphere, backend) = (
#    synthesis_scalar!((spat=similar(spat), spec=copy(spec)), sph, backend ))

synthesis_scalar(spec::AC64, sph::SHTns_sphere, backend) = (
    synthesis_scalar!(similar_spat(spec, sph), spec, sph, backend) )

#synthesis_scalar(spec::AC64, sph::SHTns_sphere, backend) = (
#    synthesis_scalar!(shtns_alloc_spat(sph, extra_dims(spec)...), spec, sph, backend ))

function synthesis_scalar!((spat, spec)::SHTScalar, sph::SHTns_sphere, backend)
    synthesis_scalar!(spat, spec, sph, backend)
    return (; spat, spec)
end

function synthesis_scalar!((outspat, outspec)::SHTScalar, (inspat, inspec)::SHTScalar, sph::SHTns_sphere, backend)
    synthesis_scalar!(outspat, inspec, sph, backend)
    return (; outspat, outspec)
end

function synthesis_scalar!(spat::MF64, spec::VC64, sph::SHTns_sphere, backend)
    # Quirk to ensure that SH_to_spat is called only once
    # when `backend` is multithread.
    @offload let range=backend(1:1 ; ptr=sph.ptr, spec, spat)
        for i in range
            SH_to_spat(ptr, spec, spat)
        end
    end
    return spat
end

function synthesis_scalar!(spat::AF64{3}, spec::AC64{2}, sph::SHTns_sphere, backend)
    @offload let range=backend(axes(spat,3) ; sph, spat, spec)
        for k in range
            @views SH_to_spat(sph.ptr, spec[:,k], spat[:,:,k])
        end
    end
    return spat
end

function synthesis_scalar!(spat::AF64{4}, spec::AC64{3}, sph::SHTns_sphere, backend)
    @offload let range=backend(axes(spat,3) ; sph, spat, spec)
        for iq in axes(spat, 4)
            for k in range
                @views SH_to_spat(sph.ptr, spec[:,k,iq], spat[:,:,k,iq])
            end
        end
    end
    return spat
end

#========= vector spectral transforms ========#

function analysis_vector!(vec::SHTVector, sph::SHTns_sphere)
    spat_to_SHsphtor(sph.ptr, vec.ucolat, vec.ulon, vec.spheroidal, vec.toroidal)
    return vec
end

function analysis_vector!( vec::SHTVectorSpec, (ucolat,ulon)::SHTVectorSpat, sph::SHTns_sphere, backend)
    spat_to_SHsphtor(sph.ptr, ucolat, ulon, vec.spheroidal, vec.toroidal)
    return vec
end

function analysis_vector!( spec::SHTVectorSpec{2}, spat::SHTVectorSpat{3}, sph::SHTns_sphere, backend)
    @offload let range=backend(axes(spec.spheroidal,2) ; sph, spec, spat)
        for k in range
            @views spat_to_SHsphtor(sph.ptr, spat.ucolat[:,:,k], spat.ulon[:,:,k], spec.spheroidal[:,k], spec.toroidal[:,k])
        end
    end
end

analysis_vector(spat::SHTVectorSpat, sph::SHTns_sphere, backend) = analysis_vector!(similar_spec(spat, sph), spat, sph, backend)

function synthesis_spheroidal!(vec::SHTVectorSpat, spec::VC64, sph::SHTns_sphere, backend)
    SHsph_to_spat(sph.ptr, spec, vec.ucolat, vec.ulon)
    return vec
end

synthesis_spheroidal(spec::VC64, sph::SHTns_sphere, backend) =
    synthesis_spheroidal!(allocate_shtns(Val(:vector_spat), sph), spec, sph, backend)

#========= divergence ========#

function divergence!(div::SHTScalar, vec::SHTVector, sph::SHTns_sphere, backend=nothing)
    spec, spheroidal, laplace = div.spec, vec.spheroidal, sph.laplace
    @. spec = spheroidal*laplace
    return div
end

function divergence!(spec::VC64, vec::SHTVectorSpec, sph::SHTns_sphere, backend=nothing)
    spheroidal, laplace = vec.spheroidal, sph.laplace
    @. spec = spheroidal*laplace
    return spec
end

function divergence(vec::SHTVectorSpec, sph::SHTns_sphere, backend=nothing)
    spheroidal, laplace = vec.spheroidal, sph.laplace
    return spheroidal.*laplace
end

function analysis_div(spat::SHTVectorSpat, sph::SHTns_sphere, backend=nothing)
    (; spheroidal) = analysis_vector(spat, sph, backend)
    return spheroidal .* sph.laplace
end

end
