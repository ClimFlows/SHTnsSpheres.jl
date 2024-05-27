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

# A=Array, M=Matrix, V=Vector
# const ArrayOrSub{N,F} = StridedArray{F,N}
const AF64{N} = StridedArray{Float64,N}
const AC64{N} = StridedArray{ComplexF64,N}

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

function spat_to_SH(ptr::SHTConfig, qlm::VC64, vr::MF64)
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

function spat_to_SHsphtor(ptr::SHTConfig, slm::VC64, tlm::VC64, vt::MF64, vp::MF64)
    ccall(
        (:spat_to_SHsphtor, :libshtns),       # name of C function and library
        Cvoid,                                # output type
        (SHTConfig, PF64, PF64, PC64, PC64),  # tuple of input types
        ptr, vt, vp, slm, tlm)                # arguments
end
