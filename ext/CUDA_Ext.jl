module CUDA_Ext

using CUDA
using SHTnsSpheres: SHTnsSphere, SHTVectorSpec, SHTVectorSpat, priv
using SHTnsSpheres.priv: SHTConfig, libshtns
import SHTnsSpheres: create_backend, backend_array, transform!

mutable struct CUDAState{T}
    ptr::SHTConfig
    context::CUDA.CuContext
    stream::CUDA.CuStream
    batch::Int
end

const SHT_FP32 = Cint(256 * 1024)

destroy(ptr) = ccall((:shtns_destroy, libshtns), Cvoid, (SHTConfig,), ptr)

function create_backend(::Val{:cuda}, ::Type{T}, nlat, lmax, nthreads, batch) where {T<:Union{Float32,Float64}}
    1 <= batch <= typemax(Cint) || throw(ArgumentError("batch size must be positive and fit Cint"))
    ctx, stream = CUDA.context(), CUDA.stream()
    flags = Cint(priv.sht_gauss) | (T === Float32 ? SHT_FP32 : Cint(0))
    ptr = priv.shtns_create(lmax, lmax)
    try
        count = ccall((:shtns_set_many, libshtns), Cint,
            (SHTConfig, Cint, Clong), ptr, batch, 0)
        count == batch || error("SHTns rejected the requested batch size")
        ccall((:shtns_set_grid, libshtns), Cint,
            (SHTConfig, Cint, Cdouble, Cint, Cint), ptr, flags, 1e-10, nlat, 2nlat)
        info = unsafe_load(ptr)
        info.nlat == info.nlat_padded == nlat && info.nphi == 2nlat &&
            info.nspat == nlat * 2nlat * batch ||
            throw(ArgumentError("SHTns CUDA requires an unpadded latitude-longitude grid"))
        ccall((:cushtns_set_streams, libshtns), Cvoid,
            (SHTConfig, CUDA.CUstream, CUDA.CUstream), ptr, stream, C_NULL)
        device = ccall((:cushtns_init_gpu, libshtns), Cint, (SHTConfig,), ptr)
        device >= 0 || error("SHTns could not initialize CUDA for this grid")
        return CUDAState{T}(ptr, ctx, stream, batch), ptr, SHTConfig[]
    catch
        destroy(ptr)
        rethrow()
    end
end

backend_array(::CUDAState{T}, a) where T = CuArray{T}(a)

Base.close(sph::SHTnsSphere{<:CUDAState}) = close(sph.backend)

function Base.close(state::CUDAState)
    state.ptr == C_NULL && return nothing
    CUDA.context!(state.context) do
        destroy(state.ptr)
        state.ptr = C_NULL
    end
    return nothing
end

transform!(fun, sph::SHTnsSphere{<:CUDAState}, spec::AbstractArray, spat::AbstractArray, op) =
    cuda_transform!(fun, sph, (spec,), (spat,))

transform!(fun, sph::SHTnsSphere{<:CUDAState}, spec::SHTVectorSpec, spat::SHTVectorSpat, op) =
    cuda_transform!(fun, sph, values(spec), values(spat))

transform!(fun::typeof(priv.SHsph_to_spat), sph::SHTnsSphere{CUDAState{Float64}},
           spec::AbstractArray, spat::SHTVectorSpat, op) =
    cuda_transform!(fun, sph, (spec,), values(spat))

function transform!(::typeof(priv.SHsph_to_spat), sph::SHTnsSphere{CUDAState{Float32}},
                    spec::AbstractArray, spat::SHTVectorSpat, op)
    # SHTns 3.7 lacks the dedicated Float32 spheroidal symbol.
    toroidal = fill!(similar(spec), 0)
    return cuda_transform!(priv.SHsphtor_to_spat, sph, (spec, toroidal), values(spat))
end

function cuda_transform!(fun, sph::SHTnsSphere{CUDAState{T}}, spectra, fields) where T
    state = sph.backend
    state.ptr != C_NULL || throw(ArgumentError("CUDA sphere is closed"))
    tail = size(first(spectra))[2:end]
    prod(tail) == state.batch || throw(DimensionMismatch("array batch size differs from CUDA sphere"))
    for (arrays, type, shape) in ((spectra, Complex{T}, (sph.nml, tail...)),
                                 (fields, T, (sph.nlat, sph.nlon, tail...)))
        all(a -> a isa CuArray{type}, arrays) || throw(ArgumentError("expected CuArray{$type}"))
        all(a -> size(a) == shape, arrays) || throw(DimensionMismatch("expected shape $shape"))
        all(a -> CUDA.context(a) == state.context, arrays) ||
            throw(ArgumentError("arrays must use the sphere's CUDA context"))
    end
    CUDA.context() == state.context && CUDA.stream() == state.stream ||
        throw(ArgumentError("use the sphere on the CUDA context and stream where it was created"))
    GC.@preserve spectra fields begin
        cuda_call!(fun, state, spectra..., fields..., sph.lmax)
        CUDA.synchronize(state.stream)
    end
    return nothing
end

function cuda_call!(::typeof(priv.SHsph_to_spat), state::CUDAState{Float64}, s, vt, vp, lmax)
    ccall((:cu_SHsph_to_spat, libshtns), Cvoid,
        (SHTConfig, CuPtr{ComplexF64}, CuPtr{Float64}, CuPtr{Float64}, Cint),
        state.ptr, s, vt, vp, lmax)
end

for (T, suffix) in ((Float64, ""), (Float32, "_float"))
    P = CuPtr{T}
    C = CuPtr{Complex{T}}
    analysis = Symbol("cu_spat_to_SH", suffix)
    synthesis = Symbol("cu_SH_to_spat", suffix)
    vector_analysis = Symbol("cu_spat_to_SHsphtor", suffix)
    vector_synthesis = Symbol("cu_SHsphtor_to_spat", suffix)
    @eval begin
        function cuda_call!(::typeof(priv.spat_to_SH), state::CUDAState{$T}, q, v, lmax)
            ccall(($(QuoteNode(analysis)), libshtns), Cvoid,
                (SHTConfig, $P, $C, Cint), state.ptr, v, q, lmax)
        end
        function cuda_call!(::typeof(priv.SH_to_spat), state::CUDAState{$T}, q, v, lmax)
            ccall(($(QuoteNode(synthesis)), libshtns), Cvoid,
                (SHTConfig, $C, $P, Cint), state.ptr, q, v, lmax)
        end
        function cuda_call!(::typeof(priv.spat_to_SHsphtor), state::CUDAState{$T}, s, t, vt, vp, lmax)
            ccall(($(QuoteNode(vector_analysis)), libshtns), Cvoid,
                (SHTConfig, $P, $P, $C, $C, Cint), state.ptr, vt, vp, s, t, lmax)
        end
        function cuda_call!(::typeof(priv.SHsphtor_to_spat), state::CUDAState{$T}, s, t, vt, vp, lmax)
            ccall(($(QuoteNode(vector_synthesis)), libshtns), Cvoid,
                (SHTConfig, $C, $C, $P, $P, Cint), state.ptr, s, t, vt, vp, lmax)
        end
    end
end

end
