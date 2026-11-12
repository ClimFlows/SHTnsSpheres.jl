module CUDA_Ext

using CUDA
using SHTnsSpheres: SHTnsSphere, SHTVectorSpec, SHTVectorSpat, priv
using SHTnsSpheres.priv: SHTConfig, libshtns
import SHTnsSpheres: create_backend, backend_array, transform!

struct CUDAState{T}
    plans::Dict{Int,SHTConfig}
    context::CUDA.CuContext
    stream::CUDA.CuStream
    lock::ReentrantLock
end

const SHT_FP32 = Cint(256 * 1024)

destroy(ptr) = ccall((:shtns_destroy, libshtns), Cvoid, (SHTConfig,), ptr)

function checked_batch(batch)
    batch isa Integer && 1 <= batch <= typemax(Cint) ||
        throw(ArgumentError("batch size must be a positive integer that fits Cint"))
    return Int(batch)
end

function create_backend(::Val{:cuda}, ::Type{T}, nlat, lmax, nthreads, batch) where {T<:Union{Float32,Float64}}
    batches = batch isa Union{Tuple,AbstractVector} ? batch : (batch,)
    isempty(batches) && throw(ArgumentError("provide at least one initial batch size"))
    sizes = unique(map(checked_batch, batches))
    state = CUDAState{T}(Dict{Int,SHTConfig}(), CUDA.context(), CUDA.stream(), ReentrantLock())
    try
        for size in sizes
            state.plans[size] = create_plan(state, nlat, lmax, size)
        end
        return state, state.plans[first(sizes)], SHTConfig[]
    catch
        close(state)
        rethrow()
    end
end

function create_plan(state::CUDAState{T}, nlat, lmax, batch) where T
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
            (SHTConfig, CUDA.CUstream, CUDA.CUstream), ptr, state.stream, C_NULL)
        device = ccall((:cushtns_init_gpu, libshtns), Cint, (SHTConfig,), ptr)
        device >= 0 || error("SHTns could not initialize CUDA for this grid")
        return ptr
    catch
        destroy(ptr)
        rethrow()
    end
end

backend_array(::CUDAState{T}, a) where T = CuArray{T}(a)

Base.close(sph::SHTnsSphere{<:CUDAState}) = close(sph.backend)

function Base.close(state::CUDAState)
    lock(state.lock) do
        isempty(state.plans) && return
        CUDA.context!(state.context) do
            foreach(destroy, values(state.plans))
            empty!(state.plans)
        end
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
    tail = size(first(spectra))[2:end]
    batch = checked_batch(prod(tail))
    for (arrays, type, shape) in ((spectra, Complex{T}, (sph.nml, tail...)),
                                 (fields, T, (sph.nlat, sph.nlon, tail...)))
        all(a -> a isa CuArray{type}, arrays) || throw(ArgumentError("expected CuArray{$type}"))
        all(a -> size(a) == shape, arrays) || throw(DimensionMismatch("expected shape $shape"))
        all(a -> CUDA.context(a) == state.context, arrays) ||
            throw(ArgumentError("arrays must use the sphere's CUDA context"))
    end
    CUDA.context() == state.context && CUDA.stream() == state.stream ||
        throw(ArgumentError("use the sphere on the CUDA context and stream where it was created"))
    lock(state.lock) do
        isempty(state.plans) && throw(ArgumentError("CUDA sphere is closed"))
        ptr = get!(state.plans, batch) do
            create_plan(state, sph.nlat, sph.lmax, batch)
        end
        GC.@preserve spectra fields begin
            cuda_call!(fun, T, ptr, spectra..., fields..., sph.lmax)
            CUDA.synchronize(state.stream)
        end
    end
    return nothing
end

function cuda_call!(::typeof(priv.SHsph_to_spat), ::Type{Float64}, ptr, s, vt, vp, lmax)
    ccall((:cu_SHsph_to_spat, libshtns), Cvoid,
        (SHTConfig, CuPtr{ComplexF64}, CuPtr{Float64}, CuPtr{Float64}, Cint),
        ptr, s, vt, vp, lmax)
end

for (T, suffix) in ((Float64, ""), (Float32, "_float"))
    P = CuPtr{T}
    C = CuPtr{Complex{T}}
    analysis = Symbol("cu_spat_to_SH", suffix)
    synthesis = Symbol("cu_SH_to_spat", suffix)
    vector_analysis = Symbol("cu_spat_to_SHsphtor", suffix)
    vector_synthesis = Symbol("cu_SHsphtor_to_spat", suffix)
    @eval begin
        function cuda_call!(::typeof(priv.spat_to_SH), ::Type{$T}, ptr, q, v, lmax)
            ccall(($(QuoteNode(analysis)), libshtns), Cvoid,
                (SHTConfig, $P, $C, Cint), ptr, v, q, lmax)
        end
        function cuda_call!(::typeof(priv.SH_to_spat), ::Type{$T}, ptr, q, v, lmax)
            ccall(($(QuoteNode(synthesis)), libshtns), Cvoid,
                (SHTConfig, $C, $P, Cint), ptr, q, v, lmax)
        end
        function cuda_call!(::typeof(priv.spat_to_SHsphtor), ::Type{$T}, ptr, s, t, vt, vp, lmax)
            ccall(($(QuoteNode(vector_analysis)), libshtns), Cvoid,
                (SHTConfig, $P, $P, $C, $C, Cint), ptr, vt, vp, s, t, lmax)
        end
        function cuda_call!(::typeof(priv.SHsphtor_to_spat), ::Type{$T}, ptr, s, t, vt, vp, lmax)
            ccall(($(QuoteNode(vector_synthesis)), libshtns), Cvoid,
                (SHTConfig, $C, $C, $P, $P, Cint), ptr, s, t, vt, vp, lmax)
        end
    end
end

end
