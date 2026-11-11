module CUDA_Ext

using CUDA
using Libdl: dlopen, dlclose, dlsym, RTLD_LOCAL, RTLD_NOW, RTLD_DEEPBIND
using SHTnsSpheres: SHTnsSphere, Void, readable, writable
import SHTnsSpheres: cuda_plan, analysis_scalar!, synthesis_scalar!,
    analysis_vector!, synthesis_vector!, synthesis_spheroidal!

using CUDA: CuPtr, CUstream
using SHTnsSpheres.priv: SHTConfig, sht_gauss

const TransformSymbols = NamedTuple{
    (:destroy, :analyse, :synthesise, :analyse_vector, :synthesise_vector),
    NTuple{5,Ptr{Cvoid}},
}

mutable struct CUDAPlan{T}
    library::Ptr{Cvoid}
    symbols::TransformSymbols
    ptr::SHTConfig
    context::CUDA.CuContext
    stream::CUDA.CuStream
    guard::ReentrantLock
    nml::Int
    nlat::Int
    nlon::Int
    lmax::Int
    batch::Int
end

const SHT_FP32 = Cint(256 * 1024)

function cuda_plan(sphere::SHTnsSphere, ::Type{T}=Float64; library::AbstractString, batch::Int=1) where {T<:Union{Float32,Float64}}
    1 <= batch <= typemax(Cint) || throw(ArgumentError("batch size must be positive and fit Cint"))
    ctx = CUDA.context()
    stream = CUDA.stream()
    flags = Cint(sht_gauss) | (T === Float32 ? SHT_FP32 : Cint(0))
    lib = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND)
    ptr = SHTConfig(C_NULL)
    try
        suffix = T === Float32 ? "_float" : ""
        symbols = (
            destroy=dlsym(lib, :shtns_destroy),
            analyse=dlsym(lib, "cu_spat_to_SH" * suffix),
            synthesise=dlsym(lib, "cu_SH_to_spat" * suffix),
            analyse_vector=dlsym(lib, "cu_spat_to_SHsphtor" * suffix),
            synthesise_vector=dlsym(lib, "cu_SHsphtor_to_spat" * suffix),
        )
        create = dlsym(lib, :shtns_create)
        set_many = dlsym(lib, :shtns_set_many)
        set_grid = dlsym(lib, :shtns_set_grid)
        set_streams = dlsym(lib, :cushtns_set_streams)
        init_gpu = dlsym(lib, :cushtns_init_gpu)
        ptr = ccall(create, SHTConfig, (Cint, Cint, Cint, Cint), sphere.lmax, sphere.mmax, 1, 0)
        ptr == C_NULL && throw(OutOfMemoryError())
        count = ccall(set_many, Cint, (SHTConfig, Cint, Clong), ptr, batch, sphere.nml)
        count == batch || error("SHTns rejected the requested batch size")
        ccall(set_grid, Cint, (SHTConfig, Cint, Cdouble, Cint, Cint),
            ptr, flags, 1e-10, sphere.nlat, sphere.nlon)
        info = unsafe_load(ptr)
        info.nml == sphere.nml && info.nlat == sphere.nlat && info.nphi == sphere.nlon &&
            info.nlat_padded == sphere.nlat && info.nspat == sphere.nlat * sphere.nlon * batch ||
            throw(ArgumentError("SHTns CUDA plan requires an unpadded latitude-longitude grid"))
        ccall(set_streams, Cvoid, (SHTConfig, CUstream, CUstream), ptr, stream, C_NULL)
        device = ccall(init_gpu, Cint, (SHTConfig,), ptr)
        device >= 0 || error("SHTns could not initialize CUDA for this grid")
        return CUDAPlan{T}(lib, symbols, ptr, ctx, stream, ReentrantLock(),
            sphere.nml, sphere.nlat, sphere.nlon, sphere.lmax, batch)
    catch
        if ptr != C_NULL
            destroy = dlsym(lib, :shtns_destroy)
            ccall(destroy, Cvoid, (SHTConfig,), ptr)
        end
        dlclose(lib)
        rethrow()
    end
end

function cuda_plan(f::Function, sphere::SHTnsSphere, args...; kwargs...)
    plan = cuda_plan(sphere, args...; kwargs...)
    try
        return f(plan)
    finally
        close(plan)
    end
end

function Base.close(plan::CUDAPlan)
    lock(plan.guard) do
        if plan.ptr != C_NULL
            CUDA.context!(plan.context) do
                ccall(plan.symbols.destroy, Cvoid, (SHTConfig,), plan.ptr)
                plan.ptr = C_NULL
                dlclose(plan.library)
            end
        end
    end
    return nothing
end

function transform!(run, symbol, plan, spectra, fields)
    tail = size(first(spectra))[2:end]
    prod(tail) == plan.batch || throw(DimensionMismatch("array batch size differs from CUDA plan"))
    all(spec -> size(spec) == (plan.nml, tail...), spectra) ||
        throw(DimensionMismatch("incorrect spectral shape"))
    all(spat -> size(spat) == (plan.nlat, plan.nlon, tail...), fields) ||
        throw(DimensionMismatch("incorrect spatial shape"))
    CUDA.context() == plan.context && all(a -> CUDA.context(a) == plan.context, (spectra..., fields...)) ||
        throw(ArgumentError("CUDA plan and arrays must use the active CUDA context"))
    CUDA.stream() == plan.stream ||
        throw(ArgumentError("CUDA plan must be used on the CUDA stream on which it was created"))
    lock(plan.guard) do
        plan.ptr != C_NULL || throw(ArgumentError("CUDA plan is closed"))
        GC.@preserve spectra fields begin
            run(symbol, plan.ptr, spectra..., fields..., plan.lmax)
            CUDA.synchronize(plan.stream)
        end
    end
    return nothing
end

function analyse(symbol, ptr, spec, spat::CuArray{T}, lmax) where T
    ccall(symbol, Cvoid,
        (SHTConfig, CuPtr{T}, CuPtr{Complex{T}}, Cint), ptr, spat, spec, lmax)
end

function synthesise(symbol, ptr, spec, spat::CuArray{T}, lmax) where T
    ccall(symbol, Cvoid,
        (SHTConfig, CuPtr{Complex{T}}, CuPtr{T}, Cint), ptr, spec, spat, lmax)
end

function analysis_scalar!(out, input, plan::CUDAPlan{T}) where T
    readable(input) isa CuArray{T} || throw(ArgumentError("expected a CuArray{$T}"))
    spec = out isa Void ? CuArray{Complex{T}}(undef, plan.nml, size(readable(input))[3:end]...) : out
    spec isa CuArray{Complex{T}} || throw(ArgumentError("expected a CuArray{Complex{$T}} output"))
    transform!(analyse, plan.symbols.analyse, plan, (spec,), (writable(input),))
    return spec
end

function synthesis_scalar!(out, input, plan::CUDAPlan{T}) where T
    spec = readable(input)
    spec isa CuArray{Complex{T}} || throw(ArgumentError("expected a CuArray{Complex{$T}}"))
    spat = out isa Void ? CuArray{T}(undef, plan.nlat, plan.nlon, size(spec)[2:end]...) : out
    spat isa CuArray{T} || throw(ArgumentError("expected a CuArray{$T} output"))
    transform!(synthesise, plan.symbols.synthesise, plan, (spec,), (spat,))
    return spat
end

function vector_spectra(spec, ::Type{T}) where T
    spec isa NamedTuple{(:spheroidal, :toroidal)} &&
        all(a -> a isa CuArray{Complex{T}}, spec) ||
        throw(ArgumentError("expected spheroidal and toroidal CuArray{Complex{$T}} coefficients"))
    return values(spec)
end

function vector_fields(spat, ::Type{T}) where T
    spat isa NamedTuple{(:ucolat, :ulon)} && all(a -> a isa CuArray{T}, spat) ||
        throw(ArgumentError("expected ucolat and ulon CuArray{$T} fields"))
    return values(spat)
end

function analysis_vector!(out, input, plan::CUDAPlan{T}) where T
    vector_fields(readable(input), T)
    if out isa Void
        dims = (plan.nml, size(readable(input).ucolat)[3:end]...)
        out = (spheroidal=CuArray{Complex{T}}(undef, dims),
            toroidal=CuArray{Complex{T}}(undef, dims))
    end
    transform!(analyse_vector, plan.symbols.analyse_vector, plan, vector_spectra(out, T), values(writable(input)))
    return out
end

function spatial_output(out, spec, plan::CUDAPlan{T}) where T
    out isa Void || return out
    dims = (plan.nlat, plan.nlon, size(spec)[2:end]...)
    return (ucolat=CuArray{T}(undef, dims), ulon=CuArray{T}(undef, dims))
end

function synthesis_vector!(out, input, plan::CUDAPlan{T}) where T
    spectra = vector_spectra(readable(input), T)
    spat = spatial_output(out, first(spectra), plan)
    transform!(synthesise_vector, plan.symbols.synthesise_vector, plan, spectra, vector_fields(spat, T))
    return spat
end

function synthesis_spheroidal!(out, spec, plan::CUDAPlan{T}) where T
    spec = readable(spec)
    spec isa CuArray{Complex{T}} || throw(ArgumentError("expected a CuArray{Complex{$T}}"))
    # SHTns 3.7.0 omits the dedicated Float32 symbol. The vector synthesis
    # with zero toroidal coefficients also works in both precisions.
    return synthesis_vector!(out, (spheroidal=spec, toroidal=CUDA.zeros(Complex{T}, size(spec))), plan)
end

function analyse_vector(symbol, ptr, s, t, vt::CuArray{T}, vp, lmax) where T
    ccall(symbol, Cvoid,
        (SHTConfig, CuPtr{T}, CuPtr{T}, CuPtr{Complex{T}}, CuPtr{Complex{T}}, Cint),
        ptr, vt, vp, s, t, lmax)
end

function synthesise_vector(symbol, ptr, s, t, vt::CuArray{T}, vp, lmax) where T
    ccall(symbol, Cvoid,
        (SHTConfig, CuPtr{Complex{T}}, CuPtr{Complex{T}}, CuPtr{T}, CuPtr{T}, Cint),
        ptr, s, t, vt, vp, lmax)
end

end
