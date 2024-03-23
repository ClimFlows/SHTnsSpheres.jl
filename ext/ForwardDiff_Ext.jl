module ForwardDiff_Ext

import SHTnsSpheres:
    Void, void, In,
    similar_spec, similar_spat, allocate_shtns,
    analysis_scalar, analysis_scalar!,
    analysis_vector, analysis_vector!,
    analysis_div,
    synthesis_scalar!,
#    synthesis_vector, synthesis_vector!,
    synthesis_spheroidal, synthesis_spheroidal!

using ForwardDiff: Dual, Partials

#=== Abbreviations for types we commonly dispatch on ===#

const DualF64{D,T,N} = Array{Dual{T,Float64,N},D}
const DualC64{D,T,N} = Array{Complex{Dual{T,Float64,N}},D}

const ScalarSpat{T,N} = DualF64{2,T,N}
const ScalarSpec{T,N} = DualC64{1,T,N}
const VectorSpat{T,N} = @NamedTuple{ucolat::DualF64{2,T,N}, ulon::DualF64{2,T,N}}
const VectorSpec{T,N} = @NamedTuple{spheroidal::DualC64{1,T,N}, toroidal::DualC64{1,T,N}}

tag(::ScalarSpat{T}) where T = T
tag(::ScalarSpec{T}) where T = T
tag(::VectorSpat{T}) where T = T
tag(::VectorSpec{T}) where T = T

similar_spec(spat::DualF64{1,T,N}, sph) where {T,N} = allocate_shtns(Dual{T,Float64,N}, Val(:scalar_spec), sph)
similar_spat(spec::DualC64{1,T,N}, sph) where {T,N} = allocate_shtns(Dual{T,Float64,N}, Val(:scalar_spat), sph)

allocate_shtns(T, ::Val{:scalar_spec}, sph, args...) = shtns_alloc_spec(T, sph, args...)
allocate_shtns(T, ::Val{:scalar_spat}, sph, args...) = shtns_alloc_spat(T, sph, args...)

shtns_alloc_spat(F, sph, dims...)         = Array{F}(undef, sph.nlat, 2*sph.nlat, dims...)
shtns_alloc_spec(F, sph, dims...)         = Array{Complex{F}}(undef, sph.nml, dims...)


#========= low-level helpers to separate then recombine value and partials ===========#

# extract value and partial from x::Dual or z::Complex{Dual}
value(x::Complex{<:Dual}) = complex(x.re.value, x.im.value)
value(x::Dual) = x.value
value(x::Array) = value.(x)

partial(x::Dual{T,V,1}) where {T,V} = x.partials.values[1]
partial(x::Complex{Dual{T,V,1}}) where {T,V} = complex(x.re.partials.values[1], x.im.partials.values[1])
partial(x::Array) = partial.(x)

value(uv::NamedTuple) = map(value, uv)
partial(uv::NamedTuple) = map(partial, uv)

# recombine value and partial into Dual or Complex{Dual}
struct Dualizer{T} end
(::Dualizer{T})(v::V, p::V) where {T,V<:Real} = Dual{T,V,1}(v, Partials{1,V}((p,)))
(dual::Dualizer{T})(zv::V, zp::V) where {T,V<:Complex} = complex(dual(zv.re, zp.re), dual(zv.im, zp.im))

#================== non-mutating ===================#

dual(T) = (v,p)->dual(T,v,p)
dual(T::Type, v::A, p::A) where {A<:Array} = map(Dualizer{T}(), v, p)
dual(T::Type, v::NT, p::NT) where {NT<:NamedTuple} = map(dual(T), v, p)

function apply(fun, arg, sph)
    v = fun(value(arg), sph)
    p = fun(partial(arg), sph)
    return dual(tag(arg), v, p)
end

analysis_scalar(spat::ScalarSpat, sph) = apply(analysis_scalar, spat, sph)
analysis_vector(spat::VectorSpat, sph) = apply(analysis_vector, spat, sph)
analysis_div(spat::VectorSpat, sph) = apply(analysis_div, spat, sph)
synthesis_scalar(spec::ScalarSpec, sph) = apply(synthesis_scalar, spec, sph)
synthesis_vector(spec::VectorSpec, sph) = apply(synthesis_vector, spec, sph)
synthesis_spheroidal(spec::ScalarSpec, sph) = apply(synthesis_spheroidal, spec, sph)

#==================== mutating ================#

dual!(T) = (out,v,p)->dual!(T,out,v,p)
dual!(T::Type, out::A, v, p) where {A<:Array} = map!(Dualizer{T}(), out, v, p)
dual!(T::Type, out::NT, v, p) where {NT<:NamedTuple} = map!(dual!(T), out, v, p)

function apply!(val::Val, fun!, output, input, sph)
    v = allocate_shtns(val, sph) # values
    p = allocate_shtns(val, sph) # partial
    fun!(v, value(input), sph)
    fun!(p, partial(input), sph)
    dual!(tag(input), output, v, p)
    return output
end

analysis_scalar!(spec::ScalarSpec, spat::VectorSpat, sph) =
    apply!(Val(:scalar_spec), analysis_scalar!, spec, spat, sph)
analysis_scalar!(::Void, spat::VectorSpat, sph) =
    apply!(Val(:scalar_spec), analysis_scalar!, similar_spec(spat, sph), spat, sph)

analysis_vector!(spec::VectorSpec, spat::VectorSpat, sph) =
    apply!(Val(:vector_spec), analysis_vector!, spec, spat, sph)

synthesis_scalar!(spat::ScalarSpat, spec::ScalarSpec, sph) =
    apply!(Val(:scalar_spat), synthesis_scalar!, spat, spec, sph)

synthesis_vector!(spat::VectorSpat, spec::VectorSpec, sph) =
    apply!(Val(:vector_spat), synthesis_vector!, spat, spec, sph)

synthesis_spheroidal!(spat::VectorSpat, spec::ScalarSpec, sph) =
    apply!(Val(:vector_spat), synthesis_spheroidal!, spat, spec, sph)

end
