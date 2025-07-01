module ForwardDiff_Ext

import SHTnsSpheres:
    Void, void, In, Writable, erase,
    similar_spec, similar_spat, shtns_alloc,
    analysis_scalar!, synthesis_scalar!,
    analysis_vector!, synthesis_vector!, synthesis_spheroidal!,
    SHTVectorSpat, SHTVectorSpec

using ForwardDiff: Dual, Partials

#=== Abbreviations for types we commonly dispatch on ===#

const DualF64{D,T,N} = Array{Dual{T,Float64,N},D}
const DualC64{D,T,N} = Array{Complex{Dual{T,Float64,N}},D}

const MaybeErase{T} = Union{T, Writable{T}}
const ScalarSpat{T,N} = MaybeErase{DualF64{2,T,N}}
const ScalarSpec{T,N} = MaybeErase{DualC64{1,T,N}}
const VectorSpat{T,N} = MaybeErase{@NamedTuple{ucolat::DualF64{2,T,N}, ulon::DualF64{2,T,N}}}
const VectorSpec{T,N} = MaybeErase{@NamedTuple{spheroidal::DualC64{1,T,N}, toroidal::DualC64{1,T,N}}}

tag(::ScalarSpat{T}) where T = T
tag(::ScalarSpec{T}) where T = T
tag(::VectorSpat{T}) where T = T
tag(::VectorSpec{T}) where T = T

#========= low-level helpers to separate then recombine value and partials ===========#

# extract value and partial from x::Dual or z::Complex{Dual}
value(x::Complex{<:Dual}) = complex(x.re.value, x.im.value)
value(x::Dual) = x.value
value(x::Array) = value.(x)
value(uv::NamedTuple) = map(value, uv)
value(x::Writable) = value(x.data)

partial(x::Dual{T,V,1}) where {T,V} = x.partials.values[1]
partial(x::Complex{Dual{T,V,1}}) where {T,V} = complex(x.re.partials.values[1], x.im.partials.values[1])
partial(x::Array) = partial.(x)
partial(uv::NamedTuple) = map(partial, uv)
partial(x::Writable) = partial(x.data)

# since we allocate arrays for values and partials, we can mark them as writable and avoid a few allocations
marker(x::Array) = marker(eltype(x))
marker(x::NamedTuple) = marker(eltype(x[1]))
marker(::Type{<:Complex}) = identity # do nothing for spectral data
marker(::Type{<:Real}) = erase # mark spatial data as eraseable
unmark(x::Writable) = x.data
unmark(x) = x

# recombine value and partial into Dual or Complex{Dual}
struct Dualizer{T} end
(::Dualizer{T})(v::V, p::V) where {T,V<:Real} = Dual{T,V,1}(v, Partials{1,V}((p,)))
(dual::Dualizer{T})(zv::V, zp::V) where {T,V<:Complex} = complex(dual(zv.re, zp.re), dual(zv.im, zp.im))

#================== non-mutating ===================#

dual(T) = (v,p)->dual(T,v,p)
dual(T::Type, v::A, p::A) where {A<:Array} = map(Dualizer{T}(), v, p)
dual(T::Type, v::NT, p::NT) where {NT<:NamedTuple} = map(dual(T), v, p)

function apply(fun!, arg, sph)
    arg = unmark(arg)
    mark = marker(arg)
    v = fun!(void, mark(value(arg)), sph)
    p = fun!(void, mark(partial(arg)), sph)
    return dual(tag(arg), v, p)
end

analysis_scalar!(::Void, spat::ScalarSpat, sph) = apply(analysis_scalar!, spat, sph)
analysis_vector!(::Void, spat::VectorSpat, sph) = apply(analysis_vector!, spat, sph)
analysis_div!(::Void, spat::VectorSpat, sph) = apply(analysis_div, spat!, sph)
synthesis_scalar!(::Void, spec::ScalarSpec, sph) = apply(synthesis_scalar!, spec, sph)
synthesis_vector!(::Void, spec::VectorSpec, sph) = apply(synthesis_vector!, spec, sph)
synthesis_spheroidal!(::Void, spec::ScalarSpec, sph) = apply(synthesis_spheroidal!, spec, sph)

#==================== mutating ================#

dual!(T) = (out,v,p)->dual!(T,out,v,p)
dual!(T::Type, out::A, v, p) where {A<:Array} = map!(Dualizer{T}(), out, v, p)
function dual!(T::Type, out::SHTVectorSpec, v::SHTVectorSpec, p::SHTVectorSpec)
    dual!(T, out.toroidal, v.toroidal, p.toroidal)
    dual!(T, out.spheroidal, v.spheroidal, p.spheroidal)
    return out
end

function apply!(val::Val, fun!, output, input, sph)
    v = shtns_alloc(Float64, val, sph) # values
    p = shtns_alloc(Float64, val, sph) # partial
    input = unmark(input)
    mark = marker(input)
    fun!(v, marker(value(input)), sph)
    fun!(p, marker(partial(input)), sph)
    dual!(tag(input), output, v, p)
    return output
end

analysis_scalar!(spec::ScalarSpec, spat::ScalarSpat, sph) =
    apply!(Val(:scalar_spec), analysis_scalar!, spec, spat, sph)
# analysis_scalar!(::Void, spat::ScalarSpat, sph) =
#    apply!(Val(:scalar_spec), analysis_scalar!, similar_spec(spat, sph), spat, sph)

analysis_vector!(spec::VectorSpec, spat::VectorSpat, sph) =
    apply!(Val(:vector_spec), analysis_vector!, spec, spat, sph)

synthesis_scalar!(spat::ScalarSpat, spec::ScalarSpec, sph) =
    apply!(Val(:scalar_spat), synthesis_scalar!, spat, spec, sph)

# synthesis_vector!(spat::VectorSpat, spec::VectorSpec, sph) =
#    apply!(Val(:vector_spat), synthesis_vector!, spat, spec, sph)

synthesis_spheroidal!(spat::VectorSpat, spec::ScalarSpec, sph) =
    apply!(Val(:vector_spat), synthesis_spheroidal!, spat, spec, sph)

# synthesis_spheroidal!(::Void, spec::ScalarSpec, sph) =
#    apply!(Val(:vector_spat), synthesis_spheroidal!, similar_spat(spec, sph), spec, sph)

end
