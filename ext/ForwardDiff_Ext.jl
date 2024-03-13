module ForwardDiff_Ext

import GFDomains: GFDomains,
    analysis_scalar, analysis_scalar!,
    analysis_vector, analysis_vector!,
    analysis_div,
    synthesis_scalar, synthesis_scalar!,
    synthesis_vector, synthesis_vector!,
    synthesis_spheroidal, synthesis_spheroidal!

using GFDomains.SHTns: allocate_shtns
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

function apply(fun, arg, sph, backend)
    v = fun(value(arg), sph, backend)
    p = fun(partial(arg), sph, backend)
    return dual(tag(arg), v, p)
end

analysis_scalar(spat::ScalarSpat, sph, backend) = apply(analysis_scalar, spat, sph, backend)
analysis_vector(spat::VectorSpat, sph, backend) = apply(analysis_vector, spat, sph, backend)
analysis_div(spat::VectorSpat, sph, backend) = apply(analysis_div, spat, sph, backend)
synthesis_scalar(spec::ScalarSpec, sph, backend) = apply(synthesis_scalar, spec, sph, backend)
synthesis_vector(spec::VectorSpec, sph, backend) = apply(synthesis_vector, spec, sph, backend)
synthesis_spheroidal(spec::ScalarSpec, sph, backend) = apply(synthesis_spheroidal, spec, sph, backend)

#==================== mutating ================#

dual!(T) = (out,v,p)->dual!(T,out,v,p)
dual!(T::Type, out::A, v, p) where {A<:Array} = map!(Dualizer{T}(), out, v, p)
dual!(T::Type, out::NT, v, p) where {NT<:NamedTuple} = map!(dual!(T), out, v, p)

function apply!(val::Val, fun!, output, input, sph, backend)
    v = allocate_shtns(val, sph) # values
    p = allocate_shtns(val, sph) # partial
    fun!(v, value(input), sph, backend)
    fun!(p, partial(input), sph, backend)
    dual!(tag(spec), output, v, p)
    return output
end

analysis_scalar!(spec::ScalarSpec, spat::VectorSpat, sph, backend) =
    apply!(Val(:scalar_spec), analysis_scalar!, spec, spat, sph, backend)

analysis_vector!(spec::VectorSpec, spat::VectorSpat, sph, backend) =
    apply!(Val(:vector_spec), analysis_vector!, spec, spat, sph, backend)

synthesis_scalar!(spat::ScalarSpat, spec::ScalarSpec, sph, backend) =
    apply!(Val(:scalar_spat), synthesis_scalar!, spec, spat, sph, backend)

synthesis_vector!(spat::VectorSpat, spec::VectorSpec, sph, backend) =
    apply!(Val(:vector_spat), synthesis_vector!, spec, spat, sph, backend)

synthesis_spheroidal!(spat::VectorSpat, spec::ScalarSpec, sph, backend) =
    apply!(Val(:vector_spat), synthesis_spheroidal!, spec, spat, sph, backend)

end
