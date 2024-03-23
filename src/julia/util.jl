# void type

const void = Val(:void)
const Void = Val{:void}
const Out{T} = Union{Void,T}

# isvoid(x) = isa(x, Void)
ifvoid(x, f, args) = x
ifvoid(x::Void, f, args) = f(args...)

# allow input arguments to be modified

# user :
#   pass inout(x) as input to express that it is OK to overwrite x
# this module:
#   x = readable(x) for inputs that will not be modified
#   x = writable(x) for inputs that will be modified

"""
Wraps data to mark it as writeable.
"""
struct InOut{A}
    data::A
end
const In{T} = Union{T, InOut{T}}

"""
    y = inout(x)
Some functions need to make copies of input arguments to avoid modifying their contents
and remain pure. Passing `inout(x)` as input argument is equivalent to passing `x`, except
that it explicitly allows to modify the contents of `x`, thus avoiding
copying and allocating.
"""
inout(x) = InOut(x)

readable(x) = x
readable(x::InOut) = x.data
writable(x) = copy_input(x)
writable(x::InOut) = x.data
copy_input(x) = copy(x)
copy_input(x::NamedTuple) = map(copy, x)


function sample_vector(ulonlat, sph::SHTnsSphere)
    (; x, y, z, lon, lat) = sph
    ucolat = @. -getindex(ulonlat(x, y, z, lon, lat), 2)
    ulon = @. getindex(ulonlat(x, y, z, lon, lat), 1)
    return (; ucolat, ulon)
end

#========= Check that synthesis and analysis are inverses =======#

Base.isapprox(a::NT, b::NT) where {NT<:NamedTuple} = all(map(isapprox, a,b))
Base.copy(t::NamedTuple) = map(copy, t)

## Returns a smooth vector field a∇x+b∇y+c∇z with a,b,c smooth scalar fields
function velocity(x,y,z,lon,lat)
    # ∇x = ∇(cosλ cosϕ) = -sinλ, -cosλ sinϕ
    # ∇y = ∇(sinλ cosϕ) = cosλ, -sinλ sinϕ
    # ∇z = ∇ sinϕ = 0, cos ϕ
    a, b, c = (y+2z)^2, (z+2y)^2, (x+2z)^2
    a, b, c = (0,0,1)
    ulon = b*cos(lon)-a*sin(lon)
    ulat = c*cos(lat)-sin(lat)*(a*cos(lon)+b*sin(lon))
    return ulon, ulat
end

function test_inv(sph, F=Float64)
    uv = sample_vector(velocity, sph)
    spec = analysis_vector!(void, uv, sph)
    uv2 = synthesis_vector!(void, spec, sph)
    @test uv.ulon ≈ uv2.ulon
    @test uv.ucolat ≈ uv2.ucolat
end
