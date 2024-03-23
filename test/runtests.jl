using Test, Zygote, ForwardDiff
using SHTnsSpheres: SHTnsSphere,
    synthesis_scalar, analysis_scalar, synthesis_spheroidal, analysis_div

Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}") #src
Base.isapprox(a::NT, b::NT) where {NT<:NamedTuple} = all(map(isapprox, a,b))
Base.copy(t::NamedTuple) = map(copy, t)

#========= Helper functions ==========#

@inline cprod(z1, z2) = z1.re * z2.re + z1.im * z2.im
dot_spec(a, b) = @inline mapreduce(cprod, +, a, b)
dot_spat(a, b) = @inline mapreduce(*, +, a, b)

dot(a::Array{T}, b::Array{T}) where {T<:Complex} = dot_spec(a,b)
dot(a::Array{T}, b::Array{T}) where {T<:Real} = dot_spat(a,b)

function check_gradient(fun, state, dstate, args...)
    loss(s) = sum(abs2, fun(s, args...))
    fwd_grad = ForwardDiff.derivative(x->loss(@. state + x * dstate), 0.0)
    bwd_grad = dot(Zygote.gradient(loss, copy(state))[1], dstate)
    @info fun fwd_grad bwd_grad
    @test fwd_grad ≈ bwd_grad

    # loss = <f(s),f(s)>
    # dloss = 2<f(s)|df|ds> = 2<ds|df*|f(s)>
    #    f = fun(state)
    #    f, df, adf = fun(state, dstate, f)
    #    fwd_exact = 2*dot_spec(f,df)
    #    bwd_exact = 2*dot_spec(dstate,adf)
    #    @info fun fwd_exact fwd_grad
    #    @info fun bwd_exact bwd_grad
    return nothing
end

#========= Check that synthesis and analysis are inverses =======#

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

#========= Check consistency of adjoint and tangent =======#

function test_AD(sph, F=Float64)
    @show sph
    synthesis(f) = synthesis_scalar(copy(f), sph)
    analysis(f) = analysis_scalar(copy(f), sph)

    (; x,y,z) = sph
    spat = @. (y+2z)^2
    dspat = @. x^2
    spec = analysis(spat)
    dspec = analysis(dspat)

    check_gradient(spec, dspec) do fspec
        fspat = synthesis(fspec)
        return analysis(fspat.^2)
    end

    check_gradient(spec, dspec) do fspec
        uv = synthesis_spheroidal(fspec, sph)
        u, v = uv
        k = @. u^2+v^2
        uv = map(x->k.*x, uv)
        analysis_div(uv, sph)
    end

    check_gradient(spec, dspec) do fspec
        u, v = synthesis_spheroidal(fspec, sph)
        k = @. u^2+v^2
        analysis_scalar(k, sph)
    end

end

lmax = 32
sph = SHTnsSphere(lmax)
@testset "Autodiff for SHTns" test_AD(sph)
