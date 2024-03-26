using Test, Zygote, ForwardDiff
using SHTnsSpheres: SHTnsSphere, void,
    sample_scalar!, synthesis_scalar!, analysis_scalar!,
    sample_vector!, analysis_vector!, synthesis_vector!, synthesis_spheroidal!,
    curl!, divergence!

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

scalar(x,y,z,lon,lat) = (x*y*z)^3

function test_inv(sph, F=Float64)
    spat = sample_scalar!(void, scalar, sph)
    spec = analysis_scalar!(void, copy(spat), sph) # modifies input !
    spat2 = synthesis_scalar!(void, spec, sph) # pure
    spec2 = analysis_scalar!(void, copy(spat2), sph) # modifies input !
    @test spat ≈ spat2
    @test spec ≈ spec2

    uv = sample_vector!(void, velocity, sph)
    spec = analysis_vector!(void, map(copy,uv), sph) # modifies input !
    uv2 = synthesis_vector!(void, spec, sph)
    @test uv.ulon ≈ uv2.ulon
    @test uv.ucolat ≈ uv2.ucolat
end

#========= Check consistency of adjoint and tangent =======#

function test_AD(sph, F=Float64)
    (; x,y,z) = sph
    spat = @. (y+2z)^2
    dspat = @. x^2
    spec = analysis_scalar!(void, spat, sph)
    dspec = analysis_scalar!(void, dspat, sph)

    check_gradient(spec, dspec) do fspec
        fspat = synthesis_scalar!(void, fspec, sph)
        return analysis_scalar!(void, fspat.^2, sph)
    end

    check_gradient(spec, dspec) do fspec
        u, v = synthesis_spheroidal!(void, fspec, sph)
        k = @. u^2+v^2
        analysis_scalar!(void, k, sph)
    end

    check_gradient(spec, dspec) do fspec
        uv = synthesis_spheroidal!(void, fspec, sph)
        u, v = uv
        k = @. u^2+v^2
        uv = map(x->k.*x, uv)
        uv_spec = analysis_vector!(void, uv, sph)
        divergence!(void, uv_spec, sph)
    end

    check_gradient(spec, dspec) do fspec
        uv = synthesis_spheroidal!(void, fspec, sph)
        u, v = uv
        k = @. u^2+v^2
        uv = map(x->k.*x, uv)
        uv_spec = analysis_vector!(void, uv, sph)
        curl!(void, uv_spec, sph)
    end

end

lmax = 32
sph = SHTnsSphere(lmax)
@show sph
@testset "synthesis∘analysis == identity" test_inv(sph)
@testset "Autodiff for SHTns" test_AD(sph)
