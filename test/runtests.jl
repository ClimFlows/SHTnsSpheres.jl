using Test, Zygote, ForwardDiff
using SHTnsSpheres: SHTnsSphere,
    synthesis_scalar, analysis_scalar, synthesis_spheroidal, analysis_div

Base.show(io::IO, ::Type{<:ForwardDiff.Tag}) = print(io, "Tag{...}") #src

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
