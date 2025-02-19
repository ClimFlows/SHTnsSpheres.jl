using Test, Zygote, ForwardDiff, Enzyme
using SHTnsSpheres: SHTnsSphere, void, batch, 
    shtns_use_threads,
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
    spec = analysis_scalar!(void, spat, sph)
    spat2 = synthesis_scalar!(void, spec, sph)
    spec2 = analysis_scalar!(void, spat2, sph)
    @test spat ≈ spat2
    @test spec ≈ spec2

    uv = sample_vector!(void, velocity, sph)
    spec = analysis_vector!(void, uv, sph)
    uv2 = synthesis_vector!(void, spec, sph)
    @test uv.ulon ≈ uv2.ulon
    @test uv.ucolat ≈ uv2.ucolat
end

function test_batch(sph)
    # check batched transform
    # check that we can get the same result by using `batch` manually
    function test(fun, in)
        ref = fun(void, in, sph)

        sim(x) = similar(x)
        sim(x::Union{Tuple, NamedTuple}) = map(sim, x)
        out = sim(ref)

        depth(x::Array{Float64}) = size(x,3)
        depth(x::Array{ComplexF64}) = size(x,2)
        depth(x::NamedTuple) = depth(x[1])

        batch(sph, depth(out), 1) do ptr, thread, k, _
            slice(x::Array{Float64}) = @views x[:,:,k]
            slice(x::Array{ComplexF64}) = @views x[:,k]
            slice(x::NamedTuple) = map(slice, x)
            fun(slice(out), slice(in), ptr)
        end
        passed = ref ≈ out
        passed || @error "test_batch FAILED" fun
        @test passed
        return ref
    end

    spat_2D = sample_scalar!(void, scalar, sph)
    spat = similar(spat_2D, (size(spat_2D, 1), size(spat_2D,2), 10))
    @. spat = spat_2D

    spec = test(analysis_scalar!, spat)
    spat2 = test(synthesis_scalar!, spec)
    _ = test(analysis_scalar!, spat2)

    uv_2D = sample_vector!(void, velocity, sph)
    ucolat, ulon = similar(spat), similar(spat)
    @. ucolat = uv_2D.ucolat
    @. ulon = uv_2D.ulon
    uv = (; ucolat, ulon)
    spec = test(analysis_vector!, uv)
    _ = test(synthesis_vector!, spec)
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



#========= Check the phase of the spherical harmonics on one example =======#

function Y11(x,y,z,lon,lat)
    θ = π/2 - lat
    ϕ = lon
    return -sqrt(3/(2π)) * real(exp(1im * ϕ)) * sin(θ)
end

function test_azimuthal_phase(sph)
    xy_spat = sample_scalar!(void, Y11, sph)
    xy_spec = analysis_scalar!(void, xy_spat, sph)
    l, m = 1, 1
    LM = (m * (2 * sph.lmax + 2 - (m + 1))) >> 1 + l + 1
    result = xy_spec[LM]
    @test real(result) ≈ 1.0
    @test imag(result) ≈ 0.0 atol=eps()
end


@testset "test thread setup" shtns_use_threads()

nlat = 128
sph = SHTnsSphere(nlat)
@show sph
@testset "synthesis∘analysis == identity" test_inv(sph)
@testset "batched transforms" test_batch(sph)
@testset "Autodiff for SHTns" test_AD(sph)
@testset "azimuthal phase aligned with coordinates" test_azimuthal_phase(sph)

include("scaling.jl")
