# Run in an environment with SHTnsSpheres and CUDA installed and a CUDA-enabled SHTns.
using CUDA, Test
using SHTnsSpheres: SHTnsSphere, cuda_plan, void, erase, analysis_scalar!, synthesis_scalar!

CUDA.allowscalar(false)
const CUDA_SHTNS = ENV["SHTNS_CUDA_LIBRARY"]

@testset "Scalar CUDA transforms" begin
    sph = SHTnsSphere(32, 1)
    x = 1 .+ sph.x .+ 2 .* sph.z
    ref = analysis_scalar!(void, x, sph)
    for (T, rtol) in ((Float64, 1e-12), (Float32, 2e-6))
        @testset "$T" begin
            plan = cuda_plan(sph, T; library=CUDA_SHTNS)
            d = CuArray(T.(x))
            try
                q = analysis_scalar!(void, d, plan)
                @test eltype(q) === Complex{T}
                @test Array(d) == T.(x)
                @test isapprox(Array(q), ref; rtol)
                @test isapprox(Array(synthesis_scalar!(void, CuArray(Complex{T}.(ref)), plan)), x; rtol)
                @test analysis_scalar!(q, erase(d), plan) === q
                @test synthesis_scalar!(d, q, plan) === d
                @test isapprox(Array(d), x; rtol)
                @test_throws DimensionMismatch analysis_scalar!(void, CUDA.zeros(T, 1, 1), plan)
                @test_throws ArgumentError analysis_scalar!(void, CuArray((T === Float64 ? Float32 : Float64).(x)), plan)
            finally
                close(plan)
            end
            close(plan)
            @test_throws ArgumentError analysis_scalar!(void, d, plan)
        end
    end
end

using SHTnsSpheres: analysis_vector!, synthesis_vector!, synthesis_spheroidal!

@testset "Vector CUDA transforms" begin
    sph = SHTnsSphere(32, 1)
    q = analysis_scalar!(void, sph.x .+ 2 .* sph.z, sph)
    spec = (spheroidal=q, toroidal=2im .* q)
    fields = synthesis_vector!(void, spec, sph)
    ref = analysis_vector!(void, fields, sph)
    gradient = synthesis_spheroidal!(void, q, sph)
    for (T, rtol) in ((Float64, 1e-12), (Float32, 2e-6))
        @testset "$T" begin
            plan = cuda_plan(sph, T; library=CUDA_SHTNS)
            try
                d = map(x -> CuArray(T.(x)), fields)
                c = map(x -> CuArray(Complex{T}.(x)), spec)
                result = analysis_vector!(void, d, plan)
                @test all(map((a, b) -> isapprox(Array(a), b; rtol), result, ref))
                @test all(map((a, b) -> Array(a) == T.(b), d, fields))
                @test analysis_vector!(result, erase(d), plan) === result
                @test synthesis_vector!(d, c, plan) === d
                @test all(map((a, b) -> isapprox(Array(a), b; rtol), d, fields))
                g = synthesis_spheroidal!(void, c.spheroidal, plan)
                @test all(map((a, b) -> isapprox(Array(a), b; rtol), g, gradient))
            finally
                close(plan)
            end
        end
    end
end

@testset "Batched CUDA transforms" begin
    sph = SHTnsSphere(32, 1)
    x = cat(sph.x, sph.y, sph.z; dims=3)
    ref = analysis_scalar!(void, x, sph)
    for (T, rtol) in ((Float64, 1e-12), (Float32, 2e-6))
        plan = cuda_plan(sph, T; library=CUDA_SHTNS, batch=3)
        try
            d = CuArray(T.(x))
            q = analysis_scalar!(void, d, plan)
            @test isapprox(Array(q), ref; rtol)
            @test isapprox(Array(synthesis_scalar!(void, q, plan)), x; rtol)
            spec = (spheroidal=q, toroidal=copy(q))
            fields = synthesis_vector!(void, spec, plan)
            result = analysis_vector!(void, fields, plan)
            @test all(map((a, b) -> isapprox(Array(a), Array(b); rtol), result, spec))
            @test_throws DimensionMismatch analysis_scalar!(void, CuArray(T.(sph.x)), plan)
        finally
            close(plan)
        end
    end
end

@testset "Explicit CUDA stream" begin
    sph = SHTnsSphere(32, 1)
    x = sph.x
    stream = CuStream()
    CUDA.stream!(stream) do
        cuda_plan(sph; library=CUDA_SHTNS) do plan
            d = CuArray(x)
            q = analysis_scalar!(void, d, plan)
            @test Array(synthesis_scalar!(void, q, plan)) ≈ x
            CUDA.stream!(CuStream()) do
                @test_throws ArgumentError synthesis_scalar!(void, q, plan)
            end
        end
    end
end
