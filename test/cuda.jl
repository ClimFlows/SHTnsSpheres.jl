# Run in an environment with SHTnsSpheres and CUDA installed and a CUDA-enabled SHTns.
using CUDA, Test
using SHTnsSpheres: SHTnsSphere, void, erase, sample_scalar!, analysis_scalar!, synthesis_scalar!

CUDA.allowscalar(false)

@testset "Scalar CUDA transforms" begin
    sph = SHTnsSphere(32, 1)
    x = 1 .+ sph.x .+ 2 .* sph.z
    ref = analysis_scalar!(void, x, sph)
    for (T, rtol) in ((Float64, 1e-12), (Float32, 2e-6))
        @testset "$T" begin
            plan = SHTnsSphere(32; backend=:cuda, precision=T)
            try
                d = sample_scalar!(void, (x, y, z, lon, lat) -> one(x) + x + 2z, plan)
                @test d isa CuMatrix{T}
                @test isapprox(Array(d), x; rtol)
                original = Array(d)
                q = analysis_scalar!(void, d, plan)
                @test eltype(q) === Complex{T}
                @test plan isa SHTnsSphere
                @test plan.x isa CuMatrix{T}
                @test Array(d) == original
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
            @test_throws ArgumentError analysis_scalar!(void, plan.x, plan)
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
            plan = SHTnsSphere(32; backend=:cuda, precision=T)
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
    @test_throws ArgumentError SHTnsSphere(32; backend=:cuda, batch=())
    @test_throws ArgumentError SHTnsSphere(32; backend=:cuda, batch=[1, 0])
    sph = SHTnsSphere(32, 1)
    x = cat(sph.x, sph.y, sph.z; dims=3)
    ref = analysis_scalar!(void, x, sph)
    for (T, rtol) in ((Float64, 1e-12), (Float32, 2e-6))
        plan = SHTnsSphere(32; backend=:cuda, precision=T, batch=(3, 1, 3))
        try
            @test Set(keys(plan.backend.plans)) == Set((1, 3))
            d = CuArray(T.(x))
            q = analysis_scalar!(void, d, plan)
            @test isapprox(Array(q), ref; rtol)
            @test isapprox(Array(synthesis_scalar!(void, q, plan)), x; rtol)
            spec = (spheroidal=q, toroidal=copy(q))
            fields = synthesis_vector!(void, spec, plan)
            result = analysis_vector!(void, fields, plan)
            @test all(map((a, b) -> isapprox(Array(a), Array(b); rtol), result, spec))
            d4 = reshape(cat(d, plan.x; dims=3), 32, 64, 2, 2)
            q4 = analysis_scalar!(void, d4, plan)
            @test size(q4) == (sph.nml, 2, 2)
            @test isapprox(Array(synthesis_scalar!(void, q4, plan)), Array(d4); rtol)
            cached = plan.backend.plans[4]
            @test isapprox(Array(synthesis_scalar!(void, q, plan)), x; rtol)
            @test isapprox(Array(synthesis_scalar!(void, q4, plan)), Array(d4); rtol)
            @test plan.backend.plans[4] == cached
            @test Set(keys(plan.backend.plans)) == Set((1, 3, 4))
            @test_throws ArgumentError analysis_scalar!(void, CUDA.zeros(T, 32, 64, 0), plan)
        finally
            close(plan)
        end
        @test isempty(plan.backend.plans)
    end
end

@testset "Explicit CUDA stream" begin
    sph = SHTnsSphere(32, 1)
    x = sph.x
    stream = CuStream()
    CUDA.stream!(stream) do
        plan = SHTnsSphere(32; backend=:cuda)
        try
            d = CuArray(x)
            q = analysis_scalar!(void, d, plan)
            @test Array(synthesis_scalar!(void, q, plan)) ≈ x
            CUDA.stream!(CuStream()) do
                @test_throws ArgumentError synthesis_scalar!(void, q, plan)
            end
        finally
            close(plan)
        end
    end
end
