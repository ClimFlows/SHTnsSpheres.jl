# Run with CUDA.jl, Enzyme.jl and a CUDA-enabled SHTns_jll.
using CUDA, Enzyme, Test
using SHTnsSpheres: SHTnsSphere, void, analysis_scalar!, synthesis_scalar!

CUDA.allowscalar(false)

function check_cuda_gradient(transform!, state, direction, plan, rtol)
    # Materialize abs2 on the GPU so Enzyme can use the rule for plain sum.
    loss(x) = sum(abs2.(transform!(void, x, plan)))
    y = transform!(void, state, plan)
    dy = transform!(void, direction, plan)
    reference = 2 * sum(real.(conj.(y) .* dy))
    mode = Enzyme.set_runtime_activity(Reverse)
    gradient = only(Enzyme.gradient(mode, Const(loss), copy(state)))
    @test gradient isa CuArray
    @test isapprox(sum(real.(conj.(gradient) .* direction)), reference; rtol)
end

@testset "CUDA scalar gradients with Enzyme" begin
    for (T, rtol) in ((Float64, 2e-12), (Float32, 2e-5)), batch in (1, 3)
        @testset "$T, batch=$batch" begin
            plan = SHTnsSphere(32; backend=:cuda, precision=T, batch)
            try
                x = batch == 1 ? plan.x : cat(plan.x, plan.y, plan.z; dims=3)
                dx = @. x^2 + T(0.25) * x + 1
                q = analysis_scalar!(void, x, plan)
                dq = analysis_scalar!(void, dx, plan)
                check_cuda_gradient(analysis_scalar!, x, dx, plan, rtol)
                check_cuda_gradient(synthesis_scalar!, q, dq, plan, rtol)
            finally
                close(plan)
            end
        end
    end
end
