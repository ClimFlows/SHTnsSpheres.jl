using SHTnsSpheres: shtns_alloc
using ThreadPinning
pinthreads(:cores)
threadinfo()

function time(fun, N)
    fun()
    times = [(@timed fun()).time for i=1:N+10]
    sort!(times)
    return sum(times[1:N])/N
end

function scaling(fun, name, sph, N)
    @info "Multithread scaling for $name with $sph"
    @info "Threads \t elapsed \t speedup \t efficiency"
    single=1e9
    for nt in 1:length(sph.ptrs)
        elapsed = time(N) do
            fun(SHTnsSphere(sph, nt))
        end
        nt==1 && (single=elapsed)
        speedup = single/elapsed
        percent(x) = round(100x; digits=0)
        @info "$nt \t\t $(round(elapsed; digits=4)) \t $(percent(speedup)) \t\t $(percent(speedup/nt))" 
    end
end

function scaling_synth(sph)
    nz = 40
    spec = shtns_alloc(Float64, Val(:scalar_spec), sph, nz)
    spat = shtns_alloc(Float64, Val(:scalar_spat), sph, nz)
    scaling("synthesis_scalar!", sph, 100) do sph
        synthesis_scalar!(spat, spec, sph)
    end
end

scaling_synth(sph)
