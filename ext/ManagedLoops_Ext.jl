module ManagedLoops_Ext

using SHTnsSpheres: SHTnsSphere
import SHTnsSpheres: batch

using ManagedLoops: LoopManager, no_simd, @loops

struct ManagedSphere{Mgr<:LoopManager}
    sph::SHTnsSphere
    mgr::Mgr
end

Base.getindex(sph::SHTnsSphere, mgr::LoopManager) = ManagedSphere(sph, mgr)

batch(fun, msph::ManagedSphere, nk, nl) = managed_batch(no_simd(msph.mgr), fun, msph.sph, nk, nl, length(msph.sph.ptrs))

@loops function managed_batch(_, fun, sph, nk, nl, nthreads)
    let trange = 1:nthreads
        for thread in trange
            ptr = sph.ptrs[thread]
            start, stop = div(nk*(thread-1), nthreads), div(nk*thread, nthreads)
            for k in start+1:stop, l=1:nl
                fun(ptr, k, l)
            end
        end
    end
end

end # module
