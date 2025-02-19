module Enzyme_Ext

using SHTnsSpheres: priv, Adjoints, void, Void, transform!

import Enzyme.EnzymeRules: reverse, augmented_primal
using Enzyme.EnzymeRules
using Enzyme:Const, Duplicated, Annotation, make_zero!

#=============================================================#

using SHTnsSpheres: SHTnsSphere, void, transform!, analysis_scalar!, synthesis_scalar!

function augmented_primal(
    _,
    ::Const{typeof(transform!)},
    ::Type{<:Annotation},
    fun::Const{<:Function},
    sph::Const{SHTnsSphere},
    spec::Duplicated, 
    spat::Duplicated)

    transform!(fun.val, sph.val, spec.val, spat.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function reverse(
    _,
    ::Const{typeof(transform!)},
    ::Type{<:Annotation},
    ::Nothing, # tape
    fun::Const{<:Function},
    sph::Const{SHTnsSphere},
    spec::Duplicated, 
    spat::Duplicated)

    adjoint_transform(fun.val, sph.val, spec, spat)
    return (nothing, nothing, nothing, nothing)
end

#==== adjoint of analysis: zero out spec.dval after computing spat.dval ====#

function adjoint_transform(::typeof(priv.spat_to_SH), sph, spec, spat)
    _, dspat, _, _ = Adjoints.adjoint_analysis_scalar(spec.dval, sph)
    @. spat.dval += dspat
    make_zero!(spec.dval)
    return nothing
end

function adjoint_transform(::typeof(priv.spat_to_SHsphtor), sph, spec, spat)
    _, dspat, _, _ = Adjoints.adjoint_analysis_vector(spec.dval, sph)
    @. spat.dval.ucolat += dspat.ucolat
    @. spat.dval.ulon += dspat.ulon
    make_zero!(spec.dval)
    return nothing
end

#==== adjoint of synthesis: zero out spat.dval after computing spec.dval ====#

function adjoint_transform(::typeof(priv.SH_to_spat), sph, spec, spat)
    _, dspec, _, _ = Adjoints.adjoint_synthesis_scalar(spat.dval, sph)
    @. spec.dval += dspec
    make_zero!(spat.dval)
    return nothing
end

function adjoint_transform(::typeof(priv.SHsphtor_to_spat), sph, spec, spat)
    _, dspec, _, _ = Adjoints.adjoint_synthesis_vector(spat.dval, sph)
    @. spec.dval.toroidal += dspec.toroidal
    @. spec.dval.spheroidal += dspec.spheroidal
    make_zero!(spat.dval)
    return nothing
end

function adjoint_transform(::typeof(priv.SHsph_to_spat), sph, spec, spat)
    _, dspec, _, _ = Adjoints.adjoint_synthesis_spheroidal(spat.dval, sph)
    @. spec.dval += dspec
    make_zero!(spat.dval)
    return nothing
end

end # module
