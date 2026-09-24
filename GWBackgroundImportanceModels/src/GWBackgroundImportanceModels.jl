"""
    GWBackgroundImportanceModels

Concrete, reusable importance-model adapters for `GWBackgroundInference`. The package owns
astrophysical model choices; the inference package owns the sampler.

The seam between them is a **callable**, `merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)`,
so prepared models here are functors and this package deliberately does **not** depend on
`GWBackgroundInference` -- nothing is imported from it and no methods are added to its
generics. The two-package split is a convenience, not a coupling.
"""
module GWBackgroundImportanceModels

using GWDistributions:
                       DEFAULT_Z_GRID,
                       MadauDickinsonSourceFrame,
                       RedshiftInterpolatedDistribution,
                       normalizer
using GWBackground: SGWBCatalog
using BackgroundCosmology: AbstractCosmology, cosmology, distance_and_volume_grid
using DataInterpolations: LinearInterpolation
using Turing

export AbstractPropagation,
       GR,
       ModifiedPropagation,
       propagation,
       gw_em_distance_ratio,
       log_gw_em_distance_ratio,
       apply_gw_distance_correction,
       apply_gw_distance_correction!,
       BNSMadauDickinsonImportanceModel,
       prepare_bns_madau_dickinson_model,
       AMPLITUDE_PARAMETERS,
       amplitude_H0,
       amplitude_R₀,
       merger_rate_amplitude_H0,
       merger_rate_amplitude_R₀,
       bns_amplitude_scalings,
       bns_hyperprior,
       bns_hyperprior_amplitude_marginalized

include("propagation.jl")
include("models/bns_madau_dickinson.jl")

end
