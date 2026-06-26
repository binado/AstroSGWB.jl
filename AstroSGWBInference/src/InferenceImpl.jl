module InferenceImpl

using AstroSGWB
using AstroSGWB:
                 ObservationContext,
                 PopulationModel,
                 merger_rate_and_log_weights,
                 spectral_density,
                 fiducial_spectral_density,
                 canonical_hyperparameters,
                 full_hyperparameters,
                 validate_subset,
                 normalized_ess,
                 spectral_snr_squared,
                 frequency_bin_width,
                 year_to_second
using Distributions: MvNormal, ProductNamedTupleDistribution, logpdf
using LinearAlgebra: Diagonal
using Turing

include("likelihood.jl")
include("turing_model.jl")

export build_turing_model,
       condition_turing_model,
       loglikelihood,
       logposterior,
       validate_hyperprior

end
