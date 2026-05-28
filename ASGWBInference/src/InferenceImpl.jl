module InferenceImpl

using ASGWB
using ASGWB:
             ImportanceSamplingProblem,
             PopulationModel,
             AbstractCosmology,
             LambdaCDM,
             W0CDM,
             W0WaCDM,
             cosmology_type,
             SUPPORTED_COSMOLOGIES,
             loglikelihood,
             evaluate_model_terms,
             canonical_hyperparameters,
             validate_hyperparameters,
             full_hyperparameters,
             full_hyperprior,
             OrderedUniformSourceMassPair,
             AlignedSpinChiSimple,
             redshift_prior,
             MadauDickinsonSourceFrame,
             BNS_LAMBDA_HIGH,
             validate_subset,
             normalized_ess,
             spectral_snr_squared,
             frequency_bin_width
import ASGWB: hyperparameters, hyperprior, single_event_prior
using AdvancedHMC
using Bijectors
using Distributions: MvNormal, ProductNamedTupleDistribution
using FiniteDiff
using ForwardDiff
using LinearAlgebra: Diagonal
using LogDensityProblems
using LogDensityProblemsAD
using Turing

include("populations.jl")
include("sampling.jl")
include("turing_model.jl")

export ASGWBLogDensity,
       unconstrained_initial_point,
       constrained_parameters,
       ad_logdensity,
       finite_difference_logdensity_and_gradient,
       sample_with_advancedhmc,
       build_turing_model,
       condition_turing_model,
       validate_hyperprior,
       BNSPopulationModel,
       POPULATION_REGISTRY,
       register_sample_hyperparameters

end
