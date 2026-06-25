"""
    AstroSGWB

Astrophysical stochastic gravitational-wave background modeling: importance
sampling, redshift grids, and likelihoods. Turing model construction lives in the
`AstroSGWBInference` package (see the `AstroSGWBInference/` directory in the repository).

The primary inference artifact is **`catalog.h5`** ([`WaveformCatalogFile`](@ref)):
per-sample intrinsic parameters with precomputed luminosity distances, and a
`(nfreq, nsamples)` per-sample flux matrix `|h_+|² + |h_×|²` (before the
fiducial `(D_L/D_gw)²` factor).

Callers define their population model, fiducial hyperparameters, and catalog sample
adapter in Julia, then construct a pure [`ImportanceSamplingProblem`](@ref). Derived
`Λ`-independent caches (proposal log-prob, redshift interpolant, detector PSDs) are built
into a [`ModelContext`](@ref) by [`build_model_context`](@ref).

Inference state is a flat hyperparameter `NamedTuple` validated against the
[`PopulationModel`](@ref) contract; the cosmology family `C` is threaded through atomic
calls rather than stored on the problem.
"""
module AstroSGWB

using CBCDistributions
using Cosmology
import CBCDistributions: single_event_prior
import Cosmology: cosmology, cosmology_type, gravitational_wave_distance,
                  gw_em_distance_ratio, hyperparameters,
                  propagation, propagation_type

include("types.jl")
include("models/base.jl")
include("catalog/grid.jl")
include("catalog/catalog.jl")
include("catalog/io.jl")
include("inference_types.jl")
include("detector/psd.jl")
include("detector/detector.jl")
include("detector/overlap.jl")
include("detector/effective_psd.jl")
include("detector/observation.jl")
include("importance.jl")
include("spectral_density.jl")
include("snr.jl")
include("diagnostics.jl")
include("context.jl")
include("posterior.jl")

# Types
export ImportanceSamplingProblem,
       ModelContext,
       build_model_context,
       ObservationContext,
       PopulationModel,
       hyperparameters,
       single_event_prior,
       full_hyperparameters,
       canonical_hyperparameters,
       validate_hyperparameters,
       validate_subset,
       stack_source_masses,
       validate_samples,
       CATALOG_SOURCE_TYPE_ATTR,
       CATALOG_SOURCE_TYPE_BNS,
       CumulativeIntegral1D,
       GridQuery,
       interpolate,
       cdf,
       normalizer,
       RedshiftPrior,
       redshift

# Catalog I/O
export FrequencyGrid,
       frequencies,
       in_band_mask,
       WaveformCatalog,
       WaveformCatalogMetadata,
       WaveformCatalogFile,
       load_catalog,
       save_catalog

# Detector network (ORF / PSD effective strain PSD; used by `build_model_context`)
export Detector,
       PowerSpectralDensity,
       default_detector_data_dir,
       overlap_reduction_function,
       pairwise_overlap_reduction_function,
       effective_psd,
       gaussian_bin_scale,
       gaussian_bin_variance,
       frequency_bin_width,
       build_observation_context

# Cosmology
export E,
       AbstractCosmology,
       LambdaCDM,
       W0CDM,
       W0WaCDM,
       AbstractPropagation,
       GR,
       ModifiedPropagation,
       dark_energy_eos,
       de_density_ratio,
       cosmology,
       cosmology_type,
       SUPPORTED_COSMOLOGIES,
       propagation,
       propagation_hyperparameters,
       propagation_type,
       propagation_config_name,
       SUPPORTED_PROPAGATIONS,
       CosmologyCache,
       comoving_distance,
       luminosity_distance,
       luminosity_distance_at_sample,
       differential_comoving_volume,
       gravitational_wave_distance,
       gw_em_distance_ratio,
       hubble_constant_si,
       H0,
       Ωm

# Redshift & population
export madau_dickinson_source_frame_distribution,
       detector_frame_merger_rate_density,
       build_redshift_prior,
       redshift_prior,
       MadauDickinsonSourceFrame,
       source_frame_distribution,
       DEFAULT_Z_GRID,
       redshift_log_prob,
       redshift_logpdf_eltype,
       redshift_integral,
       expected_number_of_events,
       merger_rate_per_sec

# Priors
export OrderedUniformSourceMassPair,
       AlignedSpinChiSimple,
       RedshiftInterpolatedDistribution,
       SampleField,
       sample_values,
       sample_meta,
       add_logpdfvec!,
       batched_logpdf,
       component_logpdfs,
       validate_samples,
       logprobdiff,
       logprobdiff!

# Importance sampling
export compute_importance_weights,
       spectral_density,
       inner_product,
       spectral_snr_squared,
       spectral_snr,
       Ωgw

# Diagnostics
export normalized_ess

# Posterior
export loglikelihood,
       merger_rate,
       fiducial_hyperparameters,
       fiducial_spectral_density

# Time conversions
export JULIAN_YEAR_SEC,
       year_to_second,
       second_to_year

end
