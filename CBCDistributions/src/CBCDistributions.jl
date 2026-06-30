module CBCDistributions

import Cosmology
import Cosmology: AbstractCosmology, AbstractPropagation, CosmologyCache,
                  differential_comoving_volume,
                  luminosity_distance_at_sample,
                  hyperparameters, propagation_hyperparameters
import CumulativeIntegrals: CumulativeIntegral1D, GridQuery, cdf, interpolate, normalizer

export PopulationModel, hyperparameters, single_event_prior,
       full_hyperparameters,
       canonical_hyperparameters, validate_hyperparameters
export GridQuery, luminosity_distance_at_sample
export DefaultBBHPrimaryMass, DefaultBBHMassPair, planck_taper
export JULIAN_YEAR_SEC, year_to_second, second_to_year

include("utils.jl")
include("mass/uniform.jl")
include("mass/broken_power_law_plus_two_peaks.jl")
include("spins/aligned.jl")
include("distance/redshift.jl")
include("distance/source_frame/madau_dickinson.jl")
include("distance/redshift_distribution.jl")
include("distance/redshift_prior.jl")
include("population_model.jl")

end # module
