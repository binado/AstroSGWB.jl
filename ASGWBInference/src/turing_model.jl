using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing

function condition_turing_model(
        turing_model,
        theta0::NamedTuple,
        prior::ProductNamedTupleDistribution,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}};
        order::Tuple{Vararg{Symbol}}
)
    validate_hyperprior(order, prior)
    ordered_theta0 = canonical_hyperparameters(order, theta0; context = "initial hyperparameters")
    sample_only === nothing && return turing_model
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, order)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return turing_model
    return turing_model | (; (s => ordered_theta0[s] for s in fixed)...)
end

# Generate a `sample_hyperparameters` submodel for each (cosmology, BNSPopulationModel)
# combination. `full_hyperparameters` is called at module-load time so Turing sees
# literal symbol names in the generated tilde-sites.
for C in SUPPORTED_COSMOLOGIES
    flds = full_hyperparameters(C, BNSPopulationModel())
    @eval begin
        @model function sample_hyperparameters(c::Val{$C}, pop::BNSPopulationModel, d)
            $([:($f ~ d.$f) for f in flds]...)
            return (; $(flds...))
        end
    end
end

@model function asgwb_importance_turing_model(
        track::Bool,
        problem::ImportanceSamplingProblem{C, M},
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
) where {C, M}
    Λ ~ to_submodel(sample_hyperparameters(Val(C), problem.population, prior.dists), false)
    order = full_hyperparameters(C, problem.population)
    Λc = canonical_hyperparameters(
        order,
        Λ;
        context = "sampled hyperparameters",
        eltype = nothing
    )
    terms = evaluate_model_terms(Λc, problem)

    observed_in_band ~ MvNormal(
        terms.spectral_density_in_band,
        Diagonal(problem.observation.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    obs = problem.observation
    m = obs.in_band_mask
    df = frequency_bin_width(obs.frequencies)
    snr_sq = spectral_snr_squared(
        terms.spectral_density[m], obs.effective_psd[m], obs.observation_time_sec, df
    )
    return (;
        number_of_sources = terms.expected_number_of_sources,
        effective_sample_size = normalized_ess(terms.weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

function build_turing_model(
        problem::ImportanceSamplingProblem{C, M},
        prior::ProductNamedTupleDistribution;
        track::Bool = false,
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
) where {C, M}
    order = full_hyperparameters(C, problem.population)
    validate_hyperprior(order, prior)
    return asgwb_importance_turing_model(
        track,
        problem,
        prior,
        observed_spectral_density[problem.observation.in_band_mask]
    )
end
