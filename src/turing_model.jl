using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing

"""
    condition_turing_model(model, theta0, sample_only::Union{Nothing,Tuple}) -> model

If `sample_only === nothing`, return `model` unchanged (all hyperparameters are sampled).

Otherwise `sample_only` lists the subset of [`DEFAULT_PARAMETER_ORDER`](@ref) that remain
stochastic; all other hyperparameters are **fixed** to their values in `as_flat_constrained(theta0)`
using Turing’s conditioning operator `|` (see
[Turing docs: conditioning on data](https://turinglang.org/docs/core-functionality/#conditioning-on-data)).
"""
function condition_turing_model(
    model,
    theta0::HyperParameters,
    sample_only::Union{Nothing,Tuple{Vararg{Symbol}}},
)
    sample_only === nothing && return model
    as = as_flat_constrained(theta0)
    fixed = Tuple(s for s in DEFAULT_PARAMETER_ORDER if s ∉ sample_only)
    isempty(fixed) && return model
    return model | (; (s => as[s] for s in fixed)...)
end

function _validate_sample_only(sample_only::Union{Nothing,Tuple{Vararg{Symbol}}})
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
            "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
        ),
    )
    for s in sample_only
        s in DEFAULT_PARAMETER_ORDER || throw(
            ArgumentError(
                "sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)",
            ),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

function _turing_initial_params(
    theta0::HyperParameters,
    sample_only::Union{Nothing,Tuple{Vararg{Symbol}}},
)
    as = as_flat_constrained(theta0)
    sample_only === nothing && return InitFromParams(as)
    return InitFromParams((; (s => as[s] for s in sample_only)...))
end

@model function asgwb_importance_turing_model(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors,
    observed_in_band::AbstractVector{<:Real},
)
    H0 ~ priors.H0
    Omega_m ~ priors.Omega_m
    chi0 ~ priors.chi0
    chin ~ priors.chin
    gamma ~ priors.gamma
    kappa ~ priors.kappa
    z_peak ~ priors.z_peak

    h = HyperParameters(
        CosmologicalParameters(H0, Omega_m),
        ModifiedPropagationParameters(chi0, chin),
        MadauDickinsonParameters(gamma, kappa, z_peak),
    )

    bundle = build_redshift_grid_bundle(h, problem.redshift_prior_spec)
    iw = compute_importance_weights(problem, h, bundle)
    rate = merger_rate_per_sec(
        bundle,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec,
    )
    sd = spectral_density(
        problem.proposal.cached_flux_over_dgw2, rate; weights=iw.weights,
    )
    sd_in_band = sd[problem.observation.in_band_mask]

    observed_in_band ~ MvNormal(
        sd_in_band,
        Diagonal(problem.observation.sgwb_scale_in_band .^ 2),
    )

    return (;
        number_of_sources=rate * problem.observation.observation_time_sec,
        spectral_density=sd,
        effective_sample_size=normalized_ess(iw.weights),
    )
end

function build_turing_model(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    return asgwb_importance_turing_model(
        problem,
        priors,
        observed_spectral_density[problem.observation.in_band_mask],
    )
end

"""
    sample_with_turing(problem, priors, theta0; kwargs...) -> (chain, model)

NUTS sampling for the importance likelihood model. Keyword `sample_only` lists which of
`DEFAULT_PARAMETER_ORDER` remain stochastic; all others are fixed to the corresponding entries
in `as_flat_constrained(theta0)` using Turing’s `|` operator (see
[`condition_turing_model`](@ref)).
"""
function sample_with_turing(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors,
    theta0::HyperParameters;
    n_adapts::Int=25,
    n_samples::Int=25,
    target_acceptance::Float64=0.8,
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
    sample_only::Union{Nothing,Tuple{Vararg{Symbol}}}=nothing,
)
    _validate_sample_only(sample_only)
    model = build_turing_model(
        problem,
        priors;
        observed_spectral_density=observed_spectral_density,
    )
    conditioned = condition_turing_model(model, theta0, sample_only)
    chain = sample(
        conditioned,
        Turing.NUTS(n_adapts, target_acceptance),
        n_samples;
        progress=false,
        initial_params=_turing_initial_params(theta0, sample_only),
    )
    return chain, conditioned
end
