using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing
using Turing: DynamicPPL

function validate_hyperprior(order::Tuple{Vararg{Symbol}}, prior::ProductNamedTupleDistribution)
    keys(prior.dists) == order || throw(
        ArgumentError("hyperprior must match order $(order), got $(keys(prior.dists))"),
    )
    return nothing
end

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

@model function sample_hyperparameters(order::Tuple{Vararg{Symbol}}, dists)
    values = map(order) do sym
        x ~ DynamicPPL.NamedDist(dists[sym], sym)
        x
    end
    return NamedTuple{order}(Tuple(values))
end

@model function astrosgwb_importance_turing_model(
        track::Bool,
        model,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        observation::ObservationContext,
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
)
    order = full_hyperparameters(model)
    Λ ~ to_submodel(sample_hyperparameters(order, prior.dists), false)
    Λc = canonical_hyperparameters(
        order,
        Λ;
        context = "sampled hyperparameters",
        eltype = nothing
    )

    rate, log_weights = merger_rate_and_log_weights(model, Λc, samples)
    weights = exp.(log_weights)
    Sh = spectral_density(fluxes, rate; weights = weights)

    observed_in_band ~ MvNormal(
        Sh[observation.in_band_mask],
        Diagonal(observation.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    m = observation.in_band_mask
    df = frequency_bin_width(observation.frequencies)
    obs_sec = year_to_second(observation.observation_time)
    snr_sq = spectral_snr_squared(
        Sh[m], observation.effective_psd[m], obs_sec, df)
    return (;
        number_of_sources = rate * obs_sec,
        effective_sample_size = normalized_ess(weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

function build_turing_model(
        model,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        fiducial_hyperparameters::NamedTuple,
        observation::ObservationContext,
        prior::ProductNamedTupleDistribution;
        track::Bool = false,
        observed::Union{Nothing, AbstractVector{<:Real}} = nothing
)
    order = full_hyperparameters(model)
    validate_hyperprior(order, prior)
    observed_data = if observed === nothing
        fiducial_spectral_density(model, fluxes, samples, fiducial_hyperparameters)
    else
        observed
    end
    return astrosgwb_importance_turing_model(
        track,
        model,
        fluxes,
        samples,
        observation,
        prior,
        observed_data[observation.in_band_mask]
    )
end
