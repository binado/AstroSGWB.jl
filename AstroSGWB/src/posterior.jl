"""
    merger_rate(prior, local_merger_rate, observation_time) -> Float64

Detector-frame merger rate in events/sec from an already-built `single_event_prior`. The
model's [`merger_rate_and_log_weights`](@ref) joint constructs the prior once and shares it
with the importance-weight kernel, so this is the form called directly (no cosmology/prior
rebuild). The redshift `.dists.redshift.prior` reach-through stays here rather than at every
call site.

`observation_time` is the observation duration in years (Julian year).
"""
function merger_rate(
        prior::ProductNamedTupleDistribution,
        local_merger_rate::Real,
        observation_time::Real
)
    return merger_rate_per_sec(
        prior.dists.redshift.prior,
        local_merger_rate,
        observation_time
    )
end

"""
    loglikelihood(Λ, model, problem, observation::ObservationContext, observed)

Gaussian in-band log-likelihood of the SGWB spectral density at `Λ`. Delegates the
cosmology-specific `rate`/`log_weights` to the model's [`merger_rate_and_log_weights`](@ref)
joint, exponentiates the weights, contracts the raw fluxes into `Sₕ`, and scores the in-band
residual against the `observation` masks/scales.

`observed` is the full-length strain spectral density vector (one entry per frequency bin
in `observation.frequencies`).
"""
function loglikelihood(
        Λ::NamedTuple,
        model,
        problem::ImportanceSamplingProblem,
        observation::ObservationContext,
        observed::AbstractVector{<:Real}
)
    rate, log_weights = merger_rate_and_log_weights(model, Λ, problem.samples)
    weights = exp.(log_weights)
    Sh = spectral_density(problem.fluxes, rate; weights = weights)

    mask = observation.in_band_mask
    σ = observation.sgwb_scale_in_band
    residual = observed[mask] .- Sh[mask]
    return -0.5 * sum((residual ./ σ) .^ 2 .+ log.(2π .* (σ .^ 2)))
end

function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end

"""
    fiducial_spectral_density(model, problem) -> Vector

Synthesize the default observed strain spectral density from catalog fluxes at the
problem's fiducial hyperparameters, using the same model-dispatched forward model as
[`loglikelihood`](@ref). Callers that omit `observed` in [`build_turing_model`](@ref) should
use this spectrum so modified-propagation factors `Ξ(z)` are applied consistently.
"""
function fiducial_spectral_density(model, problem::ImportanceSamplingProblem)
    rate,
    log_weights = merger_rate_and_log_weights(
        model, problem.fiducial_hyperparameters, problem.samples)
    return spectral_density(problem.fluxes, rate; weights = exp.(log_weights))
end
