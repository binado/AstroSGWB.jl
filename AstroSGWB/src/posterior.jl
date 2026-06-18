"""
    merger_rate(problem, C, Λ, ctx::ModelContext) -> Float64

Detector-frame merger rate in events/sec at hyperparameters `Λ` (cosmology family `C`),
reading the local rate and observation time from `ctx`.
"""
function merger_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        ctx::ModelContext
) where {C <: AbstractCosmology}
    return merger_rate(
        problem,
        C,
        Λ,
        ctx.local_merger_rate,
        ctx.observation.observation_time
    )
end

"""
    merger_rate(problem, C, Λ, local_merger_rate, observation_time) -> Float64

Detector-frame merger rate in events/sec from explicit observation arguments (no
`ModelContext` required). Doubles as the ctx-free oracle for the cached method above.

`observation_time` is the observation duration in years (Julian year).
"""
function merger_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        local_merger_rate::Real,
        observation_time::Real
) where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    return merger_rate(prior, local_merger_rate, observation_time)
end

"""
    merger_rate(prior, local_merger_rate, observation_time) -> Float64

Detector-frame merger rate in events/sec from an already-built `single_event_prior`. The
hot path constructs the prior once and shares it with `compute_importance_weights`, so this
is the form the forward model calls directly (no cosmology/prior rebuild). The redshift
`.dists.redshift.prior` reach-through stays here rather than at every call site.

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
    loglikelihood(Λ, problem, C, ctx, observed)

Gaussian in-band log-likelihood of the SGWB spectral density at `Λ`. Inlines the
`weights → rate → Sₕ` sequence using the cached atomics and `ctx` masks/scales. Builds the
redshift `CosmologyCache` once and shares it between `single_event_prior` and
`compute_importance_weights`, so its cumulative cosmology integral is not rebuilt per
ForwardDiff gradient step.

`observed` is the full-length strain spectral density vector (one entry per frequency bin
in `ctx.observation.frequencies`).
"""
function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        observed::AbstractVector{<:Real}
) where {C <: AbstractCosmology}
    cache = CosmologyCache(cosmology(C, Λ), ctx.redshift_grid)
    prior = single_event_prior(problem.population_model, cache, Λ)
    weights = compute_importance_weights(problem, cache, prior, ctx)
    rate = merger_rate(
        prior,
        ctx.local_merger_rate,
        ctx.observation.observation_time
    )
    Sh = spectral_density(problem.fluxes, rate; weights = weights)

    obs = ctx.observation
    mask = obs.in_band_mask
    σ = obs.sgwb_scale_in_band
    residual = observed[mask] .- Sh[mask]
    return -0.5 * sum((residual ./ σ) .^ 2 .+ log.(2π .* (σ .^ 2)))
end

function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end

"""
    fiducial_spectral_density(problem, C, ctx) -> Vector

Synthesize the default observed strain spectral density from catalog fluxes at the
problem's fiducial hyperparameters, using the same importance-weighted forward model
as [`loglikelihood`](@ref). Callers that omit `observed` in [`build_turing_model`](@ref)
should use this spectrum so modified-propagation factors `Ξ(z)` are applied consistently.
"""
function fiducial_spectral_density(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext
) where {C <: AbstractCosmology}
    Λ_fid = problem.fiducial_hyperparameters
    weights = compute_importance_weights(problem, C, Λ_fid, ctx)
    rate = merger_rate(problem, C, Λ_fid, ctx)
    return spectral_density(problem.fluxes, rate; weights = weights)
end
