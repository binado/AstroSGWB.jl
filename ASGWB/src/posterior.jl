"""
    merger_rate(problem, C, Λ, ctx::ModelContext) -> Float64

Detector-frame merger rate in events/sec at hyperparameters `Λ` (cosmology family `C`),
reading the local rate and observation times from `ctx`.
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
        ctx.observation.observation_time_yr,
        ctx.observation.observation_time_sec
    )
end

"""
    merger_rate(problem, C, Λ, local_merger_rate, observation_time_yr, observation_time_sec) -> Float64

Detector-frame merger rate in events/sec from explicit observation arguments (no
`ModelContext` required). Doubles as the ctx-free oracle for the cached method above.
"""
function merger_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        local_merger_rate::Real,
        observation_time_yr::Real,
        observation_time_sec::Real
) where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    return merger_rate_per_sec(
        prior.dists.redshift.prior,
        local_merger_rate,
        observation_time_yr,
        observation_time_sec
    )
end

"""
    weights_and_rate(problem, C, Λ, ctx::ModelContext) -> (weights, rate)

Importance weights and detector-frame merger rate (events/sec) at `Λ`, built from a single
`single_event_prior(...)` construction. The standalone `compute_importance_weights` and
`merger_rate` helpers each rebuild the cosmology and prior (including a redshift
`CosmologyCache`), so the inner forward model uses this to share that work across both.
"""
function weights_and_rate(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        ctx::ModelContext
) where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    weights = _importance_weights(problem, c, prior, ctx)
    rate = merger_rate_per_sec(
        prior.dists.redshift.prior,
        ctx.local_merger_rate,
        ctx.observation.observation_time_yr,
        ctx.observation.observation_time_sec
    )
    return weights, rate
end

"""
    loglikelihood(Λ, problem, C, ctx; observed = ctx.fiducial_spectral_density)

Gaussian in-band log-likelihood of the SGWB spectral density at `Λ`. Inlines the
`weights → rate → Sₕ` sequence using the cached atomics and `ctx` masks/scales.
"""
function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext;
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    weights, rate = weights_and_rate(problem, C, Λ, ctx)
    Sh = spectral_density(ctx.cached_flux_over_dgw2, rate; weights = weights)

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
    fiducial_redshift_integral(C, population, Λ) -> Float64

Redshift-integrated detector-frame merger-rate density at hyperparameters `Λ`.
"""
function fiducial_redshift_integral(
        ::Type{C},
        population::M,
        Λ::NamedTuple
) where {C <: AbstractCosmology, M <: PopulationModel}
    c = cosmology(C, Λ)
    prior = single_event_prior(population, c, Λ)
    return Float64(redshift_integral(prior.dists.redshift.prior))
end

"""
    fiducial_redshift_integral(problem, C) -> Float64

Redshift integral at `problem.fiducial_hyperparameters` for cosmology family `C`.
"""
function fiducial_redshift_integral(
        problem::ImportanceSamplingProblem,
        ::Type{C}
) where {C <: AbstractCosmology}
    return fiducial_redshift_integral(C, problem.population_model, problem.fiducial_hyperparameters)
end
