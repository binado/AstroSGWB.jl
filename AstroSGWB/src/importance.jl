using LinearAlgebra

# Per-sample importance weight as a product of three physically independent factors:
# the population prior ratio `exp(log_ratio)`, the FLRW background distance ratio
# `(D_L,fid/D_L,θ)²`, and the modified-propagation factor `(1/Ξ_θ)²`. The raw catalog flux
# carries `1/D_L,fid²`, so multiplying by `dl_fid_sq / (D_L,θ² · Ξ_θ²)` recovers the
# physically correct `1/D_gw,θ²` dilution.
@inline function _importance_weight_at_sample(
        log_ratio::AbstractVector,
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        interp::GridQuery,
        cosmology_cache::CosmologyCache,
        prop::AbstractPropagation,
        sample_index::Integer
)
    d_l = luminosity_distance_at_sample(cosmology_cache, interp, z, sample_index)
    Ξ_theta = gw_em_distance_ratio(z[sample_index], prop)
    return exp(log_ratio[sample_index]) * dl_fid_sq[sample_index] / (d_l^2 * Ξ_theta^2)
end

# Shared kernel for both the naive and cached `compute_importance_weights` methods. Inputs
# are explicit arrays so the only difference between the two backends is where the fiducial
# caches come from (recomputed vs read from a `ModelContext`) and how `log_ratio` (per-sample
# `log p_target − log p_proposal`) was produced (`logprobdiff` vs full two-sided
# `batched_logpdf`). Built with `map` over the index range: this keeps the result type stable
# (a properly-typed empty vector for n == 0, rather than a `Union` with `Float64[]`) so the
# AD/`Dual` likelihood path stays inferrable.
function _importance_weights_core(
        log_ratio::AbstractVector,
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        interp::GridQuery,
        cosmology_cache::CosmologyCache,
        prop::AbstractPropagation
)
    length(z) == length(log_ratio) ||
        throw(ArgumentError("population prior logpdf length must match proposal sample count"))
    return map(eachindex(z)) do i
        _importance_weight_at_sample(
            log_ratio, dl_fid_sq, z, interp, cosmology_cache, prop, i)
    end
end

"""
    compute_importance_weights(problem, C, P, Λ, ctx::ModelContext) -> Vector

Per-sample importance weights at hyperparameters `Λ` (cosmology family `C`, propagation
family `P`), reading the fiducial proposal caches from `ctx`. This is the hot path used by
the likelihood and the generative model.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ::Type{P},
        Λ::NamedTuple,
        ctx::ModelContext
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    cache = CosmologyCache(cosmology(C, Λ), ctx.redshift_grid)
    prior = single_event_prior(problem.population_model, cache, Λ)
    prop = propagation(P, Λ)
    return compute_importance_weights(problem, cache, prop, prior, ctx)
end

"""
    compute_importance_weights(problem, cache::CosmologyCache, prop, prior, ctx::ModelContext) -> Vector

Importance weights from a prebuilt `cache`, propagation `prop`, and single-event `prior`. The
forward model builds the cache once, shares it with `single_event_prior` (which uses it for the
redshift prior), and passes it here for the per-sample luminosity distance — so the cumulative
cosmology integral is computed once per evaluation rather than rebuilt in both places. This
is the bare form the forward model calls directly.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        cache::CosmologyCache,
        prop::AbstractPropagation,
        prior,
        ctx::ModelContext
)
    # `logprobdiff` skips components whose target distribution is egal to the fiducial
    # proposal's (Λ-independent factors cancel exactly), and the redshift component reuses
    # the precomputed per-sample grid locations to skip the grid search every gradient
    # evaluation.
    samples = _with_redshift_interpolant(problem.samples, ctx.sample_interpolant)
    log_ratio = logprobdiff(
        problem.population_model,
        prior,
        ctx.proposal_prior,
        ctx.proposal_log_prob,
        samples
    )
    return _importance_weights_core(
        log_ratio,
        ctx.dl_fid_sq,
        problem.samples.redshift,
        ctx.sample_interpolant,
        cache,
        prop
    )
end

"""
    compute_importance_weights(problem, C, P, Λ) -> Vector

Naive importance weights: recompute the fiducial proposal caches (`proposal_log_prob`,
`dl_fid_sq`, redshift interpolant) from scratch at `problem.fiducial_hyperparameters`,
then weight at `Λ` (cosmology family `C`, propagation family `P`). Slower than the `ctx`
method but free of any precomputed state — it deliberately uses the full two-sided
`batched_logpdf` rather than `logprobdiff`, so it serves as the correctness oracle for the
cached path (including its component skipping).
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ::Type{P},
        Λ::NamedTuple
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    Λ_fid = problem.fiducial_hyperparameters
    z = problem.samples.redshift
    c_fid = cosmology(C, Λ_fid)
    prior_fid = single_event_prior(problem.population_model, c_fid, Λ_fid)
    proposal_log_prob = batched_logpdf(prior_fid, problem.samples)
    dl_fid_sq = luminosity_distance.(z, c_fid) .^ 2
    redshift_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = GridQuery(z, redshift_grid)

    cosmology_cache = CosmologyCache(cosmology(C, Λ), redshift_grid)
    prior = single_event_prior(problem.population_model, cosmology_cache, Λ)
    prop = propagation(P, Λ)
    log_ratio = batched_logpdf(prior, problem.samples) .- proposal_log_prob
    return _importance_weights_core(
        log_ratio,
        dl_fid_sq,
        z,
        interp,
        cosmology_cache,
        prop
    )
end
