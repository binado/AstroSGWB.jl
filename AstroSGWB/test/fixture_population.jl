# Test-only reference population implementing the PopulationModel contract, plus the
# canonical out-of-package "prepared model" that implements the cosmology-agnostic inference
# contract (`merger_rate_and_log_weights` + `full_hyperparameters`). The framework owns no
# concrete population or prepared-model types; callers define the concrete models used by
# their notebooks or scripts. This file is that example.
using AstroSGWB: PopulationModel, AbstractCosmology, AbstractPropagation,
                 CosmologyCache, GridQuery, DEFAULT_Z_GRID,
                 OrderedUniformSourceMassPair, AlignedSpinChiSimple,
                 redshift_prior, MadauDickinsonSourceFrame,
                 ObservationContext, FrequencyGrid, Detector, frequencies, in_band_mask,
                 build_observation_context,
                 cosmology, propagation, luminosity_distance,
                 component_logpdfs, logprobdiff, merger_rate,
                 importance_log_weights, with_redshift_interpolant,
                 canonical_hyperparameters
import AstroSGWB: hyperparameters, single_event_prior, merger_rate_and_log_weights,
                  full_hyperparameters
using Distributions: Uniform, product_distribution, ProductNamedTupleDistribution

struct ParityBNSPopulation <: PopulationModel end

hyperparameters(::ParityBNSPopulation) = (:γ, :κ, :zpeak)

function parity_population_hyperprior()
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

function single_event_prior(::ParityBNSPopulation, cache::CosmologyCache, Λ::NamedTuple)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), cache, Λ)
    spin = AlignedSpinChiSimple()
    return product_distribution((
        mass = OrderedUniformSourceMassPair(),
        redshift = z_d,
        χ₁ = spin,
        χ₂ = spin,
        Λ₁ = Uniform(0.0, 5000.0),
        Λ₂ = Uniform(0.0, 5000.0)
    ))
end

"""
    PreparedParityModel

Canonical out-of-package prepared model: the cosmology-specific half of what used to be a
`ModelContext`, fused onto the model the author owns. Carries the population, the fiducial
proposal caches (`proposal_prior`, per-component `proposal_log_prob`, `dl_fid_sq`), the
redshift grid + interpolant, and the `local_merger_rate`/`observation_time` that make
[`merger_rate_and_log_weights`](@ref) self-contained. The background cosmology family `C`
and GW propagation family `P` are *type parameters* (model-internal), so the package never
sees a cosmology token.
"""
struct PreparedParityModel{
    C <: AbstractCosmology,
    P <: AbstractPropagation,
    Pop <: PopulationModel,
    PR <: ProductNamedTupleDistribution,
    L <: NamedTuple
}
    pop::Pop
    redshift_grid::Vector{Float64}
    sample_interpolant::GridQuery
    proposal_prior::PR
    proposal_log_prob::L
    dl_fid_sq::Vector{Float64}
    local_merger_rate::Float64
    observation_time::Float64
end

"""
    prepare_parity_model(pop, samples, fiducials, C, P, grid, detectors, observation_time,
        local_merger_rate; z_grid)
        -> (; model, observation)

Assemble a [`PreparedParityModel`](@ref) and its [`ObservationContext`](@ref) from the
caller-owned population, catalog samples, fiducials, cosmology/propagation families `C`/`P`,
catalog [`FrequencyGrid`](@ref), and detector network. Mirrors the precompute the retired
`build_model_context` did, but partitions it: the cosmology-specific caches go on the
model, the detector/observation state goes in `observation`. The proposal caches are
rebuilt at the fiducial cosmology so stale on-disk values are never trusted.
"""
function prepare_parity_model(
        pop::PopulationModel,
        samples::NamedTuple,
        fiducials::NamedTuple,
        ::Type{C},
        ::Type{P},
        grid::FrequencyGrid,
        detectors::AbstractVector{<:Detector},
        observation_time::Real,
        local_merger_rate::Real;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    length(detectors) < 2 && throw(ArgumentError(
        "prepare_parity_model: at least two detectors are required to build effective_psd and sgwb_scale"))

    Λ_fid = fiducials
    z = samples.redshift

    all_freq = frequencies(grid)
    mask = in_band_mask(grid)
    det_vec = Vector{Detector}(collect(detectors))
    observation = build_observation_context(
        all_freq, det_vec, mask, Float64(observation_time))

    c_fid = cosmology(C, Λ_fid)
    # Per-sample squared EM luminosity distance at the fiducial cosmology. The (Ξ₀, Ξₙ)
    # propagation factor is applied live in the importance weights, not baked in here.
    dl_fid_sq = luminosity_distance.(z, c_fid) .^ 2

    redshift_grid = collect(Float64, z_grid)
    interp = GridQuery(z, redshift_grid)

    # Fiducial proposal prior and its per-component log-densities, computed with the same
    # interpolant the hot path uses, so the redshift log-ratio at Λ_fid is exactly zero.
    cache_fid = CosmologyCache(c_fid, redshift_grid)
    proposal_prior = single_event_prior(pop, cache_fid, Λ_fid)
    samples_interp = with_redshift_interpolant(samples, interp)
    proposal_log_prob = component_logpdfs(proposal_prior, samples_interp)

    model = PreparedParityModel{C, P, typeof(pop),
        typeof(proposal_prior), typeof(proposal_log_prob)}(
        pop,
        redshift_grid,
        interp,
        proposal_prior,
        proposal_log_prob,
        dl_fid_sq,
        Float64(local_merger_rate),
        Float64(observation_time)
    )
    return (; model = model, observation = observation)
end

"""
    full_hyperparameters(model::PreparedParityModel) -> NTuple{N,Symbol}

Flat HMC/Turing vector layout `(cosmo…, prop…, pop…)` for the prepared model, reusing the
`CBCDistributions` family-token helper now that `C`/`P` are model-internal.
"""
function full_hyperparameters(model::PreparedParityModel{C, P}) where {C, P}
    return full_hyperparameters(C, P, model.pop)
end

"""
    merger_rate_and_log_weights(model::PreparedParityModel, Λ, samples) -> (rate, log_weights)

The fused cosmology-specific hot path: rebuild the redshift `CosmologyCache` and
`single_event_prior` at `Λ` (shared between the rate and the weights), form the prior
log-ratio against the fiducial proposal caches, and return `(rate, log_weights)`.
"""
function merger_rate_and_log_weights(
        model::PreparedParityModel{C, P}, Λ::NamedTuple, samples
) where {C, P}
    Λc = canonical_hyperparameters(
        full_hyperparameters(model), Λ; context = "joint hyperparameters", eltype = nothing)
    cache = CosmologyCache(cosmology(C, Λc), model.redshift_grid)
    prior = single_event_prior(model.pop, cache, Λc)
    prop = propagation(P, Λc)
    samples_interp = with_redshift_interpolant(samples, model.sample_interpolant)
    log_ratio = logprobdiff(
        model.pop, prior, model.proposal_prior, model.proposal_log_prob, samples_interp)
    log_weights = importance_log_weights(
        log_ratio, model.dl_fid_sq, samples.redshift,
        model.sample_interpolant, cache, prop)
    rate = merger_rate(prior, model.local_merger_rate, model.observation_time)
    return (rate, log_weights)
end
