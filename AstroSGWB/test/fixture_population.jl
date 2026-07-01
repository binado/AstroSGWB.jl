# Test-only reference population implementing the PopulationModel contract, plus the
# canonical out-of-package "prepared model" that implements the cosmology-agnostic inference
# contract (`merger_rate_and_log_weights` + `hyperparameters`). The framework owns no
# concrete population or prepared-model types; callers define the concrete models used by
# their notebooks or scripts. This file is that example, mirroring the slim
# `BNSImportanceModel` in notebooks/mcmc.jl.
using AstroSGWB: CosmologyCache, GridQuery, DEFAULT_Z_GRID,
                 OrderedUniformSourceMassPair, AlignedSpinChiSimple,
                 redshift_prior, MadauDickinsonSourceFrame,
                 ObservationContext, FrequencyGrid, Detector, frequencies, in_band_mask,
                 build_observation_context,
                 cosmology, propagation, differential_comoving_volume,
                 redshift_density, source_frame_distribution, normalizer,
                 redshift_logpdf_eltype, interpolate,
                 luminosity_distance_at_sample, gw_em_distance_ratio,
                 integrated_merger_rate
using CBCDistributions: PopulationModel, full_hyperparameters, single_event_prior
import Cosmology
using Cosmology: AbstractCosmology, AbstractPropagation
import CBCDistributions: single_event_prior
using Distributions: Uniform, product_distribution

struct ParityBNSPopulation <: PopulationModel end

Cosmology.hyperparameters(::ParityBNSPopulation) = (:γ, :κ, :zpeak)

function parity_population_hyperprior()
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

# Population sampler contract (used by the population-injection workflow): the per-event
# intrinsic prior as a product distribution. The slim prepared model below does not use it
# (the Λ-independent components cancel exactly), but it documents the sampler interface.
function single_event_prior(::ParityBNSPopulation, cache::CosmologyCache, Λ::NamedTuple)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), Λ, cache)
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

Canonical out-of-package prepared model: a single concrete struct implementing the
two-method inference contract, mirroring `BNSImportanceModel`. The six-component prior
collapses to a redshift log-ratio (mass/spin/tidal are Λ-independent and cancel exactly)
plus a distance/propagation factor, so `merger_rate_and_log_weights` inlines the redshift +
importance-weight math. The background cosmology family `C` and GW propagation family `P`
are *type parameters* (model-internal), so the package never sees a cosmology token.
"""
struct PreparedParityModel{C, P, Pop}
    pop::Pop
    z_grid::Vector{Float64}
    query::GridQuery
    proposal_log_pdf::Vector{Float64}
    local_merger_rate::Float64
    observation_time::Float64
end

"""
    prepare_parity_model(pop, samples, fiducials, C, P, grid, detectors, observation_time,
        local_merger_rate; z_grid)
        -> (; model, observation)

Assemble a [`PreparedParityModel`](@ref) and its [`ObservationContext`](@ref). The fiducial
proposal redshift log-density is evaluated once; the `samples` NamedTuple is the single
source of truth for the fiducial EM `luminosity_distance`.
"""
function prepare_parity_model(
        pop,
        samples::NamedTuple,
        fiducials::NamedTuple,
        ::Type{C},
        ::Type{P},
        grid::FrequencyGrid,
        detectors::AbstractVector{<:Detector},
        observation_time::Real,
        local_merger_rate::Real;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C, P}
    length(detectors) < 2 && throw(ArgumentError(
        "prepare_parity_model: at least two detectors are required to build effective_psd and sgwb_scale"))

    z = samples.redshift
    observation = build_observation_context(
        frequencies(grid), Vector{Detector}(collect(detectors)),
        in_band_mask(grid), Float64(observation_time))
    zg = collect(Float64, z_grid)
    query = GridQuery(z, zg)

    cache_fid = CosmologyCache(cosmology(C, fiducials), zg)
    dvc_fid = differential_comoving_volume.(zg, Ref(cache_fid))
    dN_dz_fid = redshift_density(zg, dvc_fid, MadauDickinsonSourceFrame(), fiducials)
    norm_fid = normalizer(dN_dz_fid)
    proposal_log_pdf = [
        log(interpolate(dN_dz_fid, query, i)) - log(norm_fid)
        for i in eachindex(z)
    ]

    model = PreparedParityModel{C, P, typeof(pop)}(
        pop, zg, query, proposal_log_pdf,
        Float64(local_merger_rate), Float64(observation_time))
    return (; model = model, observation = observation)
end

if @isdefined AstroSGWBInference
    import AstroSGWBInference: merger_rate_and_log_weights

    function AstroSGWBInference.hyperparameters(
            model::PreparedParityModel{C, P}) where {C, P}
        return full_hyperparameters(C, P, model.pop)
    end

    function merger_rate_and_log_weights(
            m::PreparedParityModel{C, P}, Λ::NamedTuple, samples
    ) where {C, P}
        z = samples.redshift
        d_l_fid = samples.luminosity_distance
        cache = CosmologyCache(cosmology(C, Λ), m.z_grid)
        prop = propagation(P, Λ)

        dvc_grid = differential_comoving_volume.(m.z_grid, Ref(cache))
        dN_dz = redshift_density(m.z_grid, dvc_grid, MadauDickinsonSourceFrame(), Λ)
        norm = normalizer(dN_dz)

        T = promote_type(redshift_logpdf_eltype(dN_dz),
            typeof(gw_em_distance_ratio(zero(eltype(z)), prop)))
        log_weights = Vector{T}(undef, length(z))
        @inbounds for i in eachindex(z)
            log_p_target = log(interpolate(dN_dz, m.query, i)) - log(norm)
            d_l_θ = luminosity_distance_at_sample(cache, m.query, z, i)
            Ξ_θ = gw_em_distance_ratio(z[i], prop)
            log_weights[i] = (log_p_target - m.proposal_log_pdf[i]) +
                             2 * log(d_l_fid[i]) - 2 * log(d_l_θ) - 2 * log(Ξ_θ)
        end

        rate = integrated_merger_rate(dN_dz, m.local_merger_rate)
        return (rate, log_weights)
    end
end
