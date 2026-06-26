# Headless, config-driven NUTS runner for the AstroSGWB importance-sampling model.
#
# This mirrors the sampling cells of notebooks/mcmc.jl but takes run-specific
# settings (catalog, detectors, fiducials, sampler, etc.) from a TOML config,
# parsed and validated via AstroSGWBInference.MCMCConfig. Hyperprior bounds, the
# cosmology family, and the population model are fixed here, matching the notebook.
#
# Run from the repository root, for example:
#   julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/example.toml

module AstroSGWBRunMCMC

const _REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

using AstroSGWB
using AstroSGWB:
                 canonical_hyperparameters,
                 load_catalog,
                 AbstractCosmology,
                 AbstractPropagation,
                 PopulationModel,
                 ObservationContext,
                 CosmologyCache,
                 GridQuery,
                 DEFAULT_Z_GRID,
                 FrequencyGrid,
                 frequencies,
                 in_band_mask,
                 build_observation_context,
                 cosmology,
                 propagation,
                 luminosity_distance,
                 component_logpdfs,
                 logprobdiff,
                 merger_rate,
                 importance_log_weights,
                 with_redshift_interpolant,
                 ModifiedPropagation,
                 GR,
                 W0CDM,
                 Detector,
                 OrderedUniformSourceMassPair,
                 AlignedSpinChiSimple,
                 redshift_prior,
                 MadauDickinsonSourceFrame,
                 stack_source_masses
import AstroSGWB: hyperparameters, single_event_prior,
                  merger_rate_and_log_weights, full_hyperparameters
using AstroSGWBInference:
                          build_turing_model,
                          condition_turing_model,
                          atomic_save_chain,
                          MCMCConfig,
                          load_config,
                          save_config,
                          validate_fiducials
using ADTypes: AutoForwardDiff
using AdvancedHMC: DenseEuclideanMetric
using Distributions: Uniform, product_distribution
using FlexiChains: VNChain
using Turing
using Random
using Logging
using LinearAlgebra: BLAS
using Dates: now, format

# --------------------------------------------------------------------------
# Population model (kept inline, matching notebooks/mcmc.jl)
# --------------------------------------------------------------------------

struct BNSPopulationModel <: PopulationModel end

hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

function single_event_prior(::BNSPopulationModel, cache::CosmologyCache, Λ::NamedTuple)
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

# --------------------------------------------------------------------------
# Prepared model (out-of-package assembly of the cosmology-agnostic contract)
# --------------------------------------------------------------------------

struct BNSPreparedModel{
    C <: AbstractCosmology,
    P <: AbstractPropagation,
    Pop <: PopulationModel,
    PR,
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

function prepare_bns_model(
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
    Λ_fid = fiducials
    z = samples.redshift

    observation = build_observation_context(
        frequencies(grid), Vector{Detector}(collect(detectors)),
        in_band_mask(grid), Float64(observation_time))

    c_fid = cosmology(C, Λ_fid)
    dl_fid_sq = luminosity_distance.(z, c_fid) .^ 2
    redshift_grid = collect(Float64, z_grid)
    interp = GridQuery(z, redshift_grid)

    cache_fid = CosmologyCache(c_fid, redshift_grid)
    proposal_prior = single_event_prior(pop, cache_fid, Λ_fid)
    samples_interp = with_redshift_interpolant(samples, interp)
    proposal_log_prob = component_logpdfs(proposal_prior, samples_interp)

    model = BNSPreparedModel{C, P, typeof(pop),
        typeof(proposal_prior), typeof(proposal_log_prob)}(
        pop, redshift_grid, interp, proposal_prior, proposal_log_prob,
        dl_fid_sq, Float64(local_merger_rate), Float64(observation_time))
    return (; model = model, observation = observation)
end

function full_hyperparameters(model::BNSPreparedModel{C, P}) where {C, P}
    return full_hyperparameters(C, P, model.pop)
end

function merger_rate_and_log_weights(
        model::BNSPreparedModel{C, P}, Λ::NamedTuple, samples
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

function bns_samples_from_catalog(catalog_samples::NamedTuple)
    return (
        mass = stack_source_masses(
            catalog_samples.mass_1_source, catalog_samples.mass_2_source),
        redshift = copy(catalog_samples.redshift),
        χ₁ = copy(catalog_samples.chi_1),
        χ₂ = copy(catalog_samples.chi_2),
        Λ₁ = copy(catalog_samples.lambda_1),
        Λ₂ = copy(catalog_samples.lambda_2)
    )
end

# Fixed model selection (see notebooks/mcmc.jl): background cosmology `C` and GW
# propagation `P` are now orthogonal axes.
const C = W0CDM
const P = ModifiedPropagation

# Hard-coded hyperprior bounds (matching notebooks/mcmc.jl).
const HYPERPRIOR = product_distribution((
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.05, 0.95),
    w0 = Uniform(-3, 1),
    Ξ₀ = Uniform(0.5, 5.0),
    Ξₙ = Uniform(0.3, 3.0),
    γ = Uniform(0.5, 10.0),
    κ = Uniform(0.05, 10.0),
    zpeak = Uniform(0.05, 10.0)
))

# --------------------------------------------------------------------------
# Materialization helpers
# --------------------------------------------------------------------------

function _resolve_catalog_path(catalog_path::AbstractString, base::AbstractString)
    return isabspath(catalog_path) ? String(catalog_path) :
           normpath(joinpath(base, catalog_path))
end

function _resolve_adtype(name::AbstractString)
    name == "ForwardDiff" && return AutoForwardDiff()
    throw(ArgumentError("unsupported ad_backend $(repr(name)) (use \"ForwardDiff\")"))
end

"""Order the validated fiducial map into the canonical NamedTuple the model expects."""
function _fiducials_namedtuple(cfg::MCMCConfig, order::Tuple{Vararg{Symbol}})
    validate_fiducials(cfg, order)
    nt = NamedTuple(Tuple(sym => cfg.fiducials[sym] for sym in order))
    return canonical_hyperparameters(order, nt; context = "fiducial hyperparameters")
end

"""Resolve `sample_only` to a tuple of symbols, validating membership in `order`."""
function _resolve_sample_only(cfg::MCMCConfig, order::Tuple{Vararg{Symbol}})
    cfg.sample_only === nothing && return nothing
    isempty(cfg.sample_only) && return nothing
    for sym in cfg.sample_only
        sym in order || throw(ArgumentError(
            "unknown hyperparameter $(repr(sym)) in sample_only; expected one of $order",
        ))
    end
    return Tuple(cfg.sample_only)
end

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

function run_mcmc(config_file::String)
    BLAS.set_num_threads(1)
    num_threads = Base.Threads.nthreads()

    @info "loading config" path = config_file
    cfg = load_config(config_file)

    catalog_path = _resolve_catalog_path(cfg.catalog_path, _REPO_ROOT)
    detectors = [Detector(n) for n in cfg.detectors]
    output_dir = joinpath(_REPO_ROOT, cfg.output_dir)
    output_prefix = cfg.output_prefix

    cfg_nchains = cfg.sampler.nchains
    nchains = cfg_nchains > 0 ? cfg_nchains : num_threads
    nchains == num_threads || throw(ArgumentError(
        "sampler.nchains must equal Base.Threads.nthreads() for MCMCThreads() " *
        "(got nchains=$nchains, nthreads()=$num_threads); " *
        "set nchains = 0 or match -t / SLURM_CPUS_PER_TASK",
    ))

    pop = BNSPopulationModel()
    order = full_hyperparameters(C, P, pop)
    @info "model" cosmology=string(C) propagation=string(P) order
    fiducials = _fiducials_namedtuple(cfg, order)
    sample_only = _resolve_sample_only(cfg, order)

    @info "seeding RNG" seed = cfg.seed
    Random.seed!(cfg.seed)

    @info "loading catalog" catalog_path detectors=join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog
    samples = bns_samples_from_catalog(catalog.samples)
    prepared = prepare_bns_model(
        pop,
        samples,
        fiducials,
        C,
        P,
        loaded.metadata.grid,
        detectors,
        cfg.observation_time,
        cfg.local_merger_rate
    )
    model = prepared.model
    observation = prepared.observation
    @info "catalog loaded" n_frequency_bins=length(observation.frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    config_stem = splitext(basename(config_file))[1]
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(config_stem)-$(params_suffix)-det=$(det_suffix)-seed$(cfg.seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")
    output_toml = joinpath(output_dir, "$base.toml")

    adtype = _resolve_adtype(cfg.sampler.ad_backend)
    @info "starting NUTS" nadapts=cfg.sampler.nadapts nsamples=cfg.sampler.nsamples target_acceptance=cfg.sampler.target_acceptance ad_backend=cfg.sampler.ad_backend sample_only nchains
    turing_model = build_turing_model(
        model,
        catalog.fluxes,
        samples,
        fiducials,
        observation,
        HYPERPRIOR;
        track = true
    )
    conditioned = condition_turing_model(
        turing_model,
        fiducials,
        HYPERPRIOR,
        sample_only;
        order = order
    )
    nuts = Turing.NUTS(
        cfg.sampler.nadapts,
        cfg.sampler.target_acceptance;
        metricT = DenseEuclideanMetric,
        adtype = adtype
    )
    initial_params = fill(InitFromPrior(), nchains)
    chain = sample(
        conditioned,
        nuts,
        MCMCThreads(),
        cfg.sampler.nsamples,
        nchains;
        progress = true,
        save_state = false,
        chain_type = VNChain,
        initial_params = initial_params
    )
    @info "NUTS finished" chain_size = size(chain)

    @info "writing chain to JLD2" path = output_jld2
    atomic_save_chain(output_jld2, chain)
    @info "writing run config to TOML" path = output_toml
    save_config(cfg, output_toml)
    @info "done" output_jld2 output_toml
    return output_jld2
end

end # module AstroSGWBRunMCMC

function (@main)(args::Vector{String})
    length(args) == 1 || throw(ArgumentError("usage: run_mcmc.jl <config.toml>"))
    AstroSGWBRunMCMC.run_mcmc(args[1])
    return 0
end
