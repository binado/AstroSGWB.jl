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
                 differential_comoving_volume,
                 redshift_density,
                 source_frame_distribution,
                 normalizer,
                 redshift_logpdf_eltype,
                 _normalized_log_density,
                 interpolate,
                 luminosity_distance_at_sample,
                 gw_em_distance_ratio,
                 integrated_merger_rate,
                 ModifiedPropagation,
                 W0CDM,
                 Detector,
                 MadauDickinsonSourceFrame
using AstroSGWBInference:
                          build_turing_model,
                          condition_turing_model,
                          atomic_save_chain,
                          MCMCConfig,
                          load_config,
                          save_config,
                          validate_fiducials
import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights
import Cosmology
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
# Slim BNS importance model (matching notebooks/mcmc.jl and Python mcmc.py)
#
# The six-component single-event prior collapses to a single redshift log-ratio
# (mass/spin/tidal are Λ-independent and cancel exactly) plus a distance and
# propagation factor, so `merger_rate_and_log_weights` inlines the redshift +
# importance-weight math over the load-bearing Cosmology / CBCDistributions
# kernels. The background cosmology `C` and propagation `P` stay compile-time
# type parameters, preserving the cosmology-agnostic dispatch.
# --------------------------------------------------------------------------

"""Joint hyperparameter names for the BNS importance model with cosmology `C`, propagation `P`."""
function bns_order(::Type{C}, ::Type{P}) where {C, P}
    (Cosmology.hyperparameters(C)...,
        Cosmology.propagation_hyperparameters(P)..., :γ, :κ, :zpeak)
end

struct BNSImportanceModel{C, P}
    z_grid::Vector{Float64}
    query::GridQuery                  # hoists per-sample grid search out of the AD loop
    proposal_log_pdf::Vector{Float64} # fiducial redshift log-density, computed once
    local_merger_rate::Float64
    observation_time::Float64
end

hyperparameters(::BNSImportanceModel{C, P}) where {C, P} = bns_order(C, P)

function prepare_bns_model(
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
    z = samples.redshift
    observation = build_observation_context(
        frequencies(grid), Vector{Detector}(collect(detectors)),
        in_band_mask(grid), Float64(observation_time))
    zg = collect(Float64, z_grid)
    query = GridQuery(z, zg)

    # Fiducial proposal redshift log-density, evaluated once (≙ mcmc.py `log_p_proposal`).
    cache_fid = CosmologyCache(cosmology(C, fiducials), zg)
    dvc_fid = differential_comoving_volume.(zg, Ref(cache_fid))
    dN_dz_fid = redshift_density(zg, dvc_fid, MadauDickinsonSourceFrame(), fiducials)
    norm_fid = normalizer(dN_dz_fid)
    tiny = floatmin(Float64)
    proposal_log_pdf = [_normalized_log_density(
                            interpolate(dN_dz_fid, query, i), norm_fid, tiny)
                        for i in eachindex(z)]

    model = BNSImportanceModel{C, P}(zg, query, proposal_log_pdf,
        Float64(local_merger_rate), Float64(observation_time))
    return (; model = model, observation = observation)
end

function merger_rate_and_log_weights(
        m::BNSImportanceModel{C, P}, Λ::NamedTuple, samples
) where {C, P}
    z = samples.redshift
    d_l_fid = samples.luminosity_distance        # EM distance at fiducial; flux ∝ 1/d_l_fid²
    cache = CosmologyCache(cosmology(C, Λ), m.z_grid)
    prop = propagation(P, Λ)

    dvc_grid = differential_comoving_volume.(m.z_grid, Ref(cache))
    dN_dz = redshift_density(                      # target detector-frame dN/dz on the grid
        m.z_grid, dvc_grid, MadauDickinsonSourceFrame(), Λ)
    norm = normalizer(dN_dz)
    tiny = floatmin(real(eltype(dN_dz.y)))        # AD-safe (Dual under ForwardDiff)

    # Preallocate to the promoted element type so the explicit loop stays type-stable under
    # ForwardDiff. Promote the redshift logpdf eltype with the propagation factor to also
    # cover the Ξ-only-sampled case; `zero(eltype(z))` is an index-free probe (empty-safe).
    T = promote_type(redshift_logpdf_eltype(dN_dz),
        typeof(gw_em_distance_ratio(zero(eltype(z)), prop)))
    log_weights = Vector{T}(undef, length(z))
    @inbounds for i in eachindex(z)               # single fused pass (≙ mcmc.py weights)
        log_p_target = _normalized_log_density(
            interpolate(dN_dz, m.query, i), norm, tiny)
        d_l_θ = luminosity_distance_at_sample(cache, m.query, z, i)
        Ξ_θ = gw_em_distance_ratio(z[i], prop)
        log_weights[i] = (log_p_target - m.proposal_log_pdf[i]) +
                         2 * log(d_l_fid[i]) - 2 * log(d_l_θ) - 2 * log(Ξ_θ)
    end

    rate = integrated_merger_rate(dN_dz, m.local_merger_rate)
    return (rate, log_weights)
end

# Restructure catalog columns into the slim `samples` the weight loop reads. The `samples`
# NamedTuple is the single source of truth for the fiducial EM distance: if the catalog
# ships a `luminosity_distance` column it is used as-is, otherwise it is generated once
# from redshift at the fiducial cosmology `C`.
function bns_samples_from_catalog(
        catalog_samples::NamedTuple, ::Type{C}, fiducials::NamedTuple) where {C}
    z = copy(catalog_samples.redshift)
    d_l = haskey(catalog_samples, :luminosity_distance) ?
          copy(catalog_samples.luminosity_distance) :
          luminosity_distance.(z, cosmology(C, fiducials))
    return (redshift = z, luminosity_distance = d_l)
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

    order = bns_order(C, P)
    @info "model" cosmology=string(C) propagation=string(P) order
    fiducials = _fiducials_namedtuple(cfg, order)
    sample_only = _resolve_sample_only(cfg, order)

    @info "seeding RNG" seed = cfg.seed
    Random.seed!(cfg.seed)

    @info "loading catalog" catalog_path detectors=join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog
    samples = bns_samples_from_catalog(catalog.samples, C, fiducials)
    prepared = prepare_bns_model(
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
        sample_only
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
