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
                 build_model_context,
                 canonical_hyperparameters,
                 full_hyperparameters,
                 load_catalog,
                 AbstractCosmology,
                 PopulationModel,
                 ImportanceSamplingProblem,
                 ModifiedPropagation,
                 W0CDM,
                 Detector,
                 OrderedUniformSourceMassPair,
                 AlignedSpinChiSimple,
                 redshift_prior,
                 MadauDickinsonSourceFrame,
                 stack_source_masses
import AstroSGWB: hyperparameters, single_event_prior
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

function single_event_prior(::BNSPopulationModel, cosmo::AbstractCosmology, Λ::NamedTuple)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), cosmo, Λ)
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

# Fixed model selection (see notebooks/mcmc.jl).
const C = ModifiedPropagation{W0CDM}

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
    order = full_hyperparameters(C, pop)
    @info "model" cosmology=string(C) order
    fiducials = _fiducials_namedtuple(cfg, order)
    sample_only = _resolve_sample_only(cfg, order)

    @info "seeding RNG" seed = cfg.seed
    Random.seed!(cfg.seed)

    @info "loading catalog" catalog_path detectors=join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog
    samples = bns_samples_from_catalog(catalog.samples)
    problem = ImportanceSamplingProblem(pop, catalog.fluxes, samples, fiducials)
    ctx = build_model_context(
        problem,
        C,
        loaded.metadata.grid,
        detectors,
        cfg.observation_time,
        cfg.local_merger_rate
    )
    @info "catalog loaded" n_frequency_bins=length(ctx.observation.frequencies) n_proposal_samples=length(
        problem.samples.redshift,
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
        problem,
        C,
        ctx,
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
