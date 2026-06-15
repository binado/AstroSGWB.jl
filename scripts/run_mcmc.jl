# Headless, config-driven NUTS runner for the ASGWB importance-sampling model.
#
# This mirrors the sampling cells of notebooks/mcmc_pluto.jl but takes every
# run-specific setting from a TOML config, so cluster runs never edit the
# notebook. Cosmology family and population are fixed here (matching the
# notebook): ModifiedPropagation{W0CDM} + BNSPopulationModel.
#
# Run from the repository root, for example:
#   julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/example.toml

module ASGWBRunMCMC

const _REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

using ASGWB
using ASGWB:
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
             BNS_LAMBDA_HIGH,
             stack_source_masses
import ASGWB: hyperparameters, single_event_prior
using ASGWBInference: build_turing_model, condition_turing_model, atomic_save_chain
using ADTypes: AutoForwardDiff, AutoReverseDiff
using AdvancedHMC: DenseEuclideanMetric
using Distributions: Uniform, product_distribution
using FlexiChains: VNChain
using Turing
using Random
using TOML
using Logging
using LinearAlgebra: BLAS
using Dates: now, format

# --------------------------------------------------------------------------
# Population model (kept inline, matching notebooks/mcmc_pluto.jl)
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
        Λ₁ = Uniform(0.0, BNS_LAMBDA_HIGH),
        Λ₂ = Uniform(0.0, BNS_LAMBDA_HIGH)
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

# Fixed model selection (see notebooks/mcmc_pluto.jl).
const C = ModifiedPropagation{W0CDM}

# Map between TOML ASCII keys and the canonical Unicode hyperparameter symbols.
const _ASCII_TO_SYM = Dict(
    "H0" => :H0,
    "Omega_m" => :Ωm,
    "w0" => :w0,
    "Xi_0" => :Ξ₀,
    "Xi_n" => :Ξₙ,
    "gamma" => :γ,
    "kappa" => :κ,
    "z_peak" => :zpeak
)
const _SYM_TO_ASCII = Dict(v => k for (k, v) in _ASCII_TO_SYM)

# --------------------------------------------------------------------------
# TOML helpers
# --------------------------------------------------------------------------

function _require(settings::Dict, key::AbstractString)
    haskey(settings, key) || throw(ArgumentError("missing required TOML key $(repr(key))"))
    return settings[key]
end

function _require_table(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Dict || throw(ArgumentError("TOML key $(repr(key)) must be a table"))
    return v
end

function _require_string_array(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Vector || throw(ArgumentError("TOML key $(repr(key)) must be an array"))
    all(x -> x isa AbstractString, v) ||
        throw(ArgumentError("TOML key $(repr(key)) must be an array of strings"))
    return Vector{String}(v)
end

function _resolve_catalog_path(catalog_path::String, base::AbstractString)
    return isabspath(catalog_path) ? catalog_path :
           normpath(joinpath(base, catalog_path))
end

"""Build the canonical fiducial NamedTuple from the `[fiducials]` table."""
function _fiducials_from_toml(fid_tbl::Dict, order::Tuple{Vararg{Symbol}})
    pairs = map(order) do sym
        ascii = _SYM_TO_ASCII[sym]
        haskey(fid_tbl, ascii) ||
            throw(ArgumentError("missing fiducial value [fiducials].$ascii"))
        sym => Float64(fid_tbl[ascii])
    end
    nt = NamedTuple(pairs)
    return canonical_hyperparameters(order, nt; context = "fiducial hyperparameters")
end

function _uniform_bounds(priors_tbl::Dict, ascii::AbstractString)
    sub = priors_tbl[ascii]
    sub isa Dict ||
        throw(ArgumentError("priors.$ascii must be a table with 'low' and 'high'"))
    lo = Float64(sub["low"])
    hi = Float64(sub["high"])
    isfinite(lo) && isfinite(hi) ||
        throw(ArgumentError("priors.$ascii: low and high must be finite"))
    lo < hi || throw(ArgumentError("priors.$ascii: require low < high, got ($lo, $hi)"))
    return lo, hi
end

"""Build the hyperprior `product_distribution` in the canonical `order`."""
function _priors_from_toml(priors_tbl::Dict, order::Tuple{Vararg{Symbol}})
    pairs = map(order) do sym
        ascii = _SYM_TO_ASCII[sym]
        haskey(priors_tbl, ascii) ||
            throw(ArgumentError("missing prior bounds [priors.$ascii]"))
        sym => Uniform(_uniform_bounds(priors_tbl, ascii)...)
    end
    return product_distribution(NamedTuple(pairs))
end

"""Normalize a `sample_only` entry (ASCII alias or Unicode string) to a symbol."""
function _to_sym(name::AbstractString)
    haskey(_ASCII_TO_SYM, name) && return _ASCII_TO_SYM[name]
    sym = Symbol(name)
    sym in values(_ASCII_TO_SYM) || throw(
        ArgumentError("unknown hyperparameter name $(repr(name)) in sample_only"),
    )
    return sym
end

function _sample_only_from_toml(cfg::Dict)
    haskey(cfg, "sample_only") || return nothing
    raw = cfg["sample_only"]
    raw === nothing && return nothing
    raw isa Vector ||
        throw(ArgumentError("sample_only must be an array of strings (or omitted)"))
    isempty(raw) && return nothing
    return Tuple(_to_sym(String(x)) for x in raw)
end

function _resolve_adtype(name::AbstractString)
    if name == "ForwardDiff"
        return AutoForwardDiff()
    elseif name == "ReverseDiff"
        return AutoReverseDiff()
    else
        throw(ArgumentError("unsupported ad_backend $(repr(name)) (use \"ForwardDiff\" or \"ReverseDiff\")"))
    end
end

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

function _run(config_file::String)
    BLAS.set_num_threads(1)
    num_threads = Base.Threads.nthreads()

    @info "loading config" path = config_file
    cfg = TOML.parsefile(config_file)

    catalog_path = _resolve_catalog_path(_require(cfg, "catalog_path")::String, _REPO_ROOT)
    detectors = [Detector(n) for n in _require_string_array(cfg, "detectors")]
    seed = Int(_require(cfg, "seed"))
    local_merger_rate = Float64(_require(cfg, "local_merger_rate"))
    observation_time_yr = Float64(_require(cfg, "observation_time_yr"))
    output_dir = joinpath(_REPO_ROOT, String(get(cfg, "output_dir", "chains")))
    output_prefix = String(get(cfg, "output_prefix", "chains"))

    sampler_tbl = _require_table(cfg, "sampler")
    n_samples = Int(_require(sampler_tbl, "n_samples"))
    n_adapts = Int(_require(sampler_tbl, "n_adapts"))
    target_acceptance = Float64(_require(sampler_tbl, "target_acceptance"))
    ad_backend = String(_require(sampler_tbl, "ad_backend"))
    cfg_num_chains = Int(get(sampler_tbl, "num_chains", 0))

    num_chains = cfg_num_chains > 0 ? cfg_num_chains : num_threads
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    pop = BNSPopulationModel()
    order = full_hyperparameters(C, pop)
    @info "model" cosmology=string(C) order
    fiducials = _fiducials_from_toml(_require_table(cfg, "fiducials"), order)
    hyperprior = _priors_from_toml(_require_table(cfg, "priors"), order)
    sample_only = _sample_only_from_toml(cfg)

    @info "seeding RNG" seed
    Random.seed!(seed)

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
        observation_time_yr,
        local_merger_rate
    )
    @info "catalog loaded" n_frequency_bins=length(ctx.observation.frequencies) n_proposal_samples=length(
        problem.samples.redshift,
    )

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = ctx.fiducial_spectral_density

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")

    adtype = _resolve_adtype(ad_backend)
    @info "starting NUTS" n_adapts n_samples target_acceptance ad_backend sample_only num_chains
    turing_model = build_turing_model(
        problem,
        C,
        ctx,
        hyperprior;
        track = true,
        observed = observed
    )
    conditioned = condition_turing_model(
        turing_model,
        fiducials,
        hyperprior,
        sample_only;
        order = order
    )
    nuts = Turing.NUTS(
        n_adapts,
        target_acceptance;
        metricT = DenseEuclideanMetric,
        adtype = adtype
    )
    initial_params = fill(InitFromPrior(), num_chains)
    chain = sample(
        conditioned,
        nuts,
        MCMCThreads(),
        n_samples,
        num_chains;
        progress = true,
        save_state = false,
        chain_type = VNChain,
        initial_params = initial_params
    )
    @info "NUTS finished" chain_size = size(chain)

    @info "writing chain to JLD2" path = output_jld2
    atomic_save_chain(output_jld2, chain)
    @info "done" output_jld2
    return output_jld2
end

function command_main(args::Vector{String} = ARGS)::Cint
    try
        length(args) == 1 ||
            throw(ArgumentError("usage: run_mcmc.jl <config.toml>"))
        _run(args[1])
        return Cint(0)
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        return Cint(1)
    end
end

end # module ASGWBRunMCMC

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(Base.invokelatest(ASGWBRunMCMC.command_main))
end
