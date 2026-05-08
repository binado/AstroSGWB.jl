#!/usr/bin/env julia

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

module RunInferenceCLI

using ASGWB
using ASGWB: load_cache, build_turing_model, Detector, DEFAULT_PARAMETER_ORDER

using Turing
using AdvancedHMC
using Random
using Serialization
using ArviZ
using NCDatasets
using Distributions
using TOML
using Pkg
using LinearAlgebra: BLAS
using MCMCChains: Chains
using AbstractMCMC: bundle_samples, chainsstack
using Dates: now, format


"""Check each `init` scalar has positive prior density under the matching `priors` entry."""
function validate_init_against_priors(priors, init)
    for (k, d) in pairs(priors)
        v = init[k]
        isfinite(logpdf(d, v)) || throw(
            ArgumentError("init.$k = $v is outside the support of the corresponding prior"),
        )
    end
    return nothing
end

const PRIORS = (
    H0 = Uniform(20, 140),
    Ωm = Uniform(0.05, 0.95),
    Ξ₀ = Uniform(0.5, 5),
    Ξₙ = Uniform(0.05, 3),
    γ = Uniform(0.5, 10),
    κ = Uniform(0.05, 10),
    zpeak = Uniform(0.05, 10)
)

"""Resolve `path` relative to `base` if it is not absolute."""
function resolve_path(path::AbstractString, base::AbstractString)
    isabspath(path) ? path : normpath(joinpath(base, path))
end

"""
    CheckpointCallback(every, path, model, sampler, num_chains)

AbstractMCMC callback that buffers per-chain transitions and the latest sampler
state, and serializes a multi-chain `MCMCChains.Chains` snapshot to `path` every
time the minimum number of samples across chains crosses a new multiple of
`every`. With `save_state = true` on the bundled samples, the snapshot is
compatible with `resume_from`.
"""
mutable struct CheckpointCallback{M, S}
    every::Int
    path::String
    model::M
    sampler::S
    transitions::Vector{Vector{Any}}
    states::Vector{Any}
    last_checkpoint_iter::Int
    lock::ReentrantLock
end

function CheckpointCallback(
        every::Int, path::AbstractString, model, sampler, num_chains::Int
)
    return CheckpointCallback(
        every,
        String(path),
        model,
        sampler,
        [Vector{Any}() for _ in 1:num_chains],
        Vector{Any}(undef, num_chains),
        0,
        ReentrantLock()
    )
end

function (cb::CheckpointCallback)(
        rng, model, sampler, transition, state, iteration;
        chain_number::Int = 1, kwargs...
)
    lock(cb.lock) do
        push!(cb.transitions[chain_number], transition)
        cb.states[chain_number] = state

        n = minimum(length, cb.transitions)
        target = (cb.last_checkpoint_iter ÷ cb.every + 1) * cb.every
        n >= target || return nothing

        per_chain = map(eachindex(cb.transitions)) do c
            bundle_samples(
                cb.transitions[c][1:n], cb.model, cb.sampler, cb.states[c],
                Chains; save_state = true
            )
        end
        snapshot = chainsstack(per_chain)

        tmp = cb.path * ".tmp"
        Serialization.serialize(tmp, snapshot)
        mv(tmp, cb.path; force = true)
        cb.last_checkpoint_iter = n
        @info "checkpoint written" path=cb.path iteration=n
    end
    return nothing
end


function _run(settings::Dict, settings_dir::AbstractString)
    cache = resolve_path(settings["cache_path"]::String, settings_dir)
    detectors = [Detector(n) for n in settings["detectors"]]
    sample_only = Tuple(Symbol(s) for s in settings["sample_only"])
    seed = settings["seed"]::Int
    init = (; (Symbol(k) => v for (k, v) in settings["init"])...)

    sampler = settings["sampler"]
    n_samples = sampler["n_samples"]::Int
    n_adapts = sampler["n_adapts"]::Int
    target_acceptance = sampler["target_acceptance"]::Float64
    num_chains = get(sampler, "num_chains", 0)::Int
    num_chains = num_chains > 0 ? num_chains : Base.Threads.nthreads()
    checkpoint_every = get(sampler, "checkpoint_every", 0)::Int

    output_dir = resolve_path(get(settings, "output_dir", ".")::String, settings_dir)
    output_prefix = get(settings, "output_prefix", "chains")::String
    mkpath(output_dir)

    # Validate sample_only
    isempty(sample_only) && throw(ArgumentError("sample_only must not be empty"))
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    for s in sample_only
        s in DEFAULT_PARAMETER_ORDER || throw(
            ArgumentError("sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)"),
        )
    end

    fixed_sites = (; (k => init[k] for k in DEFAULT_PARAMETER_ORDER if k ∉ sample_only)...)

    timestamp = format(now(), "yyyymmdd-HHMMSS")
    params_suffix = join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-seed$(seed)-$(timestamp)"
    output_jls = joinpath(output_dir, "$base.jls")
    output_netcdf = joinpath(output_dir, "$base.nc")
    checkpoint_path = joinpath(output_dir, "$base.partial.jls")

    validate_init_against_priors(PRIORS, init)
    priors_turing = product_distribution(PRIORS)

    # Cluster-friendly defaults: avoid BLAS oversubscription with MCMCThreads
    # and disable the carriage-return progress bar in non-TTY contexts.
    BLAS.set_num_threads(1)
    progress = isinteractive()

    num_threads = Base.Threads.nthreads()
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    @info "starting run" julia=VERSION threads=num_threads chains=num_chains blas_threads=BLAS.get_num_threads() cache detectors=join(
        (d.name for d in detectors), ",") sample_only output_dir
    @info "package versions"
    Pkg.status()

    @info "loading importance cache" path=cache
    t_cache = time()
    problem = load_cache(cache, detectors)
    @info "cache loaded" seconds=round(time()-t_cache; digits = 2) n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = problem.observation.fiducial_spectral_density

    @info "seeding RNG" rng_seed=seed
    Random.seed!(seed)

    @info "starting NUTS" n_adapts n_samples target_acceptance sample_only checkpoint_every
    model = build_turing_model(problem, priors_turing; track = true, observed_spectral_density = observed)
    conditioned = model | fixed_sites
    nuts = Turing.NUTS(
        n_adapts,
        target_acceptance;
        metricT = AdvancedHMC.DenseEuclideanMetric
    )
    callback = checkpoint_every > 0 ?
               CheckpointCallback(
        checkpoint_every, checkpoint_path, conditioned, nuts, num_chains
    ) : nothing

    chain = if callback === nothing
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = true
        )
    else
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = true, callback = callback
        )
    end
    @info "NUTS finished" chain_size=size(chain)

    @info "writing chain to JLS" path=output_jls
    Serialization.serialize(output_jls, chain)
    @info "wrote chain to JLS" path=output_jls

    @info "writing InferenceData to NetCDF" path=output_netcdf
    idata = from_mcmcchains(chain; library = "Turing")
    to_netcdf(idata, output_netcdf)
    @info "wrote InferenceData to NetCDF" path=output_netcdf

    if isfile(checkpoint_path)
        @info "removing checkpoint" path=checkpoint_path
        rm(checkpoint_path; force = true)
    end

    @info "done"
    return nothing
end

function main()
    default_path = joinpath(@__DIR__, "run_inference.toml")
    settings_path = get(ENV, "MCMC_CONFIG_FILEPATH",
        isempty(ARGS) ? default_path : ARGS[1])
    settings_path = abspath(settings_path)
    @info "loading settings" path=settings_path
    s = TOML.parsefile(settings_path)
    return _run(s, dirname(settings_path))
end

end # module RunInferenceCLI

RunInferenceCLI.main()
