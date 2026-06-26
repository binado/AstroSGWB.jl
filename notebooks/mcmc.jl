### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using AstroSGWB
    using AstroSGWB:
                     canonical_hyperparameters,
                     cosmology_type,
                     Detector,
                     PopulationModel,
                     AbstractCosmology,
                     AbstractPropagation,
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
                     load_catalog,
                     OrderedUniformSourceMassPair,
                     AlignedSpinChiSimple,
                     redshift_prior,
                     MadauDickinsonSourceFrame,
                     stack_source_masses,
                     spectral_density,
                     year_to_second,
                     Ωgw
    using AstroSGWBInference: build_turing_model, condition_turing_model
    using AstroSGWBInference: MCMCConfig, SamplerConfig, save_config
    using AstroSGWBInference.ChainIO: atomic_save_chain
    using Distributions: Uniform, product_distribution
    using Turing
    using AdvancedHMC
    using ADTypes
    using Random
    using JLD2: load
    using Logging
    using FlexiChains
    using FlexiChains: VNChain
    using PairPlots
    using CairoMakie
    using LaTeXStrings
    using Dates: now, format
    using Printf
    using LinearAlgebra: BLAS
    BLAS.set_num_threads(1)
end

# ╔═╡ 8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
md"""
# Cosmological parameter inference with the astrophysical GWB

In this notebook, we perform Bayesian inference on the cosmological and astrophysical parameters that play into the gravitational-wave background of stellar-mass compact binary coalescences (CBCs) such as neutron stars or black holes.

To properly run the notebook, you must specify a path to a catalog HDF5 file containing the intrinsic parameter samples of the CBC population as well as the associated waveforms.
"""

# ╔═╡ 9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# ╔═╡ a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
md"""
## Population model

Inference requires specifying a population model, parametrized by a vector ``\Lambda`` ,  which characterizes the distribution of the intrinsic parameters ``p(\theta | \Lambda)``.

At the level of the code, the user must implement a concrete `PopulationModel` subtype (see `AstroSGWB/src/models/base.jl`), overriding the following two methods:

- **`hyperparameters`** — declares which population hyperparameters (beyond cosmology) enter the model. For instance, for a Madau-Dickinson like redshift distribution, that would be `:γ`, `:κ`, `:zpeak`.
- **`single_event_prior`** — defines the per-event intrinsic prior as a `product_distribution` over mass, redshift, spin, and (for BNS) tidal parameters.

`bns_samples_from_catalog` restructures catalog columns into the `NamedTuple` layout expected by `single_event_prior`.
"""

# ╔═╡ 2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
begin
    import AstroSGWB: hyperparameters, single_event_prior,
                      merger_rate_and_log_weights, full_hyperparameters

    struct BNSPopulationModel <: PopulationModel end

    hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

    function single_event_prior(
            ::BNSPopulationModel, cache::CosmologyCache, Λ::NamedTuple)
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

    # Prepared model: out-of-package assembly of the cosmology-agnostic inference contract.
    # The background cosmology `C` and propagation `P` are type parameters, so the package
    # never sees a cosmology token; it dispatches solely on this model.
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
            full_hyperparameters(model), Λ; context = "joint hyperparameters",
            eltype = nothing)
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
end

# ╔═╡ b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
md"""
## Configuration

Edit runtime settings here: `catalog_path`, detectors, observation time, merger rate, fiducials, `hyperprior_dists` / `hyperprior`, sampler (`nsamples`, `nadapts`, `ad_backend`, `nchains`), output paths, `chain_input_jld2`, and `DEBUG`.
"""

# ╔═╡ c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f
begin
    DEBUG = false
    @info debug = DEBUG

    function resolve_adtype(name::AbstractString)
        if name == "ForwardDiff"
            return ADTypes.AutoForwardDiff()
        else
            throw(ArgumentError(
                "this notebook supports only ad_backend = \"ForwardDiff\"; got $(repr(name))",
            ))
        end
    end

    _repo_root = normpath(joinpath(@__DIR__, ".."))

    catalog_path = joinpath(_repo_root, "catalog.h5")
    detnames = [:S1, :R1, :C1]
    detectors = map(Detector ∘ string, detnames)
    sample_only = (:H0,)

    seed = 42
    @info "seeding RNG" rng_seed = seed
    Random.seed!(seed)

    local_merger_rate = 161.0 # Matches COBA simulations
    observation_time = 1.0

    output_dir = joinpath(_repo_root, "chains")
    output_prefix = "chains"

    sampler = (
        nsamples = 3000,
        nadapts = 3000,
        target_acceptance = 0.9,
        ad_backend = "ForwardDiff",
        nchains = 0
    )

    cosmology_parameters = (;
        H0 = 67.66,
        Ωm = 0.3096,
        w0 = -1,
        Ξ₀ = 1.0,
        Ξₙ = 1.91
    )
    fiducials = (;
        cosmology_parameters...,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.0
    )

    # Edit hyperprior bounds here (order: cosmology, then population).
    hyperprior_dists = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        w0 = Uniform(-3, 1),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.3, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    )
    hyperprior = product_distribution(hyperprior_dists)

    # Defining cosmology, propagation, and population model. Background expansion `C`
    # and GW propagation `P` are orthogonal axes (use `GR` for standard propagation).
    C = W0CDM
    P = ModifiedPropagation
    pop = BNSPopulationModel()

    chain_input_jld2 = nothing

    nchains = sampler.nchains > 0 ? sampler.nchains : num_threads
end

# ╔═╡ 3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
begin
    if nchains != num_threads
        @warn "nchains differs from Base.Threads.nthreads()" nchains num_threads
    end

    @info "loading catalog" catalog_path detectors = join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog

    @info hyperparameters(C)
    samples = bns_samples_from_catalog(catalog.samples)
    prepared = prepare_bns_model(
        pop,
        samples,
        fiducials,
        C,
        P,
        loaded.metadata.grid,
        detectors,
        observation_time,
        local_merger_rate
    )
    model = prepared.model
    observation = prepared.observation
    order = full_hyperparameters(C, P, pop)
    @info order
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)

    @info "catalog loaded" n_frequency_bins=length(observation.frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")
    output_toml = joinpath(output_dir, "$base.toml")

    # Reproducible record of this run's settings, dumped on a successful run.
    run_config = MCMCConfig(
        1,
        catalog_path,
        string.(detnames),
        seed,
        observation_time,
        local_merger_rate,
        SamplerConfig(
            sampler.nsamples,
            sampler.nadapts,
            sampler.target_acceptance,
            sampler.ad_backend,
            sampler.nchains
        ),
        Dict{Symbol, Float64}(k => Float64(v) for (k, v) in pairs(fiducials)),
        sample_only_tup === nothing ? nothing : collect(Symbol, sample_only_tup),
        output_dir,
        output_prefix
    )

    nothing
end

# ╔═╡ c2627b5e-b9f4-4535-b0c3-69ce8b2a696c
md"""
## Visualizing ``\Omega_{\mathrm{GW}}``

In the cells below, we plot ``\Omega_{\mathrm{GW}}(f)`` as a function of the frequency ``f`` for the fiducial values of the parameters ``\Lambda``.
"""

# ╔═╡ d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a
function plot_fiducial_omega_gw(model, fluxes, samples, fiducials, observation)
    rate0, log_weights0 = merger_rate_and_log_weights(model, fiducials, samples)
    Sh0 = spectral_density(fluxes, rate0; weights = exp.(log_weights0))
    f = observation.frequencies
    df = frequency_bin_width(f)
    snr = spectral_snr(
        Sh0,
        observation.effective_psd,
        year_to_second(observation.observation_time),
        df
    )

    Ωgw_plot = Ωgw(Sh0, f, fiducials.H0)
    mask = Ωgw_plot .> 0.0
    fm = f[mask]
    Ωgw_pos = Ωgw_plot[mask]
    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1];
        xlabel = L"$f~\mathrm{(Hz)}$",
        ylabel = L"$\Omega_{\mathrm{GW}}(f)$",
        xscale = log10,
        yscale = log10,
        limits = (nothing, nothing, 1e-15, nothing)
    )
    if !isempty(Ωgw_pos)
        label = @sprintf "SNR = %.1f" snr
        lines!(ax, fm, Ωgw_pos; label = label)
        axislegend(ax; position = :rt)
    end
    return fig
end

# ╔═╡ 5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
plot_fiducial_omega_gw(model, catalog.fluxes, samples, fiducials, observation)

# ╔═╡ ccf43d43-7f31-41e9-85db-12842561973c
md"""
## Running the MCMC
"""

# ╔═╡ 7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
begin
    if chain_input_jld2 !== nothing
        chain_path = isabspath(chain_input_jld2) ? String(chain_input_jld2) :
                     normpath(joinpath(_repo_root, chain_input_jld2))
        isfile(chain_path) ||
            throw(ArgumentError("JLD2 chain file not found: $(repr(chain_path))"))
        @info "loading chain from JLD2" path = chain_path
        chain = load(chain_path)["chain"]
        @info "chain loaded" chain_size = size(chain)
    else
        initial_params = fill(InitFromPrior(), nchains)
        adtype = resolve_adtype(sampler.ad_backend)

        @info "starting NUTS" nadapts=sampler.nadapts nsamples=sampler.nsamples target_acceptance=sampler.target_acceptance ad_backend=sampler.ad_backend sample_only=sample_only_tup
        turing_model = build_turing_model(
            model,
            catalog.fluxes,
            samples,
            fiducials,
            observation,
            hyperprior;
            track = false
        )
        conditioned = condition_turing_model(
            turing_model,
            fiducials,
            hyperprior,
            sample_only_tup;
            order = order
        )
        nuts = Turing.NUTS(
            sampler.nadapts,
            sampler.target_acceptance;
            metricT = AdvancedHMC.DenseEuclideanMetric,
            adtype = adtype
        )
        if DEBUG
            @info "MCMC skipped for debugging"
            chain = nothing
        else
            chain = sample(
                conditioned,
                nuts,
                MCMCThreads(),
                sampler.nsamples,
                nchains;
                progress = true,
                save_state = false,
                chain_type = VNChain,
                initial_params = initial_params
            )
            @info "NUTS finished" chain_size = size(chain)
        end
    end
    chain
end

# ╔═╡ 8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
md"""
## Saving the chains to an output file
"""

# ╔═╡ 9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
begin
    if chain_input_jld2 === nothing && chain != nothing
        @info "writing chain to JLD2" path = output_jld2
        atomic_save_chain(output_jld2, chain)
        @info "writing run config to TOML" path = output_toml
        save_config(run_config, output_toml)
        @info "done"
    else
        @info "skipping JLD2 save (chain was loaded from disk)"
    end
end

# ╔═╡ 0e4f3a2b-5c6d-4e7f-8a9b-0c1d2e3f4a5b
md"""
## Diagnostic plots
"""

# ╔═╡ 1f5a4b3c-6d7e-4f8a-9b0c-1d2e3f4a5b6c
summarystats(chain)

# ╔═╡ 2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
FlexiChains.mtraceplot(chain)

# ╔═╡ 3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    FlexiChains.mautocorplot(chain; lags = 1:autocor_maxlag)
end

# ╔═╡ 4c8d7e6f-9a0b-4c1d-2e3f-4a5b6c7d8e9f
begin
    chain_params = FlexiChains.parameters(chain)
    fig = if length(chain_params) >= 2
        pairplot(chain)
    else
        Makie.density(chain)
    end
    fig
end

# ╔═╡ Cell order:
# ╠═8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
# ╠═9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
# ╠═1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
# ╠═a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
# ╠═2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
# ╟─b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
# ╠═c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f
# ╠═3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
# ╠═c2627b5e-b9f4-4535-b0c3-69ce8b2a696c
# ╠═d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a
# ╠═5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
# ╠═ccf43d43-7f31-41e9-85db-12842561973c
# ╠═7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
# ╠═8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
# ╠═9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
# ╟─0e4f3a2b-5c6d-4e7f-8a9b-0c1d2e3f4a5b
# ╠═1f5a4b3c-6d7e-4f8a-9b0c-1d2e3f4a5b6c
# ╠═2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
# ╠═3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
# ╠═4c8d7e6f-9a0b-4c1d-2e3f-4a5b6c7d8e9f
