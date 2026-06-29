using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using AstroSGWBInference: build_turing_model, condition_turing_model, logposterior,
                          merger_rate_and_log_weights
using Distributions: product_distribution, Uniform
using ForwardDiff

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "..", "..", "AstroSGWB", "test", "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "..", "..", "AstroSGWB", "test", "parity_fixtures.jl"))

@testset "Turing model smoke test" begin
    for variant in (:posterior, :full_intrinsic)
        loaded = parity_problem_context(variant, [Detector("H1"), Detector("L1")])
        prepared, observation = loaded.model, loaded.observation
        fluxes, samples, fiducials = loaded.fluxes, loaded.samples, loaded.fiducials
        theta0 = PARITY_THETA
        priors = PARITY_PRIORS
        order = _PARITY_ORDER

        model = build_turing_model(
            prepared, fluxes, samples, fiducials, observation, priors; track = false)
        observed = fiducial_spectral_density(prepared, fluxes, samples, fiducials)
        Λ_fid = canonical_hyperparameters(order, fiducials)
        rate_fid,
        log_weights_fid = merger_rate_and_log_weights(
            prepared, Λ_fid, samples)
        Sh_fid = spectral_density(fluxes, rate_fid; weights = exp.(log_weights_fid))
        @test observed ≈ Sh_fid
        @test Turing.logjoint(model, theta0) ≈
              logposterior(
            theta0, prepared, fluxes, samples, observation, priors, observed) rtol = 1e-6
        @test condition_turing_model(
            model, theta0, priors, nothing) ===
              model
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, ())
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (:unknown,))
        @test_throws ArgumentError condition_turing_model(
            model, theta0, priors, (:H0, :H0))

        model_track = build_turing_model(
            prepared, fluxes, samples, fiducials, observation, priors; track = true)
        returned_nt = Turing.returned(model_track, theta0)
        @test haskey(returned_nt, :effective_sample_size)
        @test isfinite(returned_nt.effective_sample_size)
        @test 0 < returned_nt.effective_sample_size <= 1

        @test haskey(returned_nt, :spectral_snr_squared)
        @test haskey(returned_nt, :spectral_snr)
        @test isfinite(returned_nt.spectral_snr_squared)
        @test isfinite(returned_nt.spectral_snr)
        @test returned_nt.spectral_snr >= 0
        @test returned_nt.spectral_snr^2 ≈ returned_nt.spectral_snr_squared

        sampled_model = condition_turing_model(
            model, theta0, priors, nothing)
        chain = sample(
            sampled_model,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain,
            initial_params = InitFromPrior()
        )

        @test Turing.logjoint(sampled_model, theta0) ≈ Turing.logjoint(model, theta0) rtol = 1e-6
        @test chain isa FlexiChains.VNChain
        @test size(chain, 1) == 3
        @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
              sort(collect(keys(theta0)))

        cond_h0 = condition_turing_model(
            model, theta0, priors, (:H0,))
        chain_h0 = sample(
            cond_h0,
            Turing.NUTS(3, 0.8),
            3;
            progress = false,
            chain_type = FlexiChains.VNChain,
            initial_params = InitFromPrior()
        )
        @test chain_h0 isa FlexiChains.VNChain
        pnames = sort(collect(Symbol.(FlexiChains.parameters(chain_h0))))
        @test pnames == [:H0]
        @test all(isfinite, vec(Array(chain_h0[:logjoint])))

        prior_bounds = Dict(
            :H0 => (20.0, 140.0),
            :Ωm => (0.05, 0.95),
            :Ξ₀ => (0.5, 5.0),
            :Ξₙ => (0.05, 3.0),
            :γ => (0.5, 10.0),
            :κ => (0.05, 10.0),
            :zpeak => (0.05, 10.0)
        )
        for (sym, (low, high)) in prior_bounds
            values = vec(Array(chain[sym]))
            @test all(isfinite, values)
            @test all((low .<= values) .& (values .<= high))
        end
        @test all(isfinite, vec(Array(chain[:logjoint])))
    end
end

@testset "submodel boundary lifts VarNames to parent (flat)" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    prepared, observation = loaded.model, loaded.observation
    priors = PARITY_PRIORS

    turing_model = build_turing_model(
        prepared, loaded.fluxes, loaded.samples, loaded.fiducials, observation, priors)
    vi = VarInfo(turing_model)
    present = _varinfo_symbols(vi)
    for n in (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test n in present
    end
    @test !(:cosmo_nt in present)
    @test !(:model_nt in present)
end

@testset "condition_turing_model across submodel boundary" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    prepared, observation = loaded.model, loaded.observation
    priors = PARITY_PRIORS
    order = _PARITY_ORDER
    theta0 = PARITY_THETA

    turing_model = build_turing_model(
        prepared, loaded.fluxes, loaded.samples, loaded.fiducials, observation, priors)
    @test condition_turing_model(turing_model, theta0, priors, nothing) ===
          turing_model

    cond_Ξ₀ = condition_turing_model(turing_model, theta0, priors, (:Ξ₀,))
    sampled = _varinfo_symbols(VarInfo(cond_Ξ₀))
    @test :Ξ₀ in sampled
    for n in (:H0, :Ωm, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled)
    end

    cond_H0 = condition_turing_model(turing_model, theta0, priors, (:H0,))
    sampled_H0 = _varinfo_symbols(VarInfo(cond_H0))
    @test :H0 in sampled_H0
    for n in (:Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test !(n in sampled_H0)
    end
end

# The slim prepared model preallocates `log_weights` to the promoted element type, so it
# must stay AD-safe even with zero samples: an empty result keeps the `ForwardDiff.Dual`
# eltype rather than collapsing to `Float64[]` and breaking gradient inference downstream.
@testset "importance weights are AD-safe and shape-correct" begin
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    fiducials = _parity_hyperparameters(C, P, pop, (γ = 2.7, κ = 3.0, zpeak = 2.0))
    grid = _PARITY_FREQUENCY_GRID
    dets = [Detector("H1"), Detector("L1")]

    dual(x) = ForwardDiff.Dual{Nothing}(x, one(x))
    Λ_dual = NamedTuple{keys(fiducials)}(map(dual, values(fiducials)))

    empty_samples = (redshift = Float64[], luminosity_distance = Float64[])
    one_sample = (redshift = [0.1], luminosity_distance = [500.0])

    empty_model = prepare_parity_model(
        pop, empty_samples, fiducials, C, P, grid, dets, 1.0, 1.0).model
    one_model = prepare_parity_model(
        pop, one_sample, fiducials, C, P, grid, dets, 1.0, 1.0).model

    _, empty_lw = merger_rate_and_log_weights(empty_model, Λ_dual, empty_samples)
    _, one_lw = merger_rate_and_log_weights(one_model, Λ_dual, one_sample)

    @test isempty(empty_lw)
    @test eltype(empty_lw) <: ForwardDiff.Dual
    @test all(isfinite, one_lw)
    @test eltype(one_lw) <: ForwardDiff.Dual
end
