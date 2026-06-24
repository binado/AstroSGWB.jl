using AstroSGWB
using CBCDistributions: SampleInterpolant
using ForwardDiff
using Test

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "parity_fixtures.jl"))

const _IMP_C = LambdaCDM
const _IMP_P = ModifiedPropagation
const _IMP_POP = ParityBNSPopulation()
const _IMP_ORDER = full_hyperparameters(_IMP_C, _IMP_P, _IMP_POP)

# Construct a problem + ModelContext directly (no catalog/detectors) for type/edge tests.
function _importance_type_test_fixture(n::Integer)
    samples = (
        mass = stack_source_masses(fill(1.4, n), fill(1.2, n)),
        redshift = fill(0.1, n),
        χ₁ = fill(0.0, n),
        χ₂ = fill(0.0, n),
        Λ₁ = fill(100.0, n),
        Λ₂ = fill(100.0, n)
    )
    Λ = canonical_hyperparameters(
        _IMP_ORDER,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    problem = ImportanceSamplingProblem(_IMP_POP, zeros(2, n), samples, Λ)
    z_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = SampleInterpolant(samples.redshift, z_grid)
    obs = ObservationContext(
        [1.0, 2.0], [1.0, 1.0], [1.0, 1.0], BitVector([true, true]), 1.0)
    cache_fid = CosmologyCache(cosmology(_IMP_C, Λ), z_grid)
    proposal_prior = single_event_prior(_IMP_POP, cache_fid, Λ)
    samples_with_interp = merge(samples, (;
        redshift = SampleField(samples.redshift, interp)))
    proposal_log_prob = component_logpdfs(proposal_prior, samples_with_interp)
    ctx = ModelContext(
        proposal_prior, proposal_log_prob, ones(n), z_grid, interp, obs, 1.0)
    return problem, ctx, Λ
end

@testset "importance smoke and naive/cached parity" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    problem, C, P,
    ctx = loaded.problem, loaded.cosmology_type,
    loaded.propagation_type, loaded.ctx
    theta = PARITY_THETA

    weights_cached = compute_importance_weights(problem, C, P, theta, ctx)
    weights_naive = compute_importance_weights(problem, C, P, theta)
    @test length(weights_cached) == length(problem.samples.redshift)
    @test all(isfinite, weights_cached)
    @test weights_cached ≈ weights_naive   # R2 parity oracle

    rate_cached = merger_rate(problem, C, theta, ctx)
    rate_naive = merger_rate(
        problem, C, theta,
        ctx.local_merger_rate,
        ctx.observation.observation_time
    )
    @test isfinite(rate_cached)
    @test rate_cached ≈ rate_naive

    # Spectral-density parity: raw fluxes through the same array kernel on both paths.
    Sh_cached = spectral_density(problem.fluxes, rate_cached; weights = weights_cached)
    Sh_naive = spectral_density(problem.fluxes, rate_naive; weights = weights_naive)
    @test all(isfinite, Sh_cached)
    @test Sh_cached ≈ Sh_naive

    Λ_fid = fiducial_hyperparameters(problem)
    weights_fid = compute_importance_weights(problem, C, P, Λ_fid, ctx)
    rate_fid_ctx = merger_rate(problem, C, Λ_fid, ctx)
    Sh_fid = spectral_density(problem.fluxes, rate_fid_ctx; weights = weights_fid)
    @test fiducial_spectral_density(problem, C, P, ctx) ≈ Sh_fid

    rate_fid = merger_rate(
        ctx.proposal_prior,
        ctx.local_merger_rate,
        ctx.observation.observation_time
    )
    if Λ_fid.Ξ₀ != 1.0
        @test !(spectral_density(problem.fluxes, rate_fid) ≈ Sh_fid)
    end
end

@testset "empty importance weights preserve AD element types" begin
    dual(x) = ForwardDiff.Dual{Nothing}(x, one(x))
    theta = (
        H0 = dual(67.0),
        Ωm = dual(0.315),
        Ξ₀ = dual(1.0),
        Ξₙ = dual(0.0),
        γ = dual(2.7),
        κ = dual(3.0),
        zpeak = dual(2.5)
    )

    empty_problem, empty_ctx, _ = _importance_type_test_fixture(0)
    populated_problem, populated_ctx, _ = _importance_type_test_fixture(1)

    empty_weights = compute_importance_weights(
        empty_problem, _IMP_C, _IMP_P, theta, empty_ctx)
    populated_weights = compute_importance_weights(
        populated_problem, _IMP_C, _IMP_P, theta, populated_ctx)

    @test isempty(empty_weights)
    @test all(isfinite, populated_weights)
    @test eltype(populated_weights) <: ForwardDiff.Dual
end

@testset "fiducial observed spectrum applies modified-propagation weights" begin
    loaded = parity_problem_context(:importance_context, [Detector("H1"), Detector("L1")])
    C = loaded.cosmology_type
    P = loaded.propagation_type
    problem = loaded.problem
    Λ_fid = merge(fiducial_hyperparameters(problem), (Ξ₀ = 1.2,))
    problem = ImportanceSamplingProblem(
        problem.population_model, problem.fluxes, problem.samples, Λ_fid)
    grid = FrequencyGrid(0.05, 80.0, 20.0, 15.0, 40.0)
    ctx = build_model_context(
        problem, C, grid, [Detector("H1"), Detector("L1")], 1.0, 161.0)
    Sh_fid = fiducial_spectral_density(problem, C, P, ctx)
    rate_fid = merger_rate(
        ctx.proposal_prior,
        ctx.local_merger_rate,
        ctx.observation.observation_time
    )
    @test !(spectral_density(problem.fluxes, rate_fid) ≈ Sh_fid)
end
