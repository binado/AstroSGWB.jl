using AstroSGWB
using CBCDistributions: GridQuery
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

# Construct a problem + prepared model directly (no catalog/detectors) for type/edge tests.
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
    interp = GridQuery(samples.redshift, z_grid)
    cache_fid = CosmologyCache(cosmology(_IMP_C, Λ), z_grid)
    proposal_prior = single_event_prior(_IMP_POP, cache_fid, Λ)
    samples_interp = with_redshift_interpolant(samples, interp)
    proposal_log_prob = component_logpdfs(proposal_prior, samples_interp)
    model = PreparedParityModel{_IMP_C, _IMP_P, typeof(_IMP_POP),
        typeof(proposal_prior), typeof(proposal_log_prob)}(
        _IMP_POP, z_grid, interp, proposal_prior, proposal_log_prob, ones(n), 1.0, 1.0)
    return problem, model, Λ
end

@testset "prepared-model joint: rate + log weights + spectral density" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    problem, model = loaded.problem, loaded.model
    theta = PARITY_THETA

    rate, log_weights = merger_rate_and_log_weights(model, theta, problem.samples)
    @test length(log_weights) == length(problem.samples.redshift)
    @test all(isfinite, log_weights)
    @test isfinite(rate)

    weights = exp.(log_weights)
    Sh = spectral_density(problem.fluxes, rate; weights = weights)
    @test all(isfinite, Sh)

    # Fiducial spectrum reconstructs the same forward model at the fiducial point.
    Λ_fid = fiducial_hyperparameters(problem)
    rate_fid, log_weights_fid = merger_rate_and_log_weights(model, Λ_fid, problem.samples)
    Sh_fid = spectral_density(problem.fluxes, rate_fid; weights = exp.(log_weights_fid))
    @test fiducial_spectral_density(model, problem) ≈ Sh_fid

    rate_fid_unweighted = merger_rate(
        model.proposal_prior, model.local_merger_rate, model.observation_time)
    if Λ_fid.Ξ₀ != 1.0
        @test !(spectral_density(problem.fluxes, rate_fid_unweighted) ≈ Sh_fid)
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

    empty_problem, empty_model, _ = _importance_type_test_fixture(0)
    populated_problem, populated_model, _ = _importance_type_test_fixture(1)

    _,
    empty_log_weights = merger_rate_and_log_weights(
        empty_model, theta, empty_problem.samples)
    _,
    populated_log_weights = merger_rate_and_log_weights(
        populated_model, theta, populated_problem.samples)

    @test isempty(empty_log_weights)
    @test all(isfinite, populated_log_weights)
    @test eltype(populated_log_weights) <: ForwardDiff.Dual
end

@testset "fiducial observed spectrum applies modified-propagation weights" begin
    loaded = parity_problem_context(:importance_context, [Detector("H1"), Detector("L1")])
    problem = loaded.problem
    Λ_fid = merge(fiducial_hyperparameters(problem), (Ξ₀ = 1.2,))
    problem = ImportanceSamplingProblem(
        problem.population_model, problem.fluxes, problem.samples, Λ_fid)
    grid = FrequencyGrid(0.05, 80.0, 20.0, 15.0, 40.0)
    prepared = prepare_parity_model(
        problem, loaded.cosmology_type, loaded.propagation_type,
        grid, [Detector("H1"), Detector("L1")], 1.0, 161.0)
    model = prepared.model
    Sh_fid = fiducial_spectral_density(model, problem)
    rate_fid = merger_rate(
        model.proposal_prior, model.local_merger_rate, model.observation_time)
    @test !(spectral_density(problem.fluxes, rate_fid) ≈ Sh_fid)
end
