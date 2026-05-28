using ASGWB
using ForwardDiff
using Test

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "parity_fixtures.jl"))

const _IMP_C = ModifiedPropagation{LambdaCDM}
const _IMP_POP = ParityBNSPopulation()
const _IMP_ORDER = full_hyperparameters(_IMP_C, _IMP_POP)

function _importance_type_test_problem(n::Integer)
    samples = (
        mass = stack_source_masses(fill(1.4, n), fill(1.2, n)),
        redshift = fill(0.1, n),
        χ₁ = fill(0.0, n),
        χ₂ = fill(0.0, n),
        Λ₁ = fill(100.0, n),
        Λ₂ = fill(100.0, n)
    )
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        zeros(n),
        zeros(n, length(FULL_BNS_INTRINSIC_ORDER)),
        zeros(2, n),
        ones(n)
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        BitVector([true, true]),
        [0.0, 0.0],
        1.0,
        1.0
    )
    Λ = canonical_hyperparameters(
        _IMP_ORDER,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    return importance_sampling_problem(proposal, observation, _IMP_C, _IMP_POP, Λ, 1.0)
end

@testset "importance smoke" begin
    cache = parity_load_problem(:posterior, [Detector("H1"), Detector("L1")])
    theta = PARITY_THETA

    model_evaluation = evaluate_model_terms(theta, cache)
    @test length(model_evaluation.weights) == length(cache.proposal.samples.redshift)
    @test all(isfinite, model_evaluation.weights)
    @test all(isfinite, model_evaluation.log_ratio)
    @test all(isfinite, model_evaluation.target_log_prob)
    @test all(isfinite, model_evaluation.spectral_density)
    @test isfinite(model_evaluation.redshift_integral)
    @test isfinite(model_evaluation.expected_number_of_sources)

    c = cosmology(cache.cosmology_type, theta)
    cosmology_cache = CosmologyCache(c, cache.redshift_grid)
    prior = single_event_prior(cache.population, c, theta)
    iw = compute_importance_weights(cache, theta, cosmology_cache, prior)
    @test iw.weights ≈ model_evaluation.weights
    @test iw.log_ratio ≈ model_evaluation.log_ratio
    @test iw.target_log_prob ≈ model_evaluation.target_log_prob
    @test iw.dgw_theta_sq ≈ model_evaluation.dgw_theta_sq

    rate = merger_rate_per_sec(
        prior.dists.redshift.prior,
        cache.local_merger_rate,
        cache.observation.observation_time_yr,
        cache.observation.observation_time_sec
    )
    @test isfinite(rate)
    @test model_evaluation.expected_number_of_sources ≈
          rate * cache.observation.observation_time_sec
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

    empty_problem = _importance_type_test_problem(0)
    populated_problem = _importance_type_test_problem(1)
    c_dual = cosmology(_IMP_C, theta)
    empty_cosmology_cache = CosmologyCache(c_dual, empty_problem.redshift_grid)
    empty_prior = single_event_prior(_IMP_POP, c_dual, theta)
    populated_cosmology_cache = CosmologyCache(c_dual, populated_problem.redshift_grid)
    populated_prior = single_event_prior(_IMP_POP, c_dual, theta)

    empty_iw = compute_importance_weights(
        empty_problem, theta, empty_cosmology_cache, empty_prior)
    populated_iw = compute_importance_weights(
        populated_problem, theta, populated_cosmology_cache, populated_prior)

    @test isempty(empty_iw.weights)
    @test all(isfinite, populated_iw.weights)
    @test all(isfinite, populated_iw.log_ratio)
    @test all(isfinite, populated_iw.target_log_prob)
    @test all(isfinite, populated_iw.dgw_theta_sq)
end
