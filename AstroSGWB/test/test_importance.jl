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

# Construct explicit catalog inputs + prepared model directly (no catalog/detectors) for
# type/edge tests.
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
    fluxes = zeros(2, n)
    z_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = GridQuery(samples.redshift, z_grid)
    cache_fid = CosmologyCache(cosmology(_IMP_C, Λ), z_grid)
    proposal_prior = single_event_prior(_IMP_POP, cache_fid, Λ)
    samples_interp = with_redshift_interpolant(samples, interp)
    proposal_log_prob = component_logpdfs(proposal_prior, samples_interp)
    model = PreparedParityModel{_IMP_C, _IMP_P, typeof(_IMP_POP),
        typeof(proposal_prior), typeof(proposal_log_prob)}(
        _IMP_POP, z_grid, interp, proposal_prior, proposal_log_prob, ones(n), 1.0, 1.0)
    return fluxes, samples, model, Λ
end

@testset "importance weights feed spectral-density kernel" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    model = loaded.model
    log_weights = importance_log_weights(
        zeros(length(loaded.samples.redshift)),
        model.dl_fid_sq,
        loaded.samples.redshift,
        model.sample_interpolant,
        CosmologyCache(cosmology(_IMP_C, loaded.fiducials), model.redshift_grid),
        propagation(_IMP_P, loaded.fiducials)
    )
    @test length(log_weights) == length(loaded.samples.redshift)
    @test all(isfinite, log_weights)

    weights = exp.(log_weights)
    Sh = spectral_density(loaded.fluxes, 1.0; weights = weights)
    @test all(isfinite, Sh)
end

function _direct_importance_log_weights(model::PreparedParityModel{C, P}, Λ, samples) where {
        C, P}
    cache = CosmologyCache(cosmology(C, Λ), model.redshift_grid)
    prior = single_event_prior(model.pop, cache, Λ)
    samples_interp = with_redshift_interpolant(samples, model.sample_interpolant)
    log_ratio = logprobdiff(
        model.pop, prior, model.proposal_prior, model.proposal_log_prob, samples_interp)
    return importance_log_weights(
        log_ratio, model.dl_fid_sq, samples.redshift,
        model.sample_interpolant, cache, propagation(P, Λ))
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

    _, empty_samples, empty_model, _ = _importance_type_test_fixture(0)
    _, populated_samples, populated_model, _ = _importance_type_test_fixture(1)

    empty_log_weights = _direct_importance_log_weights(empty_model, theta, empty_samples)
    populated_log_weights = _direct_importance_log_weights(populated_model, theta, populated_samples)

    @test isempty(empty_log_weights)
    @test all(isfinite, populated_log_weights)
    @test eltype(populated_log_weights) <: ForwardDiff.Dual
end
