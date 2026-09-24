using ForwardDiff
using QuadGK
using Random
using Distributions: insupport, logpdf

# Flat-ΛCDM reference for the volume grid, computed inline so the tests stay
# standalone: GWDistributions takes volume arrays, not cosmology types, and
# BackgroundCosmology is deliberately not a dependency — not even a test one.
# The fixed Gauss–Legendre rule (order 40, matching BackgroundCosmology's
# scalar distance rule) keeps the helper ForwardDiff-compatible.
const _TEST_H0 = 67.0
const _TEST_DH_MPC = 299_792.458 / _TEST_H0
const _TEST_GAUSS_LEGENDRE_40 = QuadGK.gauss(Float64, 40)

_test_E(z, Ωm) = sqrt(Ωm * (1 + z)^3 + 1 - Ωm)

function _test_comoving_distance(z::Real, Ωm::Real)
    nodes, weights = _TEST_GAUSS_LEGENDRE_40
    half = z / 2
    acc = zero(half * inv(_test_E(half, Ωm)))
    for k in eachindex(nodes)
        acc += weights[k] * inv(_test_E(half * (1 + nodes[k]), Ωm))
    end
    return _TEST_DH_MPC * half * acc
end

# Solid-angle-integrated dV_c/dz in Mpc³ per unit redshift, matching
# BackgroundCosmology.differential_comoving_volume for the same (H0, Ωm).
function _test_differential_comoving_volume(z::Real, Ωm::Real)
    4π * _TEST_DH_MPC * _test_comoving_distance(z, Ωm)^2 / _test_E(z, Ωm)
end

function _madau_dickinson_with_denom_exp(z, γ, denom_exp, zpeak)
    one_plus_z = 1 + z
    return ((one_plus_z^γ) / (1 + (one_plus_z / (1 + zpeak))^denom_exp)) *
           (1 + (1 + zpeak)^(-denom_exp))
end

@testset "Madau–Dickinson κ reparametrization" begin
    γ, κ, zpeak = 2.7, 3.0, 2.0
    denom_exp = γ + κ
    z_samples = [0.0, 0.5, zpeak, 3.0]

    @test madau_dickinson_source_frame_distribution(0.0; γ, κ, zpeak) ≈ 1.0
    for z in z_samples
        @test madau_dickinson_source_frame_distribution(z; γ, κ, zpeak) ≈
              _madau_dickinson_with_denom_exp(z, γ, denom_exp, zpeak)
    end
    R₀ = 161.0
    model = MadauDickinsonSourceFrame(; γ, κ, zpeak, R₀)
    amp = 1.0e-9 * R₀ / JULIAN_YEAR_SEC
    @test source_frame_distribution(model, 1.0) ≈
          amp * madau_dickinson_source_frame_distribution(1.0; γ, κ, zpeak)
end

@testset "redshift prior from volume grid" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5, R₀ = 161.0)
    Ωm = 0.315
    z_grid = collect(LinRange(0.0, 2.0, 101))
    source_model = MadauDickinsonSourceFrame(
        γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀)

    # Volume-array constructor (cosmology-decoupled, typed source frame)
    dvc = _test_differential_comoving_volume.(z_grid, Ωm)
    distribution = RedshiftInterpolatedDistribution(source_model, dvc, z_grid)
    expected = @. dvc * source_frame_distribution(source_model, z_grid) / (1 + z_grid)
    @test distribution.dist.x == z_grid
    @test distribution.dist.y ≈ expected
    @test normalizer(distribution) ===
          GWDistributions._trapz(distribution.dist.y, distribution.dist.x)

    # Shape-only integral × amplitude recovers the same normalizer
    shape = madau_dickinson_source_frame_distribution.(
        z_grid; γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
    shape_y = @. dvc * shape / (1 + z_grid)
    @test normalizer(distribution) ≈
          (1.0e-9 * Λ.R₀ / JULIAN_YEAR_SEC) * GWDistributions._trapz(shape_y, z_grid)

    # Wrap-inner constructor
    wrapped = RedshiftInterpolatedDistribution(Interpolated1DDistribution(z_grid, expected))
    @test normalizer(wrapped) ≈ normalizer(distribution)

    @test minimum(distribution) == first(z_grid)
    @test maximum(distribution) == last(z_grid)
    @test isfinite(logpdf(distribution, 0.5))
    @test logpdf(distribution, -0.1) == -Inf
    @test logpdf(distribution, 2.1) == -Inf

    samples = rand(MersenneTwister(1234), distribution, 100)
    @test all(x -> insupport(distribution, x), samples)
end

@testset "redshift prior preserves AD" begin
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5, R₀ = 161.0)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    f = Ωm -> begin
        source_model = MadauDickinsonSourceFrame(
            γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀)
        dvc = _test_differential_comoving_volume.(z_grid, Ref(Ωm))
        distribution = RedshiftInterpolatedDistribution(source_model, dvc, z_grid)
        normalizer(distribution)
    end
    derivative = ForwardDiff.derivative(f, 0.315)
    @test isfinite(derivative)
    @test derivative != 0.0
end
