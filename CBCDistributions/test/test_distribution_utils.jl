using Test
using Distributions
using CBCDistributions

@testset "validate_samples" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5, 0.7], y = [1.1, 1.7, 0.2], extra = [0.0, 0.0, 0.0])

    @test validate_samples(prior, samples) == 3

    @test_throws ArgumentError validate_samples(prior, (x = [0.1, 0.5],))
    @test_throws ArgumentError validate_samples(
        prior, (x = [0.1, 0.5], y = [1.1]))

    wrapped = (
        x = SampleField([0.1, 0.5], 42),
        y = [1.1, 1.7]
    )
    @test validate_samples(prior, wrapped) == 2

    mass_prior = product_distribution((
        mass = OrderedUniformSourceMassPair(low = 1.0, high = 2.0),
        redshift = Uniform(0.0, 1.0)
    ))
    mass_samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2]
    )
    @test validate_samples(mass_prior, mass_samples) == 2
end

@testset "batched_logpdf on ProductNamedTupleDistribution" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5], y = [1.1, 1.7])
    lp = batched_logpdf(prior, samples)
    @test length(lp) == 2
    @test all(isfinite, lp)

    lp_ref = [logpdf(prior, (; x = samples.x[i], y = samples.y[i])) for i in 1:2]
    @test lp ≈ lp_ref
end
