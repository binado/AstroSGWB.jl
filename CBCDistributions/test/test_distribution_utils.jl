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

@testset "component_logpdfs matches batched_logpdf component sums" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5, 0.7], y = [1.1, 1.7, 0.2])

    lps = component_logpdfs(prior, samples)
    @test keys(lps) == keys(prior.dists)
    @test lps.x ≈ logpdfvec(prior.dists.x, samples.x)
    @test lps.y ≈ logpdfvec(prior.dists.y, samples.y)
    @test lps.x .+ lps.y ≈ batched_logpdf(prior, samples)
end

@testset "component_logpdfs rejects mismatched sample field lengths" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    prior = single_event_prior(pop, cosmo, Λ)
    samples = (x = [0.1, 0.5], y = [1.1])

    @test_throws ArgumentError component_logpdfs(prior, samples)
end

@testset "logprobdiff egal skip rejects out-of-support proposal logpdfs" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    samples = (x = [0.1, 1.5], y = [1.1, 1.7])  # x[2] outside Uniform(0, α)

    proposal = single_event_prior(pop, cosmo, Λ)
    proposal_logprob = component_logpdfs(proposal, samples)
    @test !all(isfinite, proposal_logprob.x)

    prior_same = single_event_prior(pop, cosmo, Λ)
    diff = logprobdiff(pop, prior_same, proposal, proposal_logprob, samples)
    ref = batched_logpdf(prior_same, samples) .- batched_logpdf(proposal, samples)
    @test diff[1] ≈ ref[1]
    @test isnan(diff[2]) && isnan(ref[2])
end

@testset "logprobdiff default: egal skip and two-sided reference" begin
    pop = TestPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ_fid = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.6, β = 1.4)
    samples = (x = [0.1, 0.5], y = [1.1, 0.3])

    proposal = single_event_prior(pop, cosmo, Λ_fid)
    proposal_logprob = component_logpdfs(proposal, samples)

    # Same hyperparameters: every component is egal to the proposal's, so the diff is
    # exactly zero (skipped, never evaluated).
    prior_same = single_event_prior(pop, cosmo, Λ_fid)
    @test prior_same.dists.x === proposal.dists.x
    @test logpdfdiffvec(
        pop, Val(:x), prior_same.dists.x, proposal.dists.x,
        proposal_logprob.x, samples.x) === nothing
    diff_same = logprobdiff(pop, prior_same, proposal, proposal_logprob, samples)
    @test diff_same == zeros(length(samples.x))

    # Different hyperparameters: nothing is egal, so the diff matches the full
    # two-sided batched_logpdf reference.
    prior = single_event_prior(pop, cosmo, Λ)
    diff = logprobdiff(pop, prior, proposal, proposal_logprob, samples)
    @test diff ≈ batched_logpdf(prior, samples) .- batched_logpdf(proposal, samples)

    # Convenience wrapper (proposal logpdfs computed on the fly) agrees.
    @test logprobdiff(pop, prior, proposal, samples) ≈ diff
end

@testset "logprobdiff per-component vector override via Val{key}" begin
    pop = SkipYPop()
    cosmo = LambdaCDM(67.0, 0.315)
    Λ_fid = (H0 = 67.0, Ωm = 0.315, α = 0.8, β = 1.8)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.6, β = 1.4)
    samples = (x = [0.1, 0.5], y = [1.1, 0.3])

    proposal = single_event_prior(pop, cosmo, Λ_fid)
    proposal_logprob = component_logpdfs(proposal, samples)
    prior = single_event_prior(pop, cosmo, Λ)

    # Only the :x component contributes; :y is skipped by the override even though the
    # target and proposal distributions differ.
    @test logpdfdiffvec(
        pop, Val(:y), prior.dists.y, proposal.dists.y,
        proposal_logprob.y, samples.y) === nothing
    diff = logprobdiff(pop, prior, proposal, proposal_logprob, samples)
    x_only = component_logpdfs(prior, samples).x .- proposal_logprob.x
    @test diff ≈ x_only
end
