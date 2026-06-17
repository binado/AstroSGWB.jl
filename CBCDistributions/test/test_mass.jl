using Distributions
using ForwardDiff
using QuadGK
using Random

function _default_bbh_pair(; kwargs...)
    params = merge(
        (
            α1 = 1.5,
            α2 = 4.0,
            m_break = 35.0,
            μ1 = 10.0,
            σ1 = 2.0,
            μ2 = 35.0,
            σ2 = 6.0,
            m1_low = 5.0,
            δm1 = 4.0,
            λ0 = 0.55,
            λ1 = 0.25,
            βq = 1.2,
            m2_low = 4.0,
            δm2 = 3.0,
            m_high = 120.0
        ),
        kwargs
    )
    return DefaultBBHMassPair(; params...)
end

@testset "Smootherstep taper boundary behavior" begin
    @test smoothstep_taper(4.9, 5.0, 2.0) == 0.0
    @test smoothstep_taper(5.0, 5.0, 2.0) == 0.0
    @test 0.0 < smoothstep_taper(6.0, 5.0, 2.0) < 1.0
    @test smoothstep_taper(6.0, 5.0, 2.0)≈0.5 atol=1e-12  # symmetric midpoint
    @test smoothstep_taper(7.0, 5.0, 2.0) == 1.0
    @test smoothstep_taper(5.0, 5.0, 0.0) == 1.0
    @test isfinite(smoothstep_taper(5.0 + eps(), 5.0, 2.0))
    @test isfinite(smoothstep_taper(7.0 - eps(), 5.0, 2.0))
end

@testset "DefaultBBHPrimaryMass normalization" begin
    d = DefaultBBHPrimaryMass(;
        α1 = 1.0,
        α2 = 1.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 0.0,
        λ0 = 1.0,
        λ1 = 0.0,
        m_high = 120.0
    )

    z, _ = quadgk(m -> pdf(d, m), minimum(d), maximum(d))
    @test z≈1.0 rtol=1e-8

    tapered = DefaultBBHPrimaryMass(;
        α1 = 1.5,
        α2 = 4.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 4.0,
        λ0 = 0.55,
        λ1 = 0.25,
        m_high = 120.0
    )
    z_tapered, _ = quadgk(m -> pdf(tapered, m), minimum(tapered), maximum(tapered))
    @test z_tapered≈1.0 rtol=1e-7
end

@testset "DefaultBBHMassPair logpdf and conditional normalization" begin
    d = _default_bbh_pair()
    @test isfinite(logpdf(d, (35.0, 25.0)))
    @test logpdf(d, (35.0, 36.0)) == -Inf
    @test logpdf(d, (4.9, 4.5)) == -Inf
    @test logpdf(d, (121.0, 40.0)) == -Inf
    @test_throws ArgumentError _default_bbh_pair(λ0 = 0.8, λ1 = 0.3)

    m1 = 35.0
    logp_m1 = logpdf(d.primary, m1)
    z_conditional,
    _ = quadgk(m2 -> exp(logpdf(d, (m1, m2)) - logp_m1), d.m2_low, m1)
    @test z_conditional≈1.0 rtol=1e-7
end

@testset "DefaultBBHMassPair batched logpdf" begin
    d = _default_bbh_pair()
    prior = product_distribution((mass = d,))
    samples = (mass = [35.0 40.0 60.0; 25.0 38.0 20.0],)

    batched = batched_logpdf(prior, samples)
    scalar = [logpdf(d, (samples.mass[1, i], samples.mass[2, i]))
              for i in axes(samples.mass, 2)]
    @test batched ≈ scalar
end

@testset "DefaultBBHMassPair rand support and shape" begin
    rng = MersenneTwister(1234)
    d = _default_bbh_pair()
    draws = rand(rng, d, 256)

    @test size(draws) == (2, 256)
    @test all(draws[1, :] .>= draws[2, :])
    @test all(draws[1, :] .>= d.primary.m1_low)
    @test all(draws[1, :] .< d.primary.m_high)
    @test all(draws[2, :] .>= d.m2_low)
    @test all(isfinite, [logpdf(d, (draws[1, i], draws[2, i])) for i in axes(draws, 2)])

    low_draws = rand(rng, _default_bbh_pair(δm1 = 8.0, δm2 = 8.0), 2_000)
    @test any(low_draws[1, :] .< 13.0)
    @test count(low_draws[1, :] .< 7.0) < count(low_draws[1, :] .< 13.0)
end

@testset "DefaultBBHMassPair interior logpdf is differentiable in m2" begin
    d = _default_bbh_pair()
    grad = ForwardDiff.gradient(x -> logpdf(d, (35.0, x[1])), [25.0])
    @test all(isfinite, grad)

    # A point inside the secondary taper band [m2_low, m2_low + δm2] exercises the
    # smootherstep window itself, not just the flat region above it.
    @test d.m2_low < 5.5 < d.m2_low + d.δm2
    grad_band = ForwardDiff.gradient(x -> logpdf(d, (35.0, x[1])), [5.5])
    @test all(isfinite, grad_band)
end

@testset "DefaultBBHMassPair _q_normalizer matches direct quadrature" begin
    # The reference uses a tight rtol: the smootherstep integrand is only C² at the band
    # joins, so quadgk's default √eps tolerance is looser than the agreement we expect from
    # the closed form (which is exact to ~1e-15 against a BigFloat reference).
    for βq in (1.2, 2.5, 0.0, -1.0, -2.0), m1 in (10.0, 35.0, 80.0)
        d = _default_bbh_pair(βq = βq)
        q_low = d.m2_low / m1
        ref, _ = quadgk(
            q -> q^βq * smoothstep_taper(q * m1, d.m2_low, d.δm2), q_low, 1.0; rtol = 1e-12)
        @test CBCDistributions._q_normalizer(d, m1)≈ref rtol=1e-9
    end

    # Untapered limit reduces to the closed-form power integral.
    d0 = _default_bbh_pair(δm2 = 0.0)
    q_low = d0.m2_low / 35.0
    ref0, _ = quadgk(q -> q^d0.βq, q_low, 1.0; rtol = 1e-12)
    @test CBCDistributions._q_normalizer(d0, 35.0)≈ref0 rtol=1e-10
end
