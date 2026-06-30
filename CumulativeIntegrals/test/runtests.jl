using Test
using ForwardDiff
using CumulativeIntegrals
using CumulativeIntegrals: CumulativeIntegral1D, GridQuery, interpolate, cdf, normalizer

@testset "CumulativeIntegral1D" begin
    @testset "analytic linear antiderivative on smooth integrand" begin
        x = collect(LinRange(0.0, 2π, 513))
        r = CumulativeIntegral1D(x, sin)
        @test isapprox(normalizer(r), 0.0; atol = 1e-10)
        @test isapprox(cdf(r, π), 2.0; rtol = 1e-4)
        @test interpolate(r, π / 2) ≈ sin(π / 2) atol = 1e-8
        @test_throws Exception interpolate(r, 2π + 0.1)
        @test cdf(r, -1.0) == 0.0
        @test cdf(r, 2π + 0.1) == normalizer(r)
    end

    @testset "cdf uses exact within-cell linear antiderivative" begin
        x = [0.0, 1.0, 2.0]
        r = CumulativeIntegral1D(x, z -> 2.0 + 3.0z)
        @test cdf(r, 0.25) ≈ 2.0 * 0.25 + 0.5 * 3.0 * 0.25^2
        @test cdf(r, 1.5) ≈ cdf(r, 1.0) + 1.0 * (5.0 * 0.5 + 0.5 * 3.0 * 0.5^2)
    end

    @testset "from-values constructor matches function constructor" begin
        x = collect(LinRange(0.0, 10.0, 256))
        f = w -> inv(1 + w)
        from_fn = CumulativeIntegral1D(x, f)
        from_vals = CumulativeIntegral1D(x, map(f, x))
        @test from_vals.y == from_fn.y
        @test from_vals.cumulative == from_fn.cumulative
        @test_throws ArgumentError CumulativeIntegral1D([0.0], [1.0])
        @test_throws ArgumentError CumulativeIntegral1D(x, [1.0, 2.0])
    end

    @testset "ForwardDiff Duals propagate through nodal values" begin
        x = collect(LinRange(0.0, 10.0, 257))
        f = a -> begin
            r = CumulativeIntegral1D(x, map(w -> a / (1 + w), x))
            cdf(r, 1.2)
        end
        @test isfinite(ForwardDiff.derivative(f, 2.0))
        @test ForwardDiff.derivative(f, 2.0) ≈ f(1.0)
    end
end

@testset "GridQuery batched accessors" begin
    x = collect(LinRange(0.0, 2.0, 101))
    r = CumulativeIntegral1D(x, z -> 1.0 + z^2)
    samples = [0.0, 0.137, 0.9, 2.0]
    query = GridQuery(samples, x)

    @testset "batched interpolate/cdf match scalar verbs" begin
        @test [interpolate(r, query, i) for i in eachindex(samples)] ≈
              [interpolate(r, z) for z in samples]
        @test [cdf(r, query, i) for i in eachindex(samples)] ≈
              [cdf(r, z) for z in samples]
    end

    @testset "out-of-grid points throw" begin
        @test_throws ArgumentError GridQuery([-0.1], x)
        @test_throws ArgumentError GridQuery([2.1], x)
    end
end
