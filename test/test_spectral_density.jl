using Statistics
using Test

@testset "spectral_density primitive" begin
    fluxes = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    rate = 2.5
    n_samples = size(fluxes, 2)

    @testset "unweighted average over samples" begin
        expected = 0.4 .* rate .* vec(mean(fluxes; dims=2))
        @test spectral_density(fluxes, rate) ≈ expected
    end

    @testset "weighted contraction without weight normalization" begin
        w = [0.5, 1.0, 2.0]
        expected = 0.4 .* rate .* (fluxes * w) ./ n_samples
        @test spectral_density(fluxes, rate; weights=w) ≈ expected
    end

    @testset "uniform weights equal to ones give the unweighted mean" begin
        w = ones(n_samples)
        @test spectral_density(fluxes, rate; weights=w) ≈ spectral_density(fluxes, rate)
    end

    @testset "length mismatch raises" begin
        @test_throws ArgumentError spectral_density(fluxes, rate; weights=[1.0, 2.0])
    end

    @testset "output length matches n_freq" begin
        @test length(spectral_density(fluxes, rate)) == size(fluxes, 1)
        @test length(spectral_density(fluxes, rate; weights=rand(n_samples))) ==
              size(fluxes, 1)
    end
end
