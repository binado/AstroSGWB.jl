using ASGWB
using ASGWB: Detector, PowerSpectralDensity, covariance_on_grid, gaussian_bin_scale,
    frequency_bin_width, build_observation_config, default_detector_data_dir
using HDF5
using Test

@testset "PowerSpectralDensity file load" begin
    noise = joinpath(default_detector_data_dir(), "noise_curves", "AplusDesign_psd.txt")
    psd = PowerSpectralDensity(noise; curve_type=:psd)
    v = psd([20.0, 100.0, 1e6])
    @test isfinite(v[1]) && isfinite(v[2])
    @test v[3] == Inf
end

@testset "Detector from vendored TOML" begin
    d = Detector("H1")
    @test d.name == "H1"
    @test d.length == 4.0
    @test_throws ArgumentError Detector("NONEXISTENT_DETECTOR_XYZ")
end

@testset "covariance_on_grid and gaussian_bin_scale" begin
    d1 = Detector("H1")
    d2 = Detector("L1")
    f = collect(range(20.0; step=20.0, length=16))
    cov = covariance_on_grid(f, [d1, d2])
    @test length(cov) == length(f)
    @test all(isfinite, cov) && all(cov .> 0)
    mask = trues(length(f))
    scale = gaussian_bin_scale(;
        covariance=cov,
        frequencies=f,
        in_band_mask=mask,
        observation_time_sec=3.15576e7,
    )
    @test length(scale) == count(mask)
    @test all(isfinite, scale) && all(scale .> 0)
end

@testset "load_cache format v2 without covariance (reconstruct with detectors)" begin
    path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia_v2_minimal.h5")
    isfile(path) || error("missing fixture $path")
    d1 = Detector("H1")
    d2 = Detector("L1")
    p = load_cache(path; detectors=[d1, d2])
    @test length(p.observation.covariance) == length(p.observation.frequencies)
    @test all(isfinite, p.observation.covariance)
    @test length(p.observation.sgwb_scale) == length(p.observation.frequencies)
end

@testset "load_cache v2 parity with v1 when datasets present" begin
    v1 = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    p1 = load_cache(v1)
    p2 = load_cache(v1; detectors=[Detector("H1"), Detector("L1")])
    @test p1.observation.covariance == p2.observation.covariance
    @test p1.observation.sgwb_scale == p2.observation.sgwb_scale
end
