using HDF5
using Distributions: Uniform, logpdf
using Test
using AstroSGWB

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

function _load_variant(variant::Symbol)
    return parity_problem_context(variant, _TEST_LOAD_DETS)
end

@testset "save_catalog/load_catalog round-trip" begin
    grid = FrequencyGrid(1.0, 4.0, 2.0, 1.0, 2.0)
    samples = (
        mass_1_source = [1.4, 1.4],
        mass_2_source = [1.2, 1.2],
        redshift = [0.1, 0.2],
        chi_1 = [0.0, 0.0],
        chi_2 = [0.0, 0.0],
        lambda_1 = [100.0, 100.0],
        lambda_2 = [100.0, 100.0],
        luminosity_distance = [430.0, 880.0]
    )
    fluxes = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    meta = WaveformCatalogMetadata("IMRPhenomPV2", :BNS, grid, "rev", "cmd")
    catalog = WaveformCatalog(samples, fluxes)
    file = WaveformCatalogFile(catalog, meta)

    path, io = mktemp()
    close(io)
    try
        save_catalog(path, file)
        loaded = load_catalog(path)
        @test Set(keys(loaded.catalog.samples)) == Set(keys(catalog.samples))
        for k in keys(catalog.samples)
            @test loaded.catalog.samples[k] ≈ catalog.samples[k]
        end
        @test loaded.catalog.fluxes ≈ catalog.fluxes
        @test loaded.metadata.approximant == meta.approximant
        @test loaded.metadata.source_type == meta.source_type
        @test loaded.metadata.grid.duration == grid.duration
        @test loaded.metadata.grid.sampling_frequency == grid.sampling_frequency
        @test loaded.metadata.grid.minimum_frequency == grid.minimum_frequency
        @test loaded.metadata.grid.maximum_frequency == grid.maximum_frequency
    finally
        rm(path; force = true)
    end
end

@testset "FrequencyGrid validation" begin
    @test FrequencyGrid(1.0, 4.0, 2.0, 1.0).maximum_frequency == 2.0

    @test_throws ArgumentError FrequencyGrid(0.0, 4.0, 2.0, 1.0, 2.0)
    @test_throws ArgumentError FrequencyGrid(1.0, 0.0, 2.0, 1.0, 2.0)
    @test_throws ArgumentError FrequencyGrid(1.0, 4.0, 2.0, -1.0, 2.0)
    @test_throws ArgumentError FrequencyGrid(1.0, 4.0, 2.0, 2.0, 2.0)
    @test_throws ArgumentError FrequencyGrid(1.0, 4.0, 2.0, 1.0, 3.0)
end

@testset "WaveformCatalog shape validation" begin
    @test WaveformCatalog((redshift = [0.1, 0.2],), zeros(3, 2)) isa WaveformCatalog
    @test WaveformCatalog((x = [1.0], y = [2.0]), zeros(2, 1)) isa WaveformCatalog
    @test_throws ArgumentError WaveformCatalog((x = [1.0], y = [2.0, 3.0]), zeros(2, 2))
    @test_throws ArgumentError WaveformCatalog((redshift = [0.1, 0.2],), zeros(3, 3))
end

@testset "catalog inputs are explicit" begin
    loaded = _load_variant(:importance_context)

    @test loaded.pop isa ParityBNSPopulation
    @test redshift(loaded.samples) ≈ [0.1, 0.2]
    @test Vector(loaded.samples.mass[1, :]) ≈ [1.4, 1.4]
    @test Vector(loaded.samples.mass[2, :]) ≈ [1.2, 1.2]
    @test loaded.samples.χ₁ ≈ [0.0, 0.0]
    @test loaded.samples.Λ₁ ≈ [100.0, 100.0]
    @test loaded.fluxes ≈ Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]

    Λ = loaded.fiducials
    @test Λ.H0 == 67.0
    @test Λ.Ωm == 0.315
    @test Λ.Ξ₀ == 1.0
    @test Λ.γ == 2.7
end

@testset "prepared model reconstructs derived caches" begin
    loaded = _load_variant(:importance_context)
    model = loaded.model
    observation = loaded.observation

    @test keys(model.proposal_log_prob) == keys(model.proposal_prior.dists)
    @test all(lp -> all(isfinite, lp), values(model.proposal_log_prob))
    @test all(isfinite, model.dl_fid_sq)
    @test all(>(0), model.dl_fid_sq)

    @test length(model.redshift_grid) == length(DEFAULT_Z_GRID)
    @test observation.frequencies ≈ [0.0, 20.0, 40.0]
    @test observation.in_band_mask == BitVector([false, true, true])
    @test length(observation.effective_psd) == length(observation.frequencies)
    @test observation.sgwb_scale_in_band ≈
          observation.sgwb_scale[observation.in_band_mask]
    @test observation.observation_time == 1.0
    @test year_to_second(observation.observation_time) ≈ 365.25 * 24 * 3600
    @test model.local_merger_rate == 161.0

    fs = fiducial_spectral_density(
        model, loaded.fluxes, loaded.samples, loaded.fiducials)
    @test all(isfinite, fs)
    @test length(fs) == length(observation.frequencies)
end

@testset "fiducial spectral density differs across cosmologies" begin
    loaded_w0 = _load_variant(:w0cdm)
    @test loaded_w0.cosmology_type === W0CDM
    @test loaded_w0.propagation_type === ModifiedPropagation
    P = loaded_w0.propagation_type
    fs_w0 = fiducial_spectral_density(
        loaded_w0.model, loaded_w0.fluxes, loaded_w0.samples, loaded_w0.fiducials)
    @test all(isfinite, fs_w0)

    # Build a LambdaCDM prepared model from the same raw inputs and confirm the fiducial
    # spectrum is not identical (the cosmology genuinely feeds through the caches).
    Λ = loaded_w0.fiducials
    C_lcdm = LambdaCDM
    order_lcdm = full_hyperparameters(C_lcdm, P, loaded_w0.pop)
    Λ_lcdm = canonical_hyperparameters(order_lcdm, (; (k => Λ[k] for k in order_lcdm)...))
    grid = FrequencyGrid(0.05, 80.0, 20.0, 15.0, 40.0)
    prepared_lcdm = prepare_parity_model(
        loaded_w0.pop,
        loaded_w0.samples,
        Λ_lcdm,
        C_lcdm,
        P,
        grid,
        _TEST_LOAD_DETS,
        1.0,
        161.0
    )
    model_lcdm = prepared_lcdm.model
    rate_lcdm = merger_rate(
        model_lcdm.proposal_prior,
        model_lcdm.local_merger_rate,
        model_lcdm.observation_time
    )
    @test !(fs_w0 ≈ spectral_density(loaded_w0.fluxes, rate_lcdm))
    @test !(fs_w0 ≈ fiducial_spectral_density(
        model_lcdm, loaded_w0.fluxes, loaded_w0.samples, Λ_lcdm))
end
