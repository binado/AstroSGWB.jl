using HDF5
using Distributions: Uniform, logpdf
using Test
using ASGWB

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

function _load_variant(variant::Symbol)
    dir = parity_bundle_dir(variant)
    return load_problem(
        joinpath(dir, "bundle.h5"),
        joinpath(dir, "model.toml"),
        _TEST_LOAD_DETS;
        parity_observation_kwargs(variant)...
    )
end

@testset "save_bundle/load_bundle round-trip" begin
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
    meta = WaveformMetadata("IMRPhenomPV2", :BNS, grid, "deadbeef", "rev", "cmd")
    catalog = WaveformCatalog(samples, fluxes, meta)

    path, io = mktemp()
    close(io)
    try
        save_bundle(path, catalog)
        loaded = load_bundle(path)
        @test Set(keys(loaded.samples)) == Set(keys(catalog.samples))
        for k in keys(catalog.samples)
            @test loaded.samples[k] ≈ catalog.samples[k]
        end
        @test loaded.fluxes ≈ catalog.fluxes
        @test loaded.metadata.approximant == meta.approximant
        @test loaded.metadata.source_type == meta.source_type
        @test loaded.metadata.model_sha256 == meta.model_sha256
        @test loaded.metadata.grid.duration == grid.duration
        @test loaded.metadata.grid.sampling_frequency == grid.sampling_frequency
        @test loaded.metadata.grid.minimum_frequency == grid.minimum_frequency
        @test loaded.metadata.grid.maximum_frequency == grid.maximum_frequency
    finally
        rm(path; force = true)
    end
end

@testset "save_model_toml/load_model_toml round-trip" begin
    cases = (
        (
            ModifiedPropagation{LambdaCDM},
            BNSPopulationModel(),
            (H0 = 67.4, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5),
            "ModifiedPropagation{LambdaCDM}"
        ),
        (
            ModifiedPropagation{W0CDM},
            BNSPopulationModel(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0,
                zpeak = 2.5),
            "ModifiedPropagation{W0CDM}"
        ),
        (
            ModifiedPropagation{W0WaCDM},
            BNSPopulationModel(),
            (H0 = 67.4, Ωm = 0.315, w0 = -0.9, wa = 0.1, Ξ₀ = 1.0, Ξₙ = 0.0,
                γ = 2.7, κ = 3.0, zpeak = 2.5),
            "ModifiedPropagation{W0WaCDM}"
        )
    )

    for (C, pop, Λ_raw, tag) in cases
        order = full_hyperparameters(C, pop)
        Λ = canonical_hyperparameters(order, Λ_raw)
        path = joinpath(mktempdir(), "model.toml")
        save_model_toml(path, C, Λ)
        C2, pop2, Λ2 = load_model_toml(path)
        @testset "$tag" begin
            @test C2 === C
            @test pop2 isa BNSPopulationModel
            @test Λ2 == Λ
        end
    end
end

@testset "load_model_toml requires [parameters]" begin
    path = joinpath(mktempdir(), "model.toml")
    write(path,
        """
        [model]
        cosmology = "ModifiedPropagation{LambdaCDM}"
        """)
    @test_throws ArgumentError load_model_toml(path)
end

@testset "load_problem reconstructs derived fields" begin
    problem = _load_variant(:importance_context)
    z = problem.proposal.samples.redshift
    Λ = fiducial_hyperparameters(problem)
    C = problem.cosmology_type
    pop = problem.population

    expected_dgw_sq = reconstruct_dgw_fid_sq(z, C, Λ)
    expected_lp = reconstruct_proposal_log_prob(problem.proposal.samples, C, pop, Λ)
    expected_ri = fiducial_redshift_integral(C, pop, Λ)

    @test problem.proposal.dgw_fid_sq ≈ expected_dgw_sq
    @test problem.proposal.log_prob ≈ expected_lp
    @test fiducial_redshift_integral(problem) ≈ expected_ri rtol = 1e-6
    @test fiducial_spectral_density(problem) ≈ problem.observation.fiducial_spectral_density
end

@testset "importance_sampling_problem matches load_problem fixture" begin
    from_file = _load_variant(:importance_context)
    C = from_file.cosmology_type
    pop = from_file.population
    Λ = from_file.fiducial_hyperparameters
    samples = (
        mass = stack_source_masses([1.4, 1.4], [1.2, 1.2]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.0],
        χ₂ = [0.0, 0.0],
        Λ₁ = [100.0, 100.0],
        Λ₂ = [100.0, 100.0]
    )
    lp = reconstruct_proposal_log_prob(samples, C, pop, Λ)
    intrinsic_mat = Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                            1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    dgw_sq = reconstruct_dgw_fid_sq(samples.redshift, C, Λ)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        intrinsic_mat,
        from_file.proposal.cached_flux_over_dgw2,
        dgw_sq
    )
    from_memory = importance_sampling_problem(
        proposal,
        from_file.observation,
        C,
        pop,
        Λ,
        from_file.local_merger_rate
    )

    @test from_memory.proposal.intrinsic_site_order ==
          from_file.proposal.intrinsic_site_order
    @test from_memory.proposal.samples.redshift == from_file.proposal.samples.redshift
    @test from_memory.proposal.log_prob ≈ from_file.proposal.log_prob
    @test from_memory.proposal.intrinsic_vector ≈ from_file.proposal.intrinsic_vector
    @test from_memory.proposal.cached_flux_over_dgw2 ≈
          from_file.proposal.cached_flux_over_dgw2
    @test from_memory.proposal.dgw_fid_sq ≈ from_file.proposal.dgw_fid_sq
    @test from_memory.observation.frequencies ≈ from_file.observation.frequencies
    @test from_memory.observation.effective_psd ≈ from_file.observation.effective_psd
    @test from_memory.observation.sgwb_scale ≈ from_file.observation.sgwb_scale
    @test from_memory.observation.in_band_mask == from_file.observation.in_band_mask
    @test from_memory.observation.fiducial_spectral_density ≈
          from_file.observation.fiducial_spectral_density
    @test length(from_memory.redshift_grid) == length(from_file.redshift_grid)
    @test from_memory.fiducial_hyperparameters == from_file.fiducial_hyperparameters
    @test from_memory.local_merger_rate == from_file.local_merger_rate
    @test typeof(from_memory.strategy) == typeof(from_file.strategy)
end

@testset "load_problem" begin
    problem = _load_variant(:importance_context)

    @test problem.proposal.intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
    s = problem.proposal.samples
    @test s.redshift ≈ [0.1, 0.2]
    @test Vector(s.mass[1, :]) ≈ [1.4, 1.4]
    @test Vector(s.mass[2, :]) ≈ [1.2, 1.2]
    @test s.χ₁ ≈ [0.0, 0.0]
    @test s.χ₂ ≈ [0.0, 0.0]
    @test s.Λ₁ ≈ [100.0, 100.0]
    @test s.Λ₂ ≈ [100.0, 100.0]

    C = problem.cosmology_type
    pop = problem.population
    Λ = fiducial_hyperparameters(problem)

    expected_lp = reconstruct_proposal_log_prob(problem.proposal.samples, C, pop, Λ)
    @test problem.proposal.log_prob ≈ expected_lp rtol = 1e-6
    @test problem.proposal.intrinsic_vector ≈ Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                  1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    @test problem.proposal.cached_flux_over_dgw2 ≈ Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    @test problem.proposal.dgw_fid_sq ≈ reconstruct_dgw_fid_sq(
        problem.proposal.samples.redshift, C, Λ
    )

    @test problem.observation.frequencies ≈ [0.0, 20.0, 40.0]
    @test length(problem.observation.effective_psd) ==
          length(problem.observation.frequencies)
    @test length(problem.observation.sgwb_scale) == length(problem.observation.frequencies)
    @test problem.observation.in_band_mask == BitVector([false, true, true])

    ev = evaluate_model_terms(fiducial_hyperparameters(problem), problem)
    @test problem.observation.fiducial_spectral_density ≈ ev.spectral_density
    @test problem.observation.sgwb_scale_in_band ≈
          problem.observation.sgwb_scale[problem.observation.in_band_mask]
    @test problem.observation.fiducial_spectral_density_in_band ≈
          ev.spectral_density_in_band

    @test C === ModifiedPropagation{LambdaCDM}
    @test Λ.H0 == 67.0
    @test Λ.Ωm == 0.315
    @test Λ.Ξ₀ == 1.0
    @test Λ.Ξₙ == 0.0
    @test Λ.γ == 2.7
    @test Λ.κ == 3.0
    @test Λ.zpeak == 2.5
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec ≈ 365.25 * 24 * 3600
    @test fiducial_redshift_integral(problem) ≈
          fiducial_redshift_integral(C, pop, Λ) rtol = 1e-6
    @test problem.strategy isa FullBNS
    @test redshift(problem) ≈ [0.1, 0.2]
    @test length(problem.redshift_grid) == length(DEFAULT_Z_GRID)
end

@testset "fiducial_spectral_density uses W0CDM bundle" begin
    p = _load_variant(:w0cdm)
    @test p.cosmology_type === ModifiedPropagation{W0CDM}
    Λ = fiducial_hyperparameters(p)
    ev = evaluate_model_terms(Λ, p)
    @test fiducial_spectral_density(p) ≈ ev.spectral_density

    C_lcdm = ModifiedPropagation{LambdaCDM}
    pop = BNSPopulationModel()
    order_lcdm = full_hyperparameters(C_lcdm, pop)
    h_lcdm = canonical_hyperparameters(
        order_lcdm,
        (; (k => Λ[k] for k in order_lcdm)...)
    )
    p_lcdm = ASGWB.ImportanceSamplingProblem(
        p.proposal,
        p.observation,
        C_lcdm,
        pop,
        h_lcdm,
        p.redshift_grid,
        p.redshift_cache,
        p.local_merger_rate,
        p.strategy
    )
    ev_lcdm = evaluate_model_terms(h_lcdm, p_lcdm)
    @test !(fiducial_spectral_density(p) ≈ ev_lcdm.spectral_density)
end

@testset "load_problem throws on model fingerprint mismatch" begin
    dir = parity_bundle_dir(:importance_context)
    bundle_path = joinpath(dir, "bundle.h5")
    tmp_dir = mktempdir()
    bad_toml = joinpath(tmp_dir, "model.toml")
    write(bad_toml,
        """
        [model]
        cosmology = "ModifiedPropagation{LambdaCDM}"

        [parameters]
        H0 = 99.0
        "Ωm" = 0.5
        "Ξ₀" = 1.0
        "Ξₙ" = 0.0
        "γ" = 2.7
        "κ" = 3.0
        zpeak = 2.5
        """)
    @test_throws ArgumentError load_problem(
        bundle_path,
        bad_toml,
        _TEST_LOAD_DETS;
        parity_observation_kwargs(:importance_context)...
    )
end

@testset "importance_sampling_problem builds redshift cache" begin
    C = ModifiedPropagation{LambdaCDM}
    pop = BNSPopulationModel()
    order = full_hyperparameters(C, pop)
    Λ = canonical_hyperparameters(
        order,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    samples = (
        mass = stack_source_masses([1.4], [1.2]),
        redshift = [0.1],
        χ₁ = [0.0],
        χ₂ = [0.0],
        Λ₁ = [100.0],
        Λ₂ = [100.0]
    )
    lp = reconstruct_proposal_log_prob(samples, C, pop, Λ)
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        reshape([1.4, 1.2, 0.1, 0.0, 0.0, 100.0, 100.0], 1, :),
        fill(1.0, 1, 2),
        [1.0]
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
    p = importance_sampling_problem(proposal, observation, C, pop, Λ, 1.0)
    @test fiducial_redshift_integral(p) ≈ fiducial_redshift_integral(C, pop, Λ) rtol = 1e-10
end

@testset "importance_sampling_problem realizes population prior per call" begin
    C = ModifiedPropagation{LambdaCDM}
    pop = BNSPopulationModel()
    order = full_hyperparameters(C, pop)
    Λ = canonical_hyperparameters(
        order,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        zeros(2),
        Float64[1.4 1.2 0.1 0.0 0.0 100.0 150.0
                1.5 1.3 0.2 0.1 -0.2 200.0 250.0],
        ones(2, 2),
        ones(2)
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
    p = importance_sampling_problem(proposal, observation, C, pop, Λ, 1.0)
    c = cosmology(C, Λ)
    prior = single_event_prior(pop, c, Λ)
    @test compute_importance_weights(p, Λ).target_log_prob ≈ batched_logpdf(prior, samples)
end
