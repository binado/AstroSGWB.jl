using HDF5
using Distributions: Uniform, logpdf
using Test
using ASGWB

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end

const _TEST_LOAD_DETS = [Detector("H1"), Detector("L1")]

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
        @test loaded.metadata.cosmology_sha256 == meta.cosmology_sha256
        @test loaded.metadata.grid.duration == grid.duration
        @test loaded.metadata.grid.sampling_frequency == grid.sampling_frequency
        @test loaded.metadata.grid.minimum_frequency == grid.minimum_frequency
        @test loaded.metadata.grid.maximum_frequency == grid.maximum_frequency
    finally
        rm(path; force = true)
    end
end

@testset "save_cosmology_toml/load_cosmology_toml round-trip" begin
    for (cosmo, tag) in [
        (LambdaCDM(67.4, 0.315), "LambdaCDM"),
        (W0CDM(67.4, 0.315, -0.9), "W0CDM"),
        (W0WaCDM(67.4, 0.315, -0.9, 0.1), "W0WaCDM"),
    ]
        fid = FiducialParameters(
            cosmo,
            ModifiedGravity(1.0, 0.0),
            PopulationParams(MadauDickinson, 2.7, 3.0, 2.5, nothing, 0.001, 20.0, 256, nothing),
            ObservationParams(161.0, 1.0)
        )
        path = joinpath(mktempdir(), "cosmology.toml")
        save_cosmology_toml(path, fid)
        loaded = load_cosmology_toml(path)
        @testset "$tag" begin
            @test cosmology_type(loaded) == cosmology_type(fid)
            @test loaded.cosmology.H0 == fid.cosmology.H0
            @test loaded.cosmology.Ωm == fid.cosmology.Ωm
            @test loaded.modified_gravity.Ξ₀ == fid.modified_gravity.Ξ₀
            @test loaded.modified_gravity.Ξₙ == fid.modified_gravity.Ξₙ
            @test loaded.population.family == fid.population.family
            @test loaded.population.γ == fid.population.γ
            @test loaded.population.κ == fid.population.κ
            @test loaded.population.zpeak == fid.population.zpeak
            @test loaded.population.z_min == fid.population.z_min
            @test loaded.population.z_max == fid.population.z_max
            @test loaded.population.num_interp == fid.population.num_interp
            @test loaded.observation.local_merger_rate == fid.observation.local_merger_rate
            @test loaded.observation.observation_time_yr == fid.observation.observation_time_yr
        end
    end
end

@testset "load_problem reconstructs derived fields" begin
    dir = parity_bundle_dir(:importance_context)
    problem = load_problem(
        joinpath(dir, "bundle.h5"), joinpath(dir, "cosmology.toml"), _TEST_LOAD_DETS)

    z = problem.proposal.samples.redshift
    fid = problem.fiducial_parameters
    spec = problem.redshift_prior_spec

    expected_dgw_sq = reconstruct_dgw_fid_sq(z, fid)
    expected_lp = reconstruct_proposal_log_prob(problem.proposal.samples, spec, fid)
    expected_ri = fiducial_redshift_integral(fid, spec)

    @test problem.proposal.dgw_fid_sq ≈ expected_dgw_sq
    @test problem.proposal.log_prob ≈ expected_lp
    @test problem.redshift_integral_fiducial ≈ expected_ri rtol = 1e-6
    @test fiducial_spectral_density(problem) ≈ problem.observation.fiducial_spectral_density
end

@testset "importance_sampling_problem matches load_problem fixture" begin
    dir = parity_bundle_dir(:importance_context)
    from_file = load_problem(
        joinpath(dir, "bundle.h5"), joinpath(dir, "cosmology.toml"), _TEST_LOAD_DETS)

    samples = (
        mass = stack_source_masses([1.4, 1.4], [1.2, 1.2]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.0],
        χ₂ = [0.0, 0.0],
        Λ₁ = [100.0, 100.0],
        Λ₂ = [100.0, 100.0]
    )
    lp = reconstruct_proposal_log_prob(
        samples,
        from_file.redshift_prior_spec,
        from_file.fiducial_parameters
    )
    intrinsic_mat = Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                            1.4 1.2 0.2 0.0 0.0 100.0 100.0]
    dgw_sq = reconstruct_dgw_fid_sq(samples.redshift, from_file.fiducial_parameters)
    cached_flux_over_dgw2 = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        lp,
        intrinsic_mat,
        cached_flux_over_dgw2,
        dgw_sq
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 1024, nothing)
    from_memory = importance_sampling_problem(
        proposal,
        from_file.observation,
        spec,
        161.0,
        from_file.redshift_integral_fiducial,
        from_file.fiducial_parameters
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
    @test from_memory.observation.observation_time_sec ==
          from_file.observation.observation_time_sec
    @test from_memory.observation.observation_time_yr ==
          from_file.observation.observation_time_yr
    @test from_memory.redshift_prior_spec.family == from_file.redshift_prior_spec.family
    @test from_memory.redshift_prior_spec.z_min == from_file.redshift_prior_spec.z_min
    @test from_memory.redshift_prior_spec.z_max == from_file.redshift_prior_spec.z_max
    @test from_memory.redshift_prior_spec.num_interp ==
          from_file.redshift_prior_spec.num_interp
    @test from_memory.redshift_prior_spec.time_delay_model ===
          from_file.redshift_prior_spec.time_delay_model
    @test from_memory.fiducial_parameters.cosmology.H0 ==
          from_file.fiducial_parameters.cosmology.H0
    @test from_memory.fiducial_parameters.cosmology.Ωm ==
          from_file.fiducial_parameters.cosmology.Ωm
    @test from_memory.fiducial_parameters.modified_gravity.Ξ₀ ==
          from_file.fiducial_parameters.modified_gravity.Ξ₀
    @test from_memory.fiducial_parameters.modified_gravity.Ξₙ ==
          from_file.fiducial_parameters.modified_gravity.Ξₙ
    @test from_memory.local_merger_rate == from_file.local_merger_rate
    @test from_memory.redshift_integral_fiducial == from_file.redshift_integral_fiducial
    @test typeof(from_memory.strategy) == typeof(from_file.strategy)
end

@testset "load_problem" begin
    dir = parity_bundle_dir(:importance_context)
    problem = load_problem(
        joinpath(dir, "bundle.h5"), joinpath(dir, "cosmology.toml"), _TEST_LOAD_DETS)

    @test problem.proposal.intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
    s = problem.proposal.samples
    @test s.redshift ≈ [0.1, 0.2]
    @test Vector(s.mass[1, :]) ≈ [1.4, 1.4]
    @test Vector(s.mass[2, :]) ≈ [1.2, 1.2]
    @test s.χ₁ ≈ [0.0, 0.0]
    @test s.χ₂ ≈ [0.0, 0.0]
    @test s.Λ₁ ≈ [100.0, 100.0]
    @test s.Λ₂ ≈ [100.0, 100.0]

    expected_lp = reconstruct_proposal_log_prob(
        problem.proposal.samples,
        problem.redshift_prior_spec,
        problem.fiducial_parameters
    )
    @test problem.proposal.log_prob ≈ expected_lp rtol = 1e-6

    @test problem.proposal.intrinsic_vector ≈ Float64[1.4 1.2 0.1 0.0 0.0 100.0 100.0
                  1.4 1.2 0.2 0.0 0.0 100.0 100.0]

    @test problem.proposal.cached_flux_over_dgw2 ≈ Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    @test problem.proposal.dgw_fid_sq ≈ reconstruct_dgw_fid_sq(
        problem.proposal.samples.redshift,
        problem.fiducial_parameters
    )

    @test problem.observation.frequencies ≈ [0.0, 20.0, 40.0]
    @test length(problem.observation.effective_psd) == length(problem.observation.frequencies)
    @test length(problem.observation.sgwb_scale) == length(problem.observation.frequencies)
    @test problem.observation.in_band_mask == BitVector([false, true, true])

    model = propagation_model(problem.fiducial_parameters)
    ev = evaluate_model_terms(model, fiducial_hyperparameters(problem), problem)
    @test problem.observation.fiducial_spectral_density ≈ ev.spectral_density
    @test problem.observation.sgwb_scale_in_band ≈
          problem.observation.sgwb_scale[problem.observation.in_band_mask]
    @test problem.observation.fiducial_spectral_density_in_band ≈ ev.spectral_density_in_band

    @test problem.fiducial_parameters.cosmology.H0 == 67.0
    @test problem.fiducial_parameters.cosmology.Ωm == 0.315
    @test problem.fiducial_parameters.modified_gravity.Ξ₀ == 1.0
    @test problem.fiducial_parameters.modified_gravity.Ξₙ == 0.0
    @test problem.redshift_prior_spec.family == MadauDickinson
    @test problem.redshift_prior_spec.z_min == 0.001
    @test problem.redshift_prior_spec.z_max == 20.0
    @test problem.redshift_prior_spec.time_delay_model === nothing
    @test problem.redshift_prior_spec.num_interp == 1024
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec ≈ 365.25 * 24 * 3600
    spec = problem.redshift_prior_spec
    @test problem.redshift_integral_fiducial ≈ fiducial_redshift_integral(
        problem.fiducial_parameters, spec) rtol = 1e-6
    @test problem.strategy isa FullBNS
    @test redshift(problem) ≈ [0.1, 0.2]
end

@testset "fiducial_spectral_density uses W0CDM bundle" begin
    dir = parity_bundle_dir(:w0cdm)
    p = load_problem(
        joinpath(dir, "bundle.h5"), joinpath(dir, "cosmology.toml"), _TEST_LOAD_DETS)

    @test propagation_model(p.fiducial_parameters) isa
          MadauDickinsonModifiedPropagation{W0CDM}
    Λ = fiducial_hyperparameters(p)
    ev = evaluate_model_terms(propagation_model(p.fiducial_parameters), Λ, p)
    @test fiducial_spectral_density(p) ≈ ev.spectral_density

    h_lcdm = canonical_hyperparameters(
        MadauDickinsonModifiedPropagation(),
        (; (k => Λ[k] for k in hyperparameters(MadauDickinsonModifiedPropagation()))...)
    )
    ev_lcdm = evaluate_model_terms(MadauDickinsonModifiedPropagation(), h_lcdm, p)
    @test !(fiducial_spectral_density(p) ≈ ev_lcdm.spectral_density)
end

@testset "load_problem throws on cosmology fingerprint mismatch" begin
    dir = parity_bundle_dir(:importance_context)
    bundle_path = joinpath(dir, "bundle.h5")
    tmp_dir = mktempdir()
    bad_toml = joinpath(tmp_dir, "cosmology.toml")
    write(bad_toml,
        """
        [cosmology]
        type = "LambdaCDM"
        H0 = 99.0
        Omega_m = 0.5

        [modified_gravity]
        Xi_0 = 1.0
        Xi_n = 0.0

        [population]
        family = "madau_dickinson"
        gamma = 2.7
        kappa = 3.0
        z_peak = 2.5
        z_min = 0.001
        z_max = 20.0
        num_interp = 64
        time_delay_model = "none"

        [observation]
        local_merger_rate = 161.0
        observation_time_yr = 1.0
        """)
    @test_throws ArgumentError load_problem(bundle_path, bad_toml, _TEST_LOAD_DETS)
end

@testset "importance_sampling_problem 5-arg infers redshift integral" begin
    fid = FiducialParameters(
        LambdaCDM(67.0, 0.315),
        ModifiedGravity(1.0, 0.0),
        PopulationParams(MadauDickinson, 2.7, 3.0, 2.5, nothing, 0.001, 20.0, 256, nothing),
        ObservationParams(1.0, 1.0)
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    ri = fiducial_redshift_integral(fid, spec)
    samples = (
        mass = stack_source_masses([1.4], [1.2]),
        redshift = [0.1],
        χ₁ = [0.0],
        χ₂ = [0.0],
        Λ₁ = [100.0],
        Λ₂ = [100.0]
    )
    lp = reconstruct_proposal_log_prob(samples, spec, fid)
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
    p = importance_sampling_problem(proposal, observation, spec, 1.0, fid)
    @test p.redshift_integral_fiducial ≈ ri rtol = 1e-10
end

@testset "importance_sampling_problem accepts custom intrinsic prior factory" begin
    fid = FiducialParameters(
        LambdaCDM(67.0, 0.315),
        ModifiedGravity(1.0, 0.0),
        PopulationParams(MadauDickinson, 2.7, 3.0, 2.5, nothing, 0.001, 20.0, 64, nothing),
        ObservationParams(1.0, 1.0)
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 64, nothing)
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
    factory = strategy -> begin
        @test strategy isa FullBNS
        IntrinsicPrior((χ₁ = Uniform(-1.0, 1.0), Λ₁ = Uniform(0.0, 500.0)))
    end

    p = importance_sampling_problem(
        proposal,
        observation,
        spec,
        1.0,
        1.0,
        fid;
        intrinsic_prior_factory = factory
    )
    expected = [logpdf(Uniform(-1.0, 1.0), samples.χ₁[i]) +
                logpdf(Uniform(0.0, 500.0), samples.Λ₁[i])
                for i in eachindex(samples.χ₁)]
    @test p.redshift_cache.cached_intrinsic_log_prob ≈ expected
end
