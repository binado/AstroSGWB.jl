using Test

@testset "load_cache" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "importance_context_julia.h5")
    problem = load_cache(fixture_path)

    @test problem.proposal.intrinsic_site_order == ["redshift"]
    @test problem.proposal.samples["redshift"] ≈ [0.1, 0.2]
    @test problem.proposal.log_prob ≈ [0.0, 0.0]
    @test problem.proposal.intrinsic_vector ≈ reshape([0.1, 0.2], :, 1)
    @test problem.proposal.cached_flux_over_dgw2 ≈ [1.0 2.0; 1.5 2.5]
    @test problem.proposal.dgw_fid_sq ≈ [4.0, 9.0]
    @test problem.observation.frequencies ≈ [1.0, 2.0]
    @test problem.observation.covariance ≈ [1.0, 1.0]
    @test problem.observation.sgwb_scale ≈ [1 / sqrt(63115200.0), 1 / sqrt(63115200.0)]
    @test problem.observation.in_band_mask == BitVector([true, true])
    @test problem.observation.fiducial_spectral_density ≈ [0.0, 0.0]
    @test problem.observation.sgwb_scale_in_band ≈ problem.observation.sgwb_scale
    @test problem.observation.fiducial_spectral_density_in_band ≈ [0.0, 0.0]
    @test problem.hyperparameters["H0"] == 67.0
    @test problem.hyperparameters["Omega_m"] == 0.315
    @test problem.hyperparameters["chi0"] == 1.0
    @test problem.hyperparameters["chin"] == 0.0
    @test problem.redshift_prior_spec.family == "madau_dickinson"
    @test problem.redshift_prior_spec.time_delay_model === nothing
    @test problem.redshift_prior_spec.num_interp == 1024
    @test problem.local_merger_rate == 161.0
    @test problem.observation.observation_time_yr == 1.0
    @test problem.observation.observation_time_sec == 365.25 * 24 * 3600
    @test problem.redshift_integral_fiducial == 1.0
    @test problem.strategy isa RedshiftOnly
    @test redshift(problem) ≈ [0.1, 0.2]
end
