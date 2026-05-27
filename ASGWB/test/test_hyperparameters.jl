using Test
using ASGWB
using Distributions: product_distribution, Normal, ProductNamedTupleDistribution

@testset "Hyperparameter Validation" begin
    model = MadauDickinsonModifiedPropagation()
    expected_order = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    # 1. ProductNamedTupleDistribution setup
    dists = (; (k => Normal(0.0, 1.0) for k in expected_order)...)
    prior = product_distribution(dists)

    @testset "validate_subset" begin
        # Valid subsets (tuple)
        @test validate_subset((:H0, :Ωm), model) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), prior) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), expected_order) === (:H0, :Ωm)

        # Empty subset is allowed in validate_subset
        @test validate_subset((), model) === ()
        @test validate_subset((), prior) === ()
        @test validate_subset((), expected_order) === ()

        # Valid subsets (NamedTuple)
        nt = (H0 = 70.0, Ωm = 0.3)
        @test validate_subset(nt, model) === nt
        @test validate_subset(nt, prior) === nt
        @test validate_subset(nt, expected_order) === nt

        # Unknown symbols throw ArgumentError
        @test_throws ArgumentError validate_subset((:invalid_key,), model)
        @test_throws ArgumentError validate_subset((:invalid_key,), prior)
        @test_throws ArgumentError validate_subset((:invalid_key,), expected_order)
        @test_throws ArgumentError validate_subset((invalid_key = 1.0,), model)

        # Repeating/Duplicate symbols throw ArgumentError
        @test_throws ArgumentError validate_subset((:H0, :H0), model)
        @test_throws ArgumentError validate_subset((:H0, :H0), prior)
        @test_throws ArgumentError validate_subset((:H0, :H0), expected_order)
    end

    @testset "validate_hyperparameters" begin
        # Exact match
        ok_nt = (; (k => 1.0 for k in expected_order)...)
        @test validate_hyperparameters(model, ok_nt) === nothing

        # Missing keys
        missing_nt = (H0 = 70.0, Ωm = 0.3)
        @test_throws ArgumentError validate_hyperparameters(model, missing_nt)

        # Extra keys
        extra_nt = (; (k => 1.0 for k in expected_order)..., extra_key = 1.0)
        @test_throws ArgumentError validate_hyperparameters(model, extra_nt)
    end

    @testset "hyperparameters W0CDM / W0WaCDM" begin
        @test hyperparameters(MadauDickinsonModifiedPropagation{W0CDM}()) ==
              (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test hyperparameters(MadauDickinsonModifiedPropagation{W0WaCDM}()) ==
              (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    end
end

@testset "propagation_model(FiducialParameters)" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 5.7, zpeak = 2.0)
    pop = PopulationParams(MadauDickinson, base.γ, base.κ, base.zpeak, nothing, 0.001, 20.0, 64, nothing)
    mg = ModifiedGravity(base.Ξ₀, base.Ξₙ)
    obs = ObservationParams(1.0, 1.0)

    lcdm_fid = FiducialParameters(LambdaCDM(base.H0, base.Ωm), mg, pop, obs)
    model_lcdm = propagation_model(lcdm_fid)
    Λ_lcdm = (; H0 = base.H0, Ωm = base.Ωm, Ξ₀ = base.Ξ₀, Ξₙ = base.Ξₙ,
                γ = base.γ, κ = base.κ, zpeak = base.zpeak)
    @test cosmology(model_lcdm, Λ_lcdm) isa LambdaCDM

    w0_fid = FiducialParameters(W0CDM(base.H0, base.Ωm, -0.9), mg, pop, obs)
    model_w0 = propagation_model(w0_fid)
    Λ_w0 = (; Λ_lcdm..., w0 = -0.9)
    @test cosmology(model_w0, Λ_w0) isa W0CDM

    cpl_fid = FiducialParameters(W0WaCDM(base.H0, base.Ωm, -0.9, 0.2), mg, pop, obs)
    model_cpl = propagation_model(cpl_fid)
    Λ_cpl = (; Λ_lcdm..., w0 = -0.9, wa = 0.2)
    @test cosmology(model_cpl, Λ_cpl) isa W0WaCDM

    @test fiducial_cosmology(lcdm_fid) isa LambdaCDM
    @test fiducial_cosmology(w0_fid) isa W0CDM
    @test fiducial_cosmology(cpl_fid) isa W0WaCDM

    @test ASGWB._cosmology_nt(LambdaCDM(base.H0, base.Ωm)) == (H0 = base.H0, Ωm = base.Ωm)
    @test ASGWB._cosmology_nt(W0CDM(base.H0, base.Ωm, -0.9)) ==
          (H0 = base.H0, Ωm = base.Ωm, w0 = -0.9)
    @test ASGWB._cosmology_nt(W0WaCDM(base.H0, base.Ωm, -0.9, 0.2)) ==
          (H0 = base.H0, Ωm = base.Ωm, w0 = -0.9, wa = 0.2)
end
