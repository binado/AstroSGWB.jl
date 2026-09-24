using Test
using ForwardDiff
using BackgroundCosmology: LambdaCDM, W0CDM, W0WaCDM, luminosity_distance
using GWBackgroundImportanceModels: GR, ModifiedPropagation, propagation,
                                    gw_em_distance_ratio, log_gw_em_distance_ratio,
                                    apply_gw_distance_correction,
                                    apply_gw_distance_correction!

@testset "gw_em_distance_ratio" begin
    zs = (0.0, 0.3, 1.0, 2.5)
    c = LambdaCDM(67.0, 0.315)

    # Ξ₀ = 1, Ξₙ = 0 is the identity, so a fake distance vector is unchanged.
    @test gw_em_distance_ratio.([0.1, 0.2], 1.0, 0.0) .* [10.0, 20.0] ≈ [10.0, 20.0]

    # GR propagation recovers Ξ ≡ 1, so D_gw = D_L for any background cosmology.
    for cosmo in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        for z in zs
            @test gw_em_distance_ratio(z, GR()) ≈ 1.0
            @test log_gw_em_distance_ratio(z, GR()) ≈ 0.0
            @test gw_em_distance_ratio(z, GR()) * luminosity_distance(z, cosmo) ≈
                  luminosity_distance(z, cosmo)
        end
    end

    # ModifiedPropagation applies Ξ(z) = Ξ₀ + (1 - Ξ₀)/(1 + z)^Ξₙ, independent of cosmology.
    Ξ₀, Ξₙ = 1.2, 2.0
    p_mod = ModifiedPropagation(Ξ₀, Ξₙ)
    for z in zs
        Ξ = Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ
        @test gw_em_distance_ratio(z, p_mod) ≈ Ξ
        @test gw_em_distance_ratio(z, Ξ₀, Ξₙ) ≈ Ξ
        @test log_gw_em_distance_ratio(z, p_mod) ≈ log(Ξ)
        @test log_gw_em_distance_ratio(z, Ξ₀, Ξₙ) ≈ log(gw_em_distance_ratio(z, Ξ₀, Ξₙ))
        # GW luminosity distance is exactly Ξ(z) · D_L.
        @test gw_em_distance_ratio(z, p_mod) * luminosity_distance(z, c) ≈
              Ξ * luminosity_distance(z, c)
    end

    # Ξ₀ = 1 is the identity for any Ξₙ.
    for Ξₙ in (0.0, 0.7, 2.0)
        for z in zs
            @test gw_em_distance_ratio(z, 1.0, Ξₙ) ≈ 1.0
            @test log_gw_em_distance_ratio(z, ModifiedPropagation(1.0, Ξₙ)) ≈ 0.0
        end
    end
end

@testset "apply_gw_distance_correction" begin
    z = [0.1, 0.5, 2.0]
    polarization_power = Float64[1.0 2.0 3.0
                                 4.0 5.0 6.0]
    p_mod = ModifiedPropagation(1.4, 0.7)

    # GR is a true no-op and the bang form hands back the same object.
    gr_in = copy(polarization_power)
    @test apply_gw_distance_correction!(gr_in, z, GR()) === gr_in
    @test gr_in == polarization_power

    # ModifiedPropagation divides column j by Ξ(z[j])².
    expected = reduce(hcat,
        [polarization_power[:, j] ./ gw_em_distance_ratio(z[j], p_mod)^2
         for j in eachindex(z)])
    @test apply_gw_distance_correction(polarization_power, z, p_mod) ≈ expected
    # Ξ₀ = 1 makes ModifiedPropagation exactly the identity too.
    @test apply_gw_distance_correction(polarization_power, z, ModifiedPropagation(1.0, 0.7)) ==
          polarization_power

    # Out-of-place never aliases or mutates its input; the bang form mutates in place.
    untouched = copy(polarization_power)
    out = apply_gw_distance_correction(polarization_power, z, p_mod)
    @test out !== polarization_power
    @test polarization_power == untouched
    bang_target = copy(polarization_power)
    @test apply_gw_distance_correction!(bang_target, z, p_mod) === bang_target
    @test bang_target ≈ expected

    # Not idempotent: a second application squares the factor. Documented, and the
    # reason notebook call sites use the out-of-place form.
    @test apply_gw_distance_correction!(copy(expected), z, p_mod) ≈
          reduce(hcat,
        [polarization_power[:, j] ./ gw_em_distance_ratio(z[j], p_mod)^4
         for j in eachindex(z)])

    # A redshift vector that does not match the polarization-power columns (e.g. a subsetted sample
    # set) is caught rather than silently correcting only a prefix — including under GR.
    @test_throws DimensionMismatch apply_gw_distance_correction!(
        copy(polarization_power), z[1:2],
        p_mod)
    @test_throws DimensionMismatch apply_gw_distance_correction!(
        copy(polarization_power), z[1:2],
        GR())
    @test_throws DimensionMismatch apply_gw_distance_correction(
        polarization_power, [z;
                             3.0], p_mod)
end

@testset "propagation axis" begin
    @test propagation(GR, (;)) === GR()
    h_mod = (Ξ₀ = 1.2, Ξₙ = 2.0)
    p_mod = propagation(ModifiedPropagation, h_mod)
    @test p_mod isa ModifiedPropagation
    @test p_mod.Ξ₀ == 1.2 && p_mod.Ξₙ == 2.0
    # Extra (cosmology) keys are ignored when building propagation.
    @test propagation(ModifiedPropagation, (; h_mod..., H0 = 67.0, Ωm = 0.315)) == p_mod
end

@testset "ModifiedPropagation promotes mixed eltypes" begin
    # `ModifiedPropagation{T}` shares one type parameter between both fields, so without
    # the promoting outer constructor these are `MethodError`s.
    @test ModifiedPropagation(1.4, 1) === ModifiedPropagation(1.4, 1.0)
    @test propagation(ModifiedPropagation, (Ξ₀ = 1.4, Ξₙ = 1)) ===
          ModifiedPropagation(1.4, 1.0)

    # The shape a partially-sampled run produces: one slot `Dual`, the other `Float64`.
    # `Ξ(z) = Ξ₀ + (1 - Ξ₀)(1 + z)^(-Ξₙ)`, so ∂Ξ/∂Ξ₀ = 1 - (1 + z)^(-Ξₙ).
    z, Ξₙ = 0.7, 1.9
    dΞ = ForwardDiff.derivative(
        Ξ₀ -> gw_em_distance_ratio(z, propagation(ModifiedPropagation, (; Ξ₀, Ξₙ))), 1.4)
    @test dΞ ≈ 1 - (1 + z)^(-Ξₙ)

    # And the mirrored case: `Ξₙ` free, `Ξ₀` fixed.
    dΞₙ = ForwardDiff.derivative(
        Ξₙ -> gw_em_distance_ratio(z, propagation(ModifiedPropagation, (Ξ₀ = 1.4, Ξₙ))),
        1.9)
    @test dΞₙ ≈ -(1 - 1.4) * log(1 + z) * (1 + z)^(-1.9)

    dlogΞ = ForwardDiff.derivative(
        Ξ₀ -> log_gw_em_distance_ratio(z, propagation(ModifiedPropagation, (; Ξ₀, Ξₙ))),
        1.4)
    @test dlogΞ ≈ (1 - (1 + z)^(-Ξₙ)) / gw_em_distance_ratio(z, 1.4, Ξₙ)
end
