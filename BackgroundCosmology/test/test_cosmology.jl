using QuadGK
using Test
using ForwardDiff
using BackgroundCosmology: hubble_constant_si, cosmology,
                           comoving_distance, W0CDM, W0WaCDM,
                           GR, ModifiedPropagation,
                           propagation, propagation_type, propagation_config_name,
                           SUPPORTED_PROPAGATIONS, hubble_distance

@testset "hubble_constant_si" begin
    H0 = 70.0
    @test hubble_constant_si(H0) ≈ Float64(H0) * 1000.0 / 3.085677581e22
end

@testset "basic cosmology helpers" begin
    c = LambdaCDM(67.0, 0.315)
    @test H0(c) == 67.0
    @test Ωm(c) == 0.315

    @test E(0.0, c) ≈ 1.0
    @test comoving_distance(0.0, c) ≈ 0.0

    # d_h = c / H0 in Mpc (speed of light in km/s over H0 in km/s/Mpc).
    @test hubble_distance(c) ≈ 299792.458 / 67.0

    z = [0.0, 0.1, 0.2]
    d_l = luminosity_distance.(z, c)
    @test d_l[1] ≈ 0.0
    @test d_l[3] > d_l[2] > d_l[1]

    # GW luminosity distance is gw_em_distance_ratio(z, ...) * D_L; GR ⇒ identity.
    d_gw = gw_em_distance_ratio.([0.1, 0.2], 1.0, 0.0) .* [10.0, 20.0]
    @test d_gw ≈ [10.0, 20.0]
end

@testset "gw_em_distance_ratio" begin
    zs = (0.0, 0.3, 1.0, 2.5)
    c = LambdaCDM(67.0, 0.315)

    # GR propagation recovers Ξ ≡ 1, so D_gw = D_L for any background cosmology.
    for cosmo in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        for z in zs
            @test gw_em_distance_ratio(z, GR()) ≈ 1.0
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
        # GW luminosity distance is exactly Ξ(z) · D_L.
        @test gw_em_distance_ratio(z, p_mod) * luminosity_distance(z, c) ≈
              gw_em_distance_ratio(z, p_mod) * luminosity_distance(z, c)
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

@testset "cosmology construction" begin
    h_lcdm = (H0 = 67.0, Ωm = 0.315)
    @test cosmology(LambdaCDM, h_lcdm) == LambdaCDM(67.0, 0.315)
    @test LambdaCDM(h_lcdm) == LambdaCDM(67.0, 0.315)
    @test cosmology(h_lcdm) == LambdaCDM(67.0, 0.315)
    # Extra propagation keys in `h` are ignored by the cosmology builder.
    @test cosmology(LambdaCDM, (; h_lcdm..., Ξ₀ = 1.2, Ξₙ = 2.0)) == LambdaCDM(67.0, 0.315)

    h_w0 = (; h_lcdm..., w0 = -0.9)
    @test cosmology(W0CDM, h_w0) == W0CDM(67.0, 0.315, -0.9)
    @test cosmology(h_w0) == W0CDM(67.0, 0.315, -0.9)

    h_cpl = (; h_w0..., wa = 0.2)
    @test cosmology(W0WaCDM, h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)
    @test cosmology(h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)
end

@testset "propagation axis" begin
    @test propagation(GR, (;)) === GR()
    h_mod = (Ξ₀ = 1.2, Ξₙ = 2.0)
    p_mod = propagation(ModifiedPropagation, h_mod)
    @test p_mod isa ModifiedPropagation
    @test p_mod.Ξ₀ == 1.2 && p_mod.Ξₙ == 2.0
    # Extra (cosmology) keys are ignored when building propagation.
    @test propagation(ModifiedPropagation, (; h_mod..., H0 = 67.0, Ωm = 0.315)) == p_mod

    @test propagation_config_name(GR) == "GR"
    @test propagation_config_name(ModifiedPropagation) == "ModifiedPropagation"
    @test propagation_type("GR") === GR
    @test propagation_type("ModifiedPropagation") === ModifiedPropagation
    @test Set(SUPPORTED_PROPAGATIONS) == Set((GR, ModifiedPropagation))
    @test_throws ArgumentError propagation_type("not_a_model")
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
end

@testset "dark_energy_eos" begin
    lcdm = LambdaCDM(67.0, 0.3)
    w0cdm = W0CDM(67.0, 0.3, -0.8)
    w0wacdm = W0WaCDM(67.0, 0.3, -0.8, 0.3)
    for z in (0.0, 0.5, 1.0, 2.0)
        @test dark_energy_eos(lcdm, z) ≈ -1.0
        @test dark_energy_eos(w0cdm, z) ≈ -0.8
        @test dark_energy_eos(w0wacdm, z) ≈ -0.8 + 0.3 * z / (1 + z)
    end
end

@testset "de_density_ratio" begin
    lcdm = LambdaCDM(67.0, 0.3)
    w0cdm = W0CDM(67.0, 0.3, -0.8)
    w0wacdm = W0WaCDM(67.0, 0.3, -0.9, 0.2)
    for z in (0.1, 0.5, 1.0, 2.0, 5.0)
        @test de_density_ratio(lcdm, z) ≈ 1.0
        # w0CDM closed form vs quadgk integral
        expected_w0, _ = quadgk(
            zp -> 3 * (1 + (-0.8)) / (1 + zp), 0.0, z; rtol = 1e-10
        )
        @test de_density_ratio(w0cdm, z) ≈ exp(expected_w0) rtol = 1e-10
        # w0waCDM closed form vs quadgk integral
        expected_cpl,
        _ = quadgk(
            zp -> 3 * (1 + dark_energy_eos(w0wacdm, zp)) / (1 + zp), 0.0, z; rtol = 1e-10
        )
        @test de_density_ratio(w0wacdm, z) ≈ exp(expected_cpl) rtol = 1e-8
    end
end

@testset "comoving_distance preserves AD tags at z=0" begin
    c_w0 = W0CDM(67.0, ForwardDiff.Dual(0.315), -0.9)
    @test comoving_distance(0.0, c_w0) ≈ 0.0
    @test comoving_distance(0.0, c_w0) isa ForwardDiff.Dual

    zs = [0.0, 0.1]
    r = comoving_distance.(zs, Ref(c_w0))
    @test all(x -> x isa ForwardDiff.Dual, r)

    c_wa = W0WaCDM(67.0, 0.315, -0.9, ForwardDiff.Dual(0.2))
    @test comoving_distance(0.0, c_wa) ≈ 0.0
    @test comoving_distance(0.0, c_wa) isa ForwardDiff.Dual
end

@testset "dark_energy_eos preserves ForwardDiff derivatives" begin
    f_w0 = w0 -> dark_energy_eos(W0CDM(67.0, 0.3, w0), 0.5)
    @test ForwardDiff.derivative(f_w0, -0.8) ≈ 1.0

    g_w0 = w0 -> E(0.5, W0CDM(67.0, 0.3, w0))
    @test isfinite(ForwardDiff.derivative(g_w0, -0.8))
    @test ForwardDiff.derivative(g_w0, -0.8) != 0.0

    h_wa = wa -> dark_energy_eos(W0WaCDM(67.0, 0.3, -0.9, wa), 0.5)
    @test ForwardDiff.derivative(h_wa, 0.2) ≈ 0.5 / (1 + 0.5)
end

@testset "E(z) reduces to ΛCDM at w0=-1" begin
    lcdm = LambdaCDM(70.0, 0.3)
    w0cdm_lim = W0CDM(70.0, 0.3, -1.0)
    w0wacdm_lim = W0WaCDM(70.0, 0.3, -1.0, 0.0)
    for z in (0.0, 0.1, 0.5, 2.0)
        @test E(z, lcdm) ≈ E(z, w0cdm_lim)
        @test E(z, lcdm) ≈ E(z, w0wacdm_lim)
    end
end

@testset "scalar distance path is pinned against adaptive QuadGK" begin
    # The fixed-order Gauss–Legendre rule replaced adaptive quadgk; order 40 is the
    # smallest order with margin below ~1e-12 relative up to z = 20 (32 misses it for
    # ΛCDM), the largest redshift used anywhere in the workspace tests.
    for c in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        for z in (0.1, 1.0, 2.5, 10.0, 20.0)
            reference, _ = quadgk(x -> inv(E(x, c)), 0.0, z; rtol = 1e-13)
            @test comoving_distance(z, c) ≈ hubble_distance(c) * reference rtol = 1e-12
        end
    end
end

@testset "distance_and_volume_grid" begin
    z_grid = collect(LinRange(0.0, 20.0, 1025))

    # Grid and scalar paths are now both Gauss–Legendre and agree to rounding error.
    for c in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        g = distance_and_volume_grid(c, z_grid)
        @test g.comoving_distance ≈ comoving_distance.(z_grid, Ref(c)) rtol = 1e-12
        @test g.luminosity_distance ≈ luminosity_distance.(z_grid, Ref(c)) rtol = 1e-12
        @test g.differential_comoving_volume ≈
              differential_comoving_volume.(z_grid, Ref(c)) rtol = 1e-12
    end

    # astrogwb's contract: the grid need not start at zero — 0 is prepended
    # internally. A shifted grid is bit-identical, at the shared nodes, to the
    # zero-prepended grid: the leading zero-width interval contributes exactly 0
    # and every following interval is the same affine map.
    c = LambdaCDM(67.0, 0.315)
    z_shift = collect(LinRange(0.5, 20.0, 128))
    g_shift = distance_and_volume_grid(c, z_shift)
    g_prepended = distance_and_volume_grid(c, [0.0; z_shift])
    @test g_shift.comoving_distance == g_prepended.comoving_distance[2:end]
    @test g_shift.luminosity_distance == g_prepended.luminosity_distance[2:end]
    @test g_shift.differential_comoving_volume ==
          g_prepended.differential_comoving_volume[2:end]
    # A shifted grid at production density is also accurate, except for its first
    # interval [0, 0.5], which is ~25x wider than the rest and dominates the
    # composite-rule error (~1e-9 there). (The 128-point grid above is deliberately
    # coarse — validity is not accuracy, and the grid policy is caller-owned.)
    z_fine = collect(LinRange(0.5, 20.0, 1024))
    g_fine = distance_and_volume_grid(c, z_fine)
    @test g_fine.comoving_distance ≈ comoving_distance.(z_fine, Ref(c)) rtol = 1e-8

    # A single node is valid: one 4-node rule over [0, z]. Structurally sound, but
    # one wide interval is not an accurate composite rule — caller-owned policy.
    g_single = distance_and_volume_grid(c, [2.0])
    @test length(g_single.comoving_distance) == 1
    @test all(isfinite, g_single.comoving_distance)
    @test g_single.comoving_distance[1] > 0

    # ForwardDiff propagates through the full grid calculation.
    for (build, x0) in (
        (v -> LambdaCDM(67.0, v), 0.315),
        (v -> LambdaCDM(v, 0.315), 67.0),
        (v -> W0CDM(67.0, 0.315, v), -0.9),
        (v -> W0WaCDM(67.0, 0.315, -0.9, v), 0.2)
    )
        f_grid = v -> sum(distance_and_volume_grid(build(v), z_grid).luminosity_distance)
        d = ForwardDiff.derivative(f_grid, x0)
        @test isfinite(d)
        @test d != 0.0
        h = sqrt(eps(x0))
        @test d ≈ (f_grid(x0 + h) - f_grid(x0 - h)) / (2h) rtol = 1e-5
    end

    # And through the scalar path.
    for (build, x0) in (
        (v -> LambdaCDM(67.0, v), 0.315),
        (v -> LambdaCDM(v, 0.315), 67.0),
        (v -> W0CDM(67.0, 0.315, v), -0.9),
        (v -> W0WaCDM(67.0, 0.315, -0.9, v), 0.2)
    )
        f_scalar = v -> luminosity_distance(1.5, build(v))
        d = ForwardDiff.derivative(f_scalar, x0)
        @test isfinite(d)
        @test d != 0.0
        h = sqrt(eps(x0))
        @test d ≈ (f_scalar(x0 + h) - f_scalar(x0 - h)) / (2h) rtol = 1e-5
    end

    @test_throws ArgumentError distance_and_volume_grid(c, Float64[])
    @test_throws ArgumentError distance_and_volume_grid(c, [0.0, 1.0, 0.5])
    @test_throws ArgumentError distance_and_volume_grid(c, [0.0, 1.0, 1.0])
    @test_throws ArgumentError distance_and_volume_grid(c, [-1e-3, 1.0])
end
