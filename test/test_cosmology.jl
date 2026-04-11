using HDF5
using Test

@testset "basic cosmology helpers" begin
    @test E(0.0, 0.315) ≈ 1.0
    @test comoving_distance(0.0, 67.0, 0.315) ≈ 0.0

    z = [0.0, 0.1, 0.2]
    d_l = luminosity_distance.(z, 67.0, 0.315)
    @test d_l[1] ≈ 0.0
    @test d_l[3] > d_l[2] > d_l[1]

    d_gw = gravitational_wave_distance.([0.1, 0.2], [10.0, 20.0], 1.0, 0.0)
    @test d_gw ≈ [10.0, 20.0]
end

@testset "cosmology parity fixtures" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "cosmology_parity.h5")

    h5open(fixture_path, "r") do file
        z_grid = vec(Float64.(read(file["z_grid"])))
        cases = file["cases"]

        for case_name in sort!(collect(keys(cases)))
            case_group = cases[case_name]
            H0 = Float64(read(case_group["H0"]))
            Omega_m = Float64(read(case_group["Omega_m"]))
            chi0 = Float64(read(case_group["chi0"]))
            chin = Float64(read(case_group["chin"]))

            expected_dl = vec(Float64.(read(case_group["luminosity_distance"])))
            expected_dvc = vec(Float64.(read(case_group["differential_comoving_volume"])))
            expected_dgw = vec(Float64.(read(case_group["gravitational_wave_distance"])))

            @test luminosity_distance.(z_grid, H0, Omega_m) ≈ expected_dl rtol = 1e-6
            @test differential_comoving_volume.(z_grid, H0, Omega_m) ≈ expected_dvc rtol = 1e-6
            @test gravitational_wave_distance.(z_grid, expected_dl, chi0, chin) ≈ expected_dgw rtol = 1e-6
        end
    end
end
