using GWDistributions
using Test

@testset "time conversions" begin
    @test year_to_second(1.0) ≈ 365.25 * 24 * 3600
    @test second_to_year(year_to_second(2.5)) ≈ 2.5
    @test year_to_second(1.0) == JULIAN_YEAR_SEC
end

@testset "private trapezoid helpers" begin
    # Piecewise-linear y on a unit grid: the trapezoid rule is exact, and every cell
    # integral is exact in Float64, so equality is bitwise.
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1, ∫₀³ = 12
    @test GWDistributions._trapz(y, x) == 12.0
    cumulative = GWDistributions._cumtrapz(y, x)
    @test first(cumulative) == 0.0
    @test last(cumulative) === GWDistributions._trapz(y, x)

    # Non-uniform grid, still linear: exact against the analytic integral.
    xu = [0.0, 0.5, 1.5, 4.0]
    yu = [1.0, 2.0, 4.0, 9.0]  # y = 2x + 1, ∫₀⁴ = 20
    @test GWDistributions._trapz(yu, xu) ≈ 20.0 atol = 1e-14
    @test first(GWDistributions._cumtrapz(yu, xu)) == 0.0

    @test_throws ArgumentError GWDistributions._trapz([1.0], [0.0, 1.0])
    @test_throws ArgumentError GWDistributions._cumtrapz([1.0], [0.0, 1.0])
end
