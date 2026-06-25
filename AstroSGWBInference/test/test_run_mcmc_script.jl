using Test
using AstroSGWB

@testset "run_mcmc script population model" begin
    include(joinpath(@__DIR__, "..", "..", "scripts", "run_mcmc.jl"))

    pop = AstroSGWBRunMCMC.BNSPopulationModel()
    Λ = (
        H0 = 67.0,
        Ωm = 0.315,
        w0 = -1.0,
        Ξ₀ = 1.0,
        Ξₙ = 2.0,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.0
    )
    cosmo = AstroSGWB.cosmology(AstroSGWBRunMCMC.C, Λ)
    cache = AstroSGWB.CosmologyCache(cosmo, collect(LinRange(0.0, 10.0, 64)))

    prior = AstroSGWB.single_event_prior(pop, cache, Λ)

    @test keys(prior.dists) == (:mass, :redshift, :χ₁, :χ₂, :Λ₁, :Λ₂)
end
