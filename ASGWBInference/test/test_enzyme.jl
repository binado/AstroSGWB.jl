using Test
using ASGWB
using ASGWBInference: build_turing_model
using ADTypes
using Enzyme
using LogDensityProblems
using LogDensityProblemsAD
using Turing.DynamicPPL: VarInfo, link, LogDensityFunction, getlogjoint_internal

# Enzyme cannot statically prove the type of the cosmology-cache /
# DataInterpolations construction path, so we permit its best-effort type
# analysis and validate correctness numerically against ForwardDiff below. This
# is a global setting and must be enabled before the first differentiation.
Enzyme.API.looseTypeAnalysis!(true)

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_test_cache.jl"))
end
if !@isdefined PARITY_PRIORS
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))
end

@testset "Enzyme reverse-mode gradient matches ForwardDiff" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    problem, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
    priors = PARITY_PRIORS

    model = build_turing_model(problem, C, ctx, priors; track = false)
    vi = link(VarInfo(model), model)
    lf = LogDensityFunction(model, getlogjoint_internal, vi)
    z = convert(Vector{Float64}, vi[:])

    fd = LogDensityProblemsAD.ADgradient(:ForwardDiff, lf)
    v_fd, g_fd = LogDensityProblems.logdensity_and_gradient(fd, z)

    adtype = ADTypes.AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const
    )
    en = LogDensityProblemsAD.ADgradient(adtype, lf)
    v_en, g_en = LogDensityProblems.logdensity_and_gradient(en, z)

    @test v_en ≈ v_fd rtol = 1e-6
    @test g_en ≈ g_fd rtol = 1e-5
end
