using Test
using Turing
using Distributions: product_distribution, Uniform
using AstroSGWBInference: build_turing_model, hyperparameters,
                          merger_rate_and_log_weights
import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "..", "..", "AstroSGWB", "test", "parity_test_cache.jl"))
end
if !@isdefined PARITY_PRIORS
    include(joinpath(@__DIR__, "..", "..", "AstroSGWB", "test", "parity_fixtures.jl"))
end

struct InvalidNamesModel{Names}
    names::Names
end

hyperparameters(model::InvalidNamesModel) = model.names
function merger_rate_and_log_weights(::InvalidNamesModel, Λ, samples)
    (1.0, zeros(length(samples.redshift)))
end

@testset "caller-owned model hyperparameter contract" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    model = loaded.model
    expected = Set(hyperparameters(model))

    reversed_prior = product_distribution((;
        (name => PARITY_PRIORS.dists[name]
    for name in reverse(keys(PARITY_PRIORS.dists)))...))
    reversed_fiducials = (;
        (name => loaded.fiducials[name] for name in reverse(keys(loaded.fiducials)))...)

    @test Set(keys(reversed_prior.dists)) == expected
    @test keys(reversed_prior.dists) != hyperparameters(model)
    @test build_turing_model(
        model,
        loaded.fluxes,
        loaded.samples,
        reversed_fiducials,
        loaded.observation,
        reversed_prior
    ) !== nothing

    missing_prior = product_distribution((;
        (name => PARITY_PRIORS.dists[name]
    for name in keys(PARITY_PRIORS.dists) if name != :zpeak)...))
    extra_prior = product_distribution(merge(PARITY_PRIORS.dists, (extra = Uniform(0, 1),)))
    missing_fiducials = Base.structdiff(loaded.fiducials, NamedTuple{(:zpeak,)})
    extra_fiducials = merge(loaded.fiducials, (extra = 1.0,))

    @test_throws ArgumentError build_turing_model(
        model, loaded.fluxes, loaded.samples, loaded.fiducials,
        loaded.observation, missing_prior)
    @test_throws ArgumentError build_turing_model(
        model, loaded.fluxes, loaded.samples, loaded.fiducials,
        loaded.observation, extra_prior)
    @test_throws ArgumentError build_turing_model(
        model, loaded.fluxes, loaded.samples, missing_fiducials,
        loaded.observation, PARITY_PRIORS)
    @test_throws ArgumentError build_turing_model(
        model, loaded.fluxes, loaded.samples, extra_fiducials,
        loaded.observation, PARITY_PRIORS)

    for invalid in (
        InvalidNamesModel((:x, :x)),
        InvalidNamesModel((:x, "y")),
        InvalidNamesModel([:x])
    )
        err = try
            build_turing_model(
                invalid, loaded.fluxes, loaded.samples, (x = 1.0,),
                loaded.observation, product_distribution((x = Uniform(0, 2),)))
            nothing
        catch exception
            exception
        end
        @test err isa ArgumentError
        @test occursin("hyperparameters(model)", sprint(showerror, err))
    end
end
