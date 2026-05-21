using Test
using Bijectors
using Distributions: product_distribution, Uniform
using ASGWB: hyperparameter_order, coerce_hyperparameters

"""Reorder hyperparameters to match `keys(prior.dists)` for Bijectors."""
function _align_hyperparameters(θ::NamedTuple, prior)
    return (; (k => θ[k] for k in hyperparameter_order(prior))...)
end

@testset "hyperparameter_order follows prior construction order" begin
    prior_a = product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.0, 1.0),
        γ = Uniform(0.0, 5.0),
        Ξ₀ = Uniform(0.0, 2.0),
        Ξₙ = Uniform(-1.0, 1.0),
        κ = Uniform(0.0, 10.0),
        zpeak = Uniform(0.0, 5.0)
    ))
    prior_b = product_distribution((
        zpeak = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        Ξₙ = Uniform(-1.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        γ = Uniform(0.0, 5.0),
        Ωm = Uniform(0.0, 1.0),
        H0 = Uniform(20.0, 140.0)
    ))
    @test hyperparameter_order(prior_a) != hyperparameter_order(prior_b)

    θ = coerce_hyperparameters(;
        H0 = 70.0,
        Ωm = 0.3,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.0,
        κ = 3.0,
        zpeak = 1.5
    )
    @test collect(Bijectors.link(prior_a, _align_hyperparameters(θ, prior_a))) !=
          collect(Bijectors.link(prior_b, _align_hyperparameters(θ, prior_b)))
end
