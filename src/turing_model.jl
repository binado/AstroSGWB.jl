using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Turing

@model function asgwb_importance_turing_model(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors,
    observed_in_band::AbstractVector{<:Real},
)
    H0 ~ priors.H0
    Omega_m ~ priors.Omega_m
    chi0 ~ priors.chi0
    chin ~ priors.chin
    gamma ~ priors.gamma
    kappa ~ priors.kappa
    z_peak ~ priors.z_peak

    h = HyperParameters(
        CosmologicalParameters(H0, Omega_m),
        ModifiedPropagationParameters(chi0, chin),
        MadauDickinsonParameters(gamma, kappa, z_peak),
    )

    evaluation = evaluate_importance_terms(h, problem)

    observed_in_band ~ MvNormal(
        evaluation.spectral_density_in_band,
        Diagonal(problem.observation.sgwb_scale_in_band .^ 2),
    )

    return (;
        number_of_sources=evaluation.expected_number_of_sources,
        spectral_density=evaluation.spectral_density,
    )
end

function build_turing_model(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    return asgwb_importance_turing_model(
        problem,
        priors,
        observed_spectral_density[problem.observation.in_band_mask],
    )
end

function sample_with_turing(
    problem::ImportanceSamplingProblem,
    priors::InferencePriors,
    theta0::HyperParameters;
    n_adapts::Int=25,
    n_samples::Int=25,
    target_acceptance::Float64=0.8,
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    model = build_turing_model(
        problem,
        priors;
        observed_spectral_density=observed_spectral_density,
    )
    chain = sample(
        model,
        Turing.NUTS(n_adapts, target_acceptance),
        n_samples;
        progress=false,
        initial_params=InitFromParams(as_flat_constrained(theta0)),
    )
    return chain, model
end
