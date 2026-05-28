function evaluate_model_terms(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem
)
    c = cosmology(problem.cosmology_type, Λ)
    cosmology_cache = CosmologyCache(c, problem.redshift_grid)
    prior = single_event_prior(problem.population, c, Λ)
    iw = compute_importance_weights(problem, Λ, cosmology_cache, prior)
    redshift_prior_dist = _redshift_prior_distribution(prior).prior
    rate = merger_rate_per_sec(
        redshift_prior_dist,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    sd = spectral_density(problem.proposal.cached_flux_over_dgw2, rate; weights = iw.weights)
    return merge(iw,
        (
            redshift_integral = redshift_integral(redshift_prior_dist),
            expected_number_of_sources = rate * problem.observation.observation_time_sec,
            spectral_density = sd,
            spectral_density_in_band = sd[problem.observation.in_band_mask]
        ))
end

function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem;
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    evaluation = evaluate_model_terms(Λ, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end

function fiducial_spectral_density(problem::ImportanceSamplingProblem)
    return evaluate_model_terms(fiducial_hyperparameters(problem), problem).spectral_density
end

function fiducial_redshift_integral(problem::ImportanceSamplingProblem)
    Λ = problem.fiducial_hyperparameters
    c = cosmology(problem.cosmology_type, Λ)
    prior = single_event_prior(problem.population, c, Λ)
    return Float64(redshift_integral(_redshift_prior_distribution(prior).prior))
end

function fiducial_redshift_integral(
        ::Type{C},
        population::M,
        Λ::NamedTuple
) where {C <: AbstractCosmology, M <: PopulationModel}
    c = cosmology(C, Λ)
    prior = single_event_prior(population, c, Λ)
    return Float64(redshift_integral(_redshift_prior_distribution(prior).prior))
end
