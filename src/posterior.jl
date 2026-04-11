function target_log_prob_samples(theta, problem::ImportanceSamplingProblem)
    bundle = build_redshift_grid_bundle(theta, problem.redshift_prior_spec)
    redshift_log_prob = log_prob_from_bundle.(redshift(problem), Ref(bundle))
    intrinsic_log_prob = _intrinsic_log_prob(problem.strategy, problem, redshift_log_prob)
    return intrinsic_log_prob, bundle
end

function _intrinsic_log_prob(
    ::RedshiftOnly, ::ImportanceSamplingProblem, redshift_log_prob::AbstractVector{<:Real},
)
    return redshift_log_prob
end

function _intrinsic_log_prob(
    ::FullBNS, problem::ImportanceSamplingProblem, redshift_log_prob::AbstractVector{<:Real},
)
    return bns_intrinsic_log_prob_samples(problem, redshift_log_prob)
end

function evaluate_importance_terms(theta, problem::ImportanceSamplingProblem)
    z = redshift(problem)
    d_l = luminosity_distance.(z, theta.H0, theta.Omega_m)
    dgw_theta = gravitational_wave_distance.(z, d_l, theta.chi0, theta.chin)
    dgw_theta_sq = dgw_theta .^ 2

    target_log_prob, redshift_bundle = target_log_prob_samples(theta, problem)
    log_ratio = target_log_prob .- problem.proposal.log_prob
    weights = importance_weights(log_ratio, problem.proposal.dgw_fid_sq, dgw_theta_sq)

    redshift_integral = redshift_bundle.norm
    number_of_sources = expected_number_of_events(
        problem.local_merger_rate,
        redshift_integral,
        problem.observation.observation_time_yr,
    )

    spectral_density = spectral_density_from_cache(
        problem.proposal.cached_flux_over_dgw2,
        weights,
        number_of_sources,
        problem.observation.observation_time_sec,
    )
    spectral_density_in_band = spectral_density[problem.observation.in_band_mask]

    return (
        dgw_theta_sq=dgw_theta_sq,
        target_log_prob=target_log_prob,
        log_ratio=log_ratio,
        weights=weights,
        redshift_integral=redshift_integral,
        expected_number_of_sources=number_of_sources,
        spectral_density=spectral_density,
        spectral_density_in_band=spectral_density_in_band,
    )
end

function loglikelihood(
    theta,
    problem::ImportanceSamplingProblem;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    evaluation = evaluate_importance_terms(theta, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function logposterior(
    theta,
    problem::ImportanceSamplingProblem,
    priors::AbstractDict{<:AbstractString,<:Distribution};
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    return logprior(theta, priors) + loglikelihood(
        theta,
        problem;
        observed_spectral_density=observed_spectral_density,
    )
end
