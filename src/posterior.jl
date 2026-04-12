function target_log_prob_samples(h::HyperParameters, problem::ImportanceSamplingProblem)
    bundle = build_redshift_grid_bundle(h, problem.redshift_prior_spec)
    redshift_log_prob = log_prob_from_bundle.(redshift(problem), Ref(bundle))
    return bns_intrinsic_log_prob_samples(problem.proposal.samples, redshift_log_prob), bundle
end

function evaluate_importance_terms(h::HyperParameters, problem::ImportanceSamplingProblem)
    z = redshift(problem)
    d_l = luminosity_distance.(z, h.cosmological.H0, h.cosmological.Omega_m)
    dgw_theta = gravitational_wave_distance.(z, d_l, h.propagation.chi0, h.propagation.chin)
    dgw_theta_sq = dgw_theta .^ 2

    target_log_prob, redshift_bundle = target_log_prob_samples(h, problem)
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
    h::HyperParameters,
    problem::ImportanceSamplingProblem;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    evaluation = evaluate_importance_terms(h, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function logposterior(
    h::HyperParameters,
    problem::ImportanceSamplingProblem,
    priors::InferencePriors;
    observed_spectral_density::AbstractVector{<:Real}=problem.observation.fiducial_spectral_density,
)
    return logprior(h, priors) + loglikelihood(
        h,
        problem;
        observed_spectral_density=observed_spectral_density,
    )
end

"""
    fiducial_hyperparameters(problem::ImportanceSamplingProblem) -> HyperParameters

Build [`HyperParameters`](@ref) from the cache’s [`ProposalFiducialParameters`](@ref)
and [`RedshiftPriorSpec`](@ref). Same rules as [`hyperparameters_from_fiducial`](@ref)
(population scalars on the proposal fiducial dict when the prior family requires them).
"""
function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    return hyperparameters_from_fiducial(
        problem.fiducial_parameters,
        problem.redshift_prior_spec,
    )
end

"""
    fiducial_spectral_density(problem::ImportanceSamplingProblem) -> Vector{Float64}

Predicted isotropic SGWB spectrum ``Ω_\\mathrm{GW}(f)`` at [`fiducial_hyperparameters`](@ref),
using [`evaluate_importance_terms`](@ref) (cached flux, importance weights, and event rate).
"""
function fiducial_spectral_density(problem::ImportanceSamplingProblem)
    h = fiducial_hyperparameters(problem)
    return evaluate_importance_terms(h, problem).spectral_density
end

"""
    fiducial_redshift_integral(problem::ImportanceSamplingProblem) -> Float64

[`build_redshift_grid_bundle`](@ref) norm at [`fiducial_hyperparameters`](@ref) (matches
`problem.redshift_integral_fiducial` when that field was set from the fiducial population).
"""
function fiducial_redshift_integral(problem::ImportanceSamplingProblem)
    return fiducial_redshift_integral(problem.fiducial_parameters, problem.redshift_prior_spec)
end
