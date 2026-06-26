"""
    merger_rate(prior, local_merger_rate, observation_time) -> Float64

Detector-frame merger rate in events/sec from an already-built `single_event_prior`. The
model's [`merger_rate_and_log_weights`](@ref) joint constructs the prior once and shares it
with the importance-weight kernel, so this is the form called directly (no cosmology/prior
rebuild). The redshift `.dists.redshift.prior` reach-through stays here rather than at every
call site.

`observation_time` is the observation duration in years (Julian year).
"""
function merger_rate(
        prior::ProductNamedTupleDistribution,
        local_merger_rate::Real,
        observation_time::Real
)
    return merger_rate_per_sec(
        prior.dists.redshift.prior,
        local_merger_rate,
        observation_time
    )
end

"""
    fiducial_spectral_density(model, fluxes, samples, fiducial_hyperparameters) -> Vector

Synthesize the default observed strain spectral density from catalog fluxes at the
fiducial hyperparameters. Callers that omit `observed` in
`AstroSGWBInference.build_turing_model` should use this spectrum so modified-propagation
factors `Ξ(z)` are applied consistently.
"""
function fiducial_spectral_density(
        model,
        fluxes::AbstractMatrix{<:Real},
        samples::NamedTuple,
        fiducial_hyperparameters::NamedTuple
)
    rate,
    log_weights = merger_rate_and_log_weights(model, fiducial_hyperparameters, samples)
    return spectral_density(fluxes, rate; weights = exp.(log_weights))
end
