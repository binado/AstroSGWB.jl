using LinearAlgebra

function importance_weights(
    log_ratio::AbstractVector{<:Real},
    dgw_fid_sq::AbstractVector{<:Real},
    dgw_theta_sq::AbstractVector{<:Real},
)
    length(log_ratio) == length(dgw_fid_sq) == length(dgw_theta_sq) || throw(
        ArgumentError("importance weight inputs must have matching lengths"),
    )
    return exp.(log_ratio) .* dgw_fid_sq ./ dgw_theta_sq
end

function spectral_density_from_cache(
    cached_flux_over_dgw2::AbstractMatrix{<:Real},
    weights::AbstractVector{<:Real},
    number_of_sources::Real,
    observation_time_sec::Real,
)
    size(cached_flux_over_dgw2, 1) == length(weights) || throw(
        ArgumentError("cached_flux_over_dgw2 row count must match weight count"),
    )
    mean_flux = cached_flux_over_dgw2' * weights ./ size(cached_flux_over_dgw2, 1)
    rate = number_of_sources / observation_time_sec
    return 0.4 .* rate .* mean_flux
end
