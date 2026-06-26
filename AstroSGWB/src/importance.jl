using LinearAlgebra

# Per-sample *log* importance weight as a sum of three physically independent log-factors:
# the population prior log-ratio `log_ratio[i]`, the FLRW background distance log-ratio
# `log(D_L,fid²) − 2 log(D_L,θ)`, and the modified-propagation log-factor `−2 log(Ξ_θ)`.
# The raw catalog flux carries `1/D_L,fid²`, so adding `log(dl_fid_sq) − 2 log(D_L,θ) − 2 log(Ξ_θ)`
# recovers the physically correct `1/D_gw,θ²` dilution once exponentiated. Kept in log-space so
# the logpdf arithmetic upstream never round-trips through `exp`/`log`.
@inline function _importance_log_weight_at_sample(
        log_ratio::AbstractVector,
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        interp::GridQuery,
        cosmology_cache::CosmologyCache,
        prop::AbstractPropagation,
        sample_index::Integer
)
    d_l = luminosity_distance_at_sample(cosmology_cache, interp, z, sample_index)
    Ξ_theta = gw_em_distance_ratio(z[sample_index], prop)
    return log_ratio[sample_index] + log(dl_fid_sq[sample_index]) -
           2 * log(d_l) - 2 * log(Ξ_theta)
end

"""
    importance_log_weights(log_ratio, dl_fid_sq, z, interp, cache, prop) -> Vector

Per-sample **log** importance weights from explicit arrays: the population prior
log-ratio `log_ratio` (per-sample `log p_target − log p_proposal`), the squared fiducial
EM luminosity distances `dl_fid_sq`, the sample redshifts `z`, the proposal redshift
interpolant `interp`, a [`CosmologyCache`](@ref) at the target hyperparameters, and the
target propagation `prop`. Returns `log_ratio[i] + log(dl_fid_sq[i]) − 2 log(D_L,θ) − 2 log(Ξ_θ)`.

This is the reusable physics kernel that model authors assemble into their
[`merger_rate_and_log_weights`](@ref) joint. Built with `map` over the index range so the
result type stays stable (a properly-typed empty vector for `n == 0`, rather than a `Union`
with `Float64[]`), keeping the AD/`Dual` likelihood path inferrable. Exponentiate
(`exp.(...)`) at the call site for the linear weights `spectral_density`/`normalized_ess`
need.
"""
function importance_log_weights(
        log_ratio::AbstractVector,
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        interp::GridQuery,
        cosmology_cache::CosmologyCache,
        prop::AbstractPropagation
)
    length(z) == length(log_ratio) ||
        throw(ArgumentError("population prior logpdf length must match proposal sample count"))
    return map(eachindex(z)) do i
        _importance_log_weight_at_sample(
            log_ratio, dl_fid_sq, z, interp, cosmology_cache, prop, i)
    end
end
