using LinearAlgebra

"""
    merger_rate_and_log_weights(model, Λ::NamedTuple, samples) -> (rate, log_weights)

The single model-dispatched importance-sampling contract that makes the forward-model
surface cosmology-agnostic. **Model authors implement this method outside the package**, by
dispatch on their concrete prepared model type. It fuses the two cosmology-specific steps
— the detector-frame [`merger_rate`](@ref) and the per-sample importance weighting — into
one self-contained call:

- `rate::Real` — the detector-frame merger rate in events/sec at hyperparameters `Λ`.
- `log_weights::AbstractVector` — per-sample **log** importance weights (consumers
  exponentiate immediately before `spectral_density`/`normalized_ess`).

A typical implementation assembles the exported kernels: rebuild the [`CosmologyCache`](@ref)
and [`single_event_prior`](@ref) at `Λ`, form the prior log-ratio with [`logprobdiff`](@ref)
against the prepared proposal caches, call [`importance_log_weights`](@ref) for the weights,
and [`merger_rate`](@ref) for the rate. Everything *above* this boundary (cache, prior,
propagation factor `Ξ(z)`, distances) is cosmology-specific and lives on the model;
everything *below* it (`spectral_density`, Gaussian likelihoods, SNR tracking) is
cosmology-agnostic.

See also [`full_hyperparameters`](@ref)`(model)`, the companion method model authors
implement to declare the flat HMC/Turing vector layout.
"""
function merger_rate_and_log_weights end

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
