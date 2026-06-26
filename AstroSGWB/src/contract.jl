"""
    merger_rate_and_log_weights(model, Λ::NamedTuple, samples) -> (rate, log_weights)

The single model-dispatched joint that makes the inference surface cosmology-agnostic.
**Model authors implement this method outside the package**, by dispatch on their concrete
prepared model type. It fuses the two cosmology-specific steps — the detector-frame
[`merger_rate`](@ref) and the per-sample importance weighting — into one self-contained
call:

- `rate::Real` — the detector-frame merger rate in events/sec at hyperparameters `Λ`.
- `log_weights::AbstractVector` — per-sample **log** importance weights (the package-side
  consumers exponentiate immediately before `spectral_density`/`normalized_ess`).

A typical implementation assembles the exported kernels: rebuild the [`CosmologyCache`](@ref)
and [`single_event_prior`](@ref) at `Λ`, form the prior log-ratio with [`logprobdiff`](@ref)
against the prepared proposal caches, call [`importance_log_weights`](@ref) for the weights,
and [`merger_rate`](@ref) for the rate. Everything *above* this boundary (cache, prior,
propagation factor `Ξ(z)`, distances) is cosmology-specific and lives on the model;
everything *below* it (`spectral_density`, the Gaussian likelihood, SNR tracking) is
cosmology-agnostic and lives in the package.

See also [`full_hyperparameters`](@ref)`(model)`, the companion method model authors
implement to declare the flat HMC/Turing vector layout.
"""
function merger_rate_and_log_weights end
