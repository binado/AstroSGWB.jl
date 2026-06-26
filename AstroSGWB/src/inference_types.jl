"""
    ImportanceSamplingProblem{M <: PopulationModel}

Pure importance-sampling specification: the minimal raw inputs needed to define the
forward model, with no derived arrays, no cosmology, and no detector state.

Fields:
- `population_model::M` — the [`PopulationModel`](@ref) whose `single_event_prior` is the
  importance-sampling target / proposal density.
- `fluxes::Matrix{Float64}` — raw per-sample fluxes `|h₊|² + |h×|²` from the waveform
  catalog, *before* the fiducial `(D_L/D_gw)²` scaling, `(nfreq, nsamples)`.
- `samples::NamedTuple` — restructured per-event parameters (struct-of-arrays). Keys must
  include every field of `single_event_prior(...).dists` (e.g. `mass`, `redshift`, `χ₁`,
  `χ₂`, `Λ₁`, `Λ₂`); extra keys are allowed. Each field stores samples along its last
  axis (vectors for univariate components, `(dim, n)` matrices for multivariate); see
  [`validate_samples`](@ref).
- `fiducial_hyperparameters::NamedTuple` — canonical fiducial hyperparameters; the
  cosmology + propagation + population state at which the proposal caches are built.

The raw `fluxes` are used directly in the spectral-density contraction; the full distance
correction `(D_L,fid/D_L,θ)²·(1/Ξ_θ²)` lives in the importance weights, not in the fluxes.
All derived/`Λ`-independent caches (proposal log-prob, `dl_fid_sq`, redshift interpolant)
are owned by the caller's prepared model — the cosmology-specific half of what was once a
`ModelContext` — and consumed inside the model's [`merger_rate_and_log_weights`](@ref) joint.
The cosmology family is the model's own concern, never stored here.
"""
struct ImportanceSamplingProblem{M <: PopulationModel}
    population_model::M
    fluxes::Matrix{Float64}
    samples::NamedTuple
    fiducial_hyperparameters::NamedTuple
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.samples)

"""
    with_redshift_interpolant(samples::NamedTuple, query::GridQuery) -> NamedTuple

Attach the proposal redshift [`GridQuery`](@ref) to the `redshift` field of `samples`,
wrapping it in a [`SampleField`](@ref) so the redshift logpdf reuses the precomputed
per-sample grid locations instead of re-searching the grid every gradient evaluation.
Model authors call this when assembling the proposal caches and inside their
[`merger_rate_and_log_weights`](@ref) joint.
"""
function with_redshift_interpolant(samples::NamedTuple, query::GridQuery)
    return merge(samples, (; redshift = SampleField(samples.redshift, query)))
end
