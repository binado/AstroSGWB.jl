"""
    ImportanceSamplingProblem{M <: PopulationModel}

Pure importance-sampling specification: the minimal raw inputs needed to define the
forward model, with no derived arrays, no cosmology, and no detector state.

Fields:
- `population_model::M` — the [`PopulationModel`](@ref) whose `single_event_prior` is the
  importance-sampling target / proposal density.
- `fluxes::AbstractMatrix{<:Real}` — raw per-sample fluxes `|h₊|² + |h×|²` from the waveform
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
All derived/`Λ`-independent caches (`dl_fid_sq`, proposal log-prob, redshift
interpolant, detector PSDs) live in [`ModelContext`](@ref), built by
[`build_model_context`](@ref). The cosmology family `C` is passed as a call argument,
never stored here.
"""
struct ImportanceSamplingProblem{
    M <: PopulationModel,
    F <: AbstractMatrix{<:Real},
    S <: NamedTuple,
    H <: NamedTuple
}
    population_model::M
    fluxes::F
    samples::S
    fiducial_hyperparameters::H
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.samples)

function _with_redshift_interpolant(samples::NamedTuple, interp::SampleInterpolant)
    return merge(samples, (; redshift = SampleField(samples.redshift, interp)))
end

"""
    ModelContext

Flat catalog-derived cache of all `Λ`-independent derived state for an [`ImportanceSamplingProblem`](@ref),
built once by [`build_model_context`](@ref) and reused by every likelihood/model call:

- proposal caches at the fiducial point: `proposal_prior` (the fiducial
  `single_event_prior` itself, used by [`logprobdiff`](@ref) for the egal component
  skip), `proposal_log_prob` (a `NamedTuple` of per-component log-density vectors),
  and `dl_fid_sq` (squared EM luminosity distance at the fiducial cosmology),
- redshift state: `redshift_grid` and the proposal `sample_interpolant`,
- detector/observation state grouped in [`ObservationContext`](@ref), and
- `local_merger_rate` (events Gpc⁻³ yr⁻¹).

A `ModelContext` is built for a specific cosmology family `C`; calling cached atomics with
a different `C` would silently mix mismatched caches. Coherence is guaranteed by
construction: `build_model_context` and the model that uses it close over a single literal
`C`.
"""
struct ModelContext{
    P <: ProductNamedTupleDistribution,
    L <: NamedTuple,
    D <: AbstractVector{<:Real},
    Z <: AbstractVector{<:Real},
    I <: SampleInterpolant,
    O <: ObservationContext,
    R <: Real
}
    proposal_prior::P
    proposal_log_prob::L
    dl_fid_sq::D
    redshift_grid::Z
    sample_interpolant::I
    observation::O
    local_merger_rate::R
end
