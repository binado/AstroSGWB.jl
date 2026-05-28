abstract type ProposalSampleBundle end

"""
    ProposalData

Proposal-sample bundle for the importance-sampling problem.

Matrix layouts:
- `intrinsic_vector` is `(n_samples, n_intrinsic)` (rows = samples, columns = intrinsic sites).
- `cached_flux_over_dgw2` is `(n_freq, n_samples)` (column-major friendly: each proposal
  sample is a contiguous column; `fluxes * weights` contracts to a per-frequency vector).
"""
struct ProposalData
    intrinsic_site_order::Vector{String}
    samples::FullBNSSamplesSoA
    log_prob::Vector{Float64}
    intrinsic_vector::Matrix{Float64}
    cached_flux_over_dgw2::Matrix{Float64}
    dgw_fid_sq::Vector{Float64}
end

"""
    RedshiftGridCache

Precomputed redshift-grid state attached to an [`ImportanceSamplingProblem`](@ref):
the fixed redshift grid, interpolation metadata for proposal redshifts on that grid,
and cached hyperparameter-independent full-BNS intrinsic log-probability terms
(mass, spins, tidal deformability). Redshift log-probability is evaluated from the
live [`RedshiftPrior`](@ref) each likelihood call.
"""
struct RedshiftGridCache
    redshift_grid::Vector{Float64}
    sample_interpolant::SampleInterpolant
    cached_intrinsic_log_prob::Vector{Float64}
end

"""
    ImportanceSamplingProblem

In-memory importance-sampling context. See [`importance_sampling_problem`](@ref) and
[`load_problem`](@ref). The structural forward `model` and canonical
`fiducial_hyperparameters` come from [`ModelConfig`](@ref) or direct in-memory
construction.

`redshift_cache` groups the fixed grid, per-sample interpolation metadata, and cached
hyperparameter-independent full-BNS intrinsic terms (mass, spins, tidal deformability);
redshift terms are evaluated from the live prior each step.
"""
struct ImportanceSamplingProblem{M <: AbstractASGWBModel}
    proposal::ProposalData
    observation::ObservationConfig
    model::M
    fiducial_hyperparameters::NamedTuple
    redshift_prior_spec::RedshiftPriorSpec
    redshift_cache::RedshiftGridCache
    local_merger_rate::Float64
    strategy::FullBNS
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.proposal.samples)

function _validate_strategy_bundle(strategy::FullBNS, proposal::ProposalData)
    proposal.samples isa FullBNSSamplesSoA ||
        throw(ArgumentError("proposal samples must match the FullBNSSamplesSoA layout"))
    return nothing
end

function build_redshift_grid_cache(
        proposal::ProposalData,
        redshift_prior_spec::RedshiftPriorSpec;
        intrinsic_prior_factory = intrinsic_prior
)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    _validate_strategy_bundle(strategy, proposal)
    prior = intrinsic_prior_factory(strategy)
    validate_batch(prior, proposal.samples)
    cached_log_prob = logpdf(prior, proposal.samples)
    z_grid = redshift_grid(redshift_prior_spec)
    interp = SampleInterpolant(proposal.samples.redshift, z_grid)
    return RedshiftGridCache(z_grid, interp, cached_log_prob)
end

"""
    importance_sampling_problem(
        proposal, observation, model, fiducial_hyperparameters,
        redshift_prior_spec, local_merger_rate,
    ) -> ImportanceSamplingProblem

Validates [`IntrinsicPriorStrategy`](@ref) against the proposal sample bundle type.
"""
function importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        model::M,
        fiducial_hyperparameters::NamedTuple,
        redshift_prior_spec::RedshiftPriorSpec,
        local_merger_rate::Real;
        intrinsic_prior_factory = intrinsic_prior
) where {M <: AbstractASGWBModel}
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    redshift_cache = build_redshift_grid_cache(
        proposal,
        redshift_prior_spec;
        intrinsic_prior_factory = intrinsic_prior_factory
    )
    return ImportanceSamplingProblem(
        proposal,
        observation,
        model,
        fiducial_hyperparameters,
        redshift_prior_spec,
        redshift_cache,
        Float64(local_merger_rate),
        strategy
    )
end
