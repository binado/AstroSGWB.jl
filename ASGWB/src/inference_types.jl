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
the fixed redshift grid and interpolation metadata for proposal redshifts on that grid.
Redshift log-probability is evaluated from the live [`RedshiftPrior`](@ref) each
likelihood call.
"""
struct RedshiftGridCache
    redshift_grid::Vector{Float64}
    sample_interpolant::SampleInterpolant
end

"""
    ImportanceSamplingProblem{C,M}

In-memory importance-sampling context parameterised by cosmology type `C` and
population model `M`.  Build via [`importance_sampling_problem`](@ref) or
[`load_problem`](@ref).

The redshift grid is fixed to [`DEFAULT_Z_GRID`](@ref) for all evaluations;
[`single_event_prior`](@ref) is called with the live cosmology and hyperparameters
each step.
"""
struct ImportanceSamplingProblem{C <: AbstractCosmology, M <: PopulationModel}
    proposal::ProposalData
    observation::ObservationConfig
    cosmology_type::Type{C}
    population::M
    fiducial_hyperparameters::NamedTuple
    redshift_grid::Vector{Float64}
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
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    _validate_strategy_bundle(strategy, proposal)
    interp = SampleInterpolant(proposal.samples.redshift, z_grid)
    return RedshiftGridCache(collect(Float64, z_grid), interp)
end

"""
    importance_sampling_problem(
        proposal, observation, cosmology_type, population, fiducial_hyperparameters,
        local_merger_rate; z_grid,
    ) -> ImportanceSamplingProblem

Construct an importance-sampling problem from in-memory objects.  The redshift
grid defaults to [`DEFAULT_Z_GRID`](@ref) and may be overridden via `z_grid`.
"""
function importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        cosmology_type::Type{C},
        population::M,
        fiducial_hyperparameters::NamedTuple,
        local_merger_rate::Real;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C <: AbstractCosmology, M <: PopulationModel}
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    redshift_cache = build_redshift_grid_cache(proposal, z_grid)
    return ImportanceSamplingProblem(
        proposal,
        observation,
        C,
        population,
        fiducial_hyperparameters,
        redshift_cache.redshift_grid,
        redshift_cache,
        Float64(local_merger_rate),
        strategy
    )
end
