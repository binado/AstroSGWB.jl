using Distributions

"""
    PopulationModel

Abstract supertype for caller-defined population models.  Concrete subtypes
must implement the two-method contract:

- `hyperparameters(pop) -> NTuple{N, Symbol}` — ordered population parameter names.
- `single_event_prior(pop, cache::CosmologyCache, Λ) -> ProductNamedTupleDistribution`
  — per-event distribution conditioned on the cosmology carried by `cache` and
  hyperparameters `Λ`. Build the redshift component with
  `redshift_prior(sf_model, cache, Λ)` so the cache is reused. A generic
  `single_event_prior(pop, cosmo::AbstractCosmology, Λ; z_grid)` adapter is provided
  for callers that only hold a bare cosmology.

Hyperparameter priors are caller-defined (e.g. `product_distribution(...)` in
notebooks or tests); they are not part of this package API.
"""
abstract type PopulationModel end

Base.broadcastable(m::PopulationModel) = Ref(m)

"""
    hyperparameters(pop::PopulationModel) -> NTuple{N,Symbol}

Ordered tuple of hyperparameter symbols owned by `pop`.  Implement on concrete
subtypes; do not overlap with the cosmology symbols.
"""
function hyperparameters end

"""
    single_event_prior(pop, cache::CosmologyCache, Λ) -> ProductNamedTupleDistribution

Per-event distribution over intrinsic parameters for the cosmology carried by
`cache` and hyperparameter state `Λ`.  Implement on concrete `PopulationModel`
subtypes, threading `cache` into `redshift_prior` so its cumulative cosmology
integral is reused by the importance-weight path rather than rebuilt.
"""
function single_event_prior end

"""
    single_event_prior(pop, cosmo::AbstractCosmology, Λ; z_grid) -> ProductNamedTupleDistribution

Generic adapter for callers that hold a bare cosmology (the oracle and
fiducial-reconstruction paths). Builds a [`CosmologyCache`](@ref) on `z_grid`
(default [`DEFAULT_Z_GRID`](@ref)) and dispatches to the population's cache method.
The hot path builds the cache once and calls the cache method directly.
"""
function single_event_prior(
        pop::PopulationModel,
        cosmo::AbstractCosmology,
        Λ::NamedTuple;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)
    return single_event_prior(pop, CosmologyCache(cosmo, z_grid), Λ)
end

"""
    full_hyperparameters(C, pop) -> NTuple{N,Symbol}

Concatenation of cosmology and population hyperparameter symbols, in the order
used for the flat HMC/Turing parameter vector.
"""
function full_hyperparameters(::Type{C}, pop::PopulationModel) where {C <:
                                                                      AbstractCosmology}
    return (hyperparameters(C)..., hyperparameters(pop)...)
end

"""
    validate_hyperparameters(order, Λ; context) -> nothing

Assert that `keys(Λ) == order` exactly (same symbols, same order).
"""
function validate_hyperparameters(
        order::Tuple{Vararg{Symbol}},
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    keys(Λ) == order || throw(
        ArgumentError("$(context) must match order $(order), got $(keys(Λ))"),
    )
    return nothing
end

"""
    canonical_hyperparameters(order, Λ; context, eltype) -> NamedTuple

Re-key `Λ` into the order given by `order`, converting values to `eltype`.
`Λ` may have its keys in any order as long as the set matches `order` exactly.
Pass `eltype = nothing` to preserve original value types.
"""
function canonical_hyperparameters(
        order::Tuple{Vararg{Symbol}},
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters",
        eltype = Float64
)
    Set(keys(Λ)) == Set(order) || throw(
        ArgumentError(
        "$(context) must exactly match $(order), got $(keys(Λ))"),
    )
    eltype === nothing && return (; (k => Λ[k] for k in order)...)
    return (; (k => eltype(Λ[k]) for k in order)...)
end
