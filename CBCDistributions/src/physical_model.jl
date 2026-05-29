using Distributions
using Distributions: ProductNamedTupleDistribution

"""
    PopulationModel

Abstract supertype for caller-defined population models.  Concrete subtypes
must implement the three-method contract:

- `hyperparameters(pop) -> NTuple{N, Symbol}` — ordered population parameter names.
- `hyperprior(pop) -> ProductNamedTupleDistribution` — prior over those parameters.
- `single_event_prior(pop, cosmo, Λ) -> ProductNamedTupleDistribution` — per-event
  distribution conditioned on cosmology `cosmo` and hyperparameters `Λ`.
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
    hyperprior(pop::PopulationModel) -> ProductNamedTupleDistribution
    hyperprior(::Type{C}) -> ProductNamedTupleDistribution

Prior distribution over hyperparameters.  The population variant covers
`hyperparameters(pop)`; the cosmology-type variant covers `hyperparameters(C)`.
"""
function hyperprior end

"""
    single_event_prior(pop, cosmo, Λ) -> ProductNamedTupleDistribution

Per-event distribution over intrinsic parameters for a given cosmology and
hyperparameter state `Λ`.  Implement on concrete `PopulationModel` subtypes.
"""
function single_event_prior end

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
    full_hyperprior(C, pop) -> ProductNamedTupleDistribution

Combined prior over all hyperparameters: cosmology first, then population.
"""
function full_hyperprior(::Type{C}, pop::PopulationModel) where {C <: AbstractCosmology}
    return product_distribution(merge(hyperprior(C).dists, hyperprior(pop).dists))
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

# ---------------------------------------------------------------------------
# Batched log-pdf helpers (consumed by likelihood and importance-sampling paths)
# ---------------------------------------------------------------------------

function _component_batch_length(d::Distribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _batched_output_eltype(dists)
    isempty(dists) && return Float64
    return promote_type(map(eltype, values(dists))...)
end

"""
    batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple) -> Vector

Per-sample log-density of `d` evaluated against a struct-of-arrays `samples`.
Each field of `d.dists` is matched to the same field in `samples`.
"""
function batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple)
    first_key = first(keys(d.dists))
    n = _component_batch_length(d.dists[first_key], samples, first_key)
    T = _batched_output_eltype(d.dists)
    out = zeros(T, n)
    for key in keys(d.dists)
        n_key = _component_batch_length(d.dists[key], samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        _add_component_logpdf!(out, d.dists[key], samples[key])
    end
    return out
end

"""
    batched_logpdf(d::Distribution, samples) -> logpdf(d, samples)

Scalar fallback for non-batched distributions.
"""
function batched_logpdf(d::Distribution, samples)
    return logpdf(d, samples)
end
