using Distributions
using Distributions: ProductNamedTupleDistribution

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

# ---------------------------------------------------------------------------
# Batched log-pdf helpers (consumed by likelihood and importance-sampling paths)
# ---------------------------------------------------------------------------

function _component_batch_length(d::Distribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _component_batch_length(d::UnivariateDistribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _component_batch_length(d::MultivariateDistribution, samples::NamedTuple, key)
    haskey(samples, key) ||
        throw(ArgumentError("samples are missing population prior field $(repr(key))"))
    return _component_batch_length(d, samples[key], key)
end

function _component_batch_length(d::UnivariateDistribution, field, key)
    values = sample_values(field)
    values isa AbstractVector || throw(
        ArgumentError(
        "population prior field $(repr(key)) must be a vector for univariate distribution $(typeof(d))",
    ),
    )
    return length(values)
end

function _component_batch_length(d::MultivariateDistribution, field, key)
    values = sample_values(field)
    values isa AbstractMatrix || throw(
        ArgumentError(
        "population prior field $(repr(key)) must be a matrix for multivariate distribution $(typeof(d))",
    ),
    )
    expected = length(d)
    size(values, 1) == expected ||
        throw(
            ArgumentError(
            "population prior field $(repr(key)) must have $expected rows, got $(size(values, 1))",
        ),
        )
    return size(values, 2)
end

function _component_batch_length(d, field, key)
    throw(
        ArgumentError(
        "unsupported batch layout for population prior field $(repr(key)) and distribution $(typeof(d))",
    ),
    )
end

function _batched_output_eltype(dists)
    isempty(dists) && return Float64
    return promote_type(map(eltype, values(dists))...)
end

"""
    add_logpdfvec!(out, d, field) -> out

Accumulate a component distribution's batched log-density into `out`.
Implementations own the per-sample loop and add `logpdf(d, sample_i)` to
`out[i]`. `field` may be a raw batched field or a [`SampleField`](@ref) carrying
metadata for specialized fast paths.
"""
function add_logpdfvec!(
        out::AbstractVector,
        d::UnivariateDistribution,
        field
)
    values = sample_values(field)
    @inbounds for i in eachindex(out, values)
        out[i] += logpdf(d, values[i])
    end
    return out
end

function add_logpdfvec!(
        out::AbstractVector,
        d::MultivariateDistribution,
        field
)
    field_values = sample_values(field)
    logpdf_values = logpdf(d, field_values)
    length(logpdf_values) == length(out) ||
        throw(ArgumentError("component logpdf length must match output length"))
    @inbounds for i in eachindex(out, logpdf_values)
        out[i] += logpdf_values[i]
    end
    return out
end

function add_logpdfvec!(
        out::AbstractVector,
        d::OrderedUniformSourceMassPair,
        field
)
    values = sample_values(field)
    size(values, 1) == 2 ||
        throw(ArgumentError("ordered source-mass batch must have two rows"))
    @inbounds for i in eachindex(out)
        out[i] += logpdf(d, (values[1, i], values[2, i]))
    end
    return out
end

function add_logpdfvec!(
        out::AbstractVector,
        d::DefaultBBHMassPair,
        field
)
    values = sample_values(field)
    size(values, 1) == 2 ||
        throw(ArgumentError("DEFAULT BBH source-mass batch must have two rows"))
    @inbounds for i in eachindex(out)
        out[i] += logpdf(d, (values[1, i], values[2, i]))
    end
    return out
end

"""
    batched_logpdf(d::ProductNamedTupleDistribution, samples::NamedTuple) -> Vector

Per-sample log-density of `d` evaluated against a struct-of-arrays `samples`.
Each field of `d.dists` is matched to the same field in `samples`.
Individual fields may be wrapped in [`SampleField`](@ref) to carry metadata for
specialized component methods.
"""
function batched_logpdf(
        d::ProductNamedTupleDistribution,
        samples::NamedTuple
)
    first_key = first(keys(d.dists))
    n = _component_batch_length(d.dists[first_key], samples, first_key)
    T = _batched_output_eltype(d.dists)
    out = zeros(T, n)
    for key in keys(d.dists)
        n_key = _component_batch_length(d.dists[key], samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        add_logpdfvec!(out, d.dists[key], samples[key])
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

"""
    component_logpdfs(d::ProductNamedTupleDistribution, samples) -> NamedTuple

Per-component batched log-densities of `d` against a struct-of-arrays `samples`: one
vector per field of `d.dists`, keyed like [`batched_logpdf`](@ref) (whose output is the
sum of these). Used to cache the fiducial proposal log-densities per component so
[`logprobdiff`](@ref) can subtract only the components that actually change.
"""
function component_logpdfs(
        d::ProductNamedTupleDistribution,
        samples::NamedTuple
)
    ks = keys(d.dists)
    first_key = first(ks)
    n = _component_batch_length(d.dists[first_key], samples, first_key)
    vals = map(ks) do key
        dk = d.dists[key]
        n_key = _component_batch_length(dk, samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        out = zeros(_batched_output_eltype((dk,)), n)
        add_logpdfvec!(out, dk, samples[key])
    end
    return NamedTuple{ks}(vals)
end

"""
    logprobdiff!(out, model, ::Val{key}, d_target, d_proposal, proposal_logprob, x)

Accumulate into `out` the per-sample log-density difference
`logpdf(d_target, xᵢ) − proposal_logprob[i]` for one component `key` of the
single-event prior. This is the per-component extension point of
[`logprobdiff`](@ref): overload it on a concrete `PopulationModel` (and `Val{key}`
or a distribution type) when the difference has a cheaper form than the generic
two-sided evaluation.

The default skips the component entirely when `d_target === d_proposal` and every
cached proposal log-density is finite: egal distributions have identical
log-densities on support, so their difference is exactly zero. Out-of-support
samples yield `-Inf` on both sides; skipping would assign a zero log-ratio and
mask invalid catalog points, so the fast path is taken only when
`all(isfinite, proposal_logprob)`. Components built with `Λ`-independent
constructors (isbits distributions such as fixed-bound `Uniform`s) hit this path
on every in-support evaluation; `Λ`-dependent components (e.g. an interpolated
redshift prior) never compare egal and are computed. Overloads that skip a
component bake in an exactness assumption owned by the population model; keep
them in sync with `single_event_prior`.
"""
function logprobdiff!(
        out::AbstractVector,
        model::PopulationModel,
        ::Val{key},
        d_target,
        d_proposal,
        proposal_logprob::AbstractVector{<:Real},
        x
) where {key}
    if d_target === d_proposal && all(isfinite, proposal_logprob)
        return out
    end
    length(proposal_logprob) == length(out) || throw(
        ArgumentError(
        "proposal logpdf length must match batch size for population prior field $(repr(key))",
    ),
    )
    add_logpdfvec!(out, d_target, x)
    @inbounds for i in eachindex(out, proposal_logprob)
        out[i] -= proposal_logprob[i]
    end
    return out
end

"""
    logprobdiff(model, prior, proposal, proposal_logprob::NamedTuple, samples) -> Vector

Per-sample log-density ratio `log p_target − log p_proposal` between the target
single-event `prior` (built at live hyperparameters `Λ`) and the fiducial `proposal`,
with the proposal's per-component log-densities supplied precomputed in
`proposal_logprob` (see [`component_logpdfs`](@ref)). Components are accumulated via
[`logprobdiff!`](@ref), so egal components (identical between target and proposal,
i.e. `Λ`-independent) are skipped exactly, and population models can overload the
per-component method for custom cancellations.
"""
function logprobdiff(
        model::PopulationModel,
        prior::ProductNamedTupleDistribution,
        proposal::ProductNamedTupleDistribution,
        proposal_logprob::NamedTuple,
        samples::NamedTuple
)
    ks = keys(prior.dists)
    keys(proposal.dists) == ks ||
        throw(ArgumentError("proposal prior fields must match target prior fields $(ks)"))
    keys(proposal_logprob) == ks ||
        throw(ArgumentError("proposal logpdf fields must match target prior fields $(ks)"))
    first_key = first(ks)
    n = _component_batch_length(prior.dists[first_key], samples, first_key)
    T = _batched_output_eltype(prior.dists)
    out = zeros(T, n)
    for key in ks
        n_key = _component_batch_length(prior.dists[key], samples, key)
        n_key == n ||
            throw(ArgumentError("population prior sample fields must have matching lengths"))
        logprobdiff!(
            out, model, Val(key), prior.dists[key], proposal.dists[key],
            proposal_logprob[key], samples[key])
    end
    return out
end

"""
    logprobdiff(model, prior, proposal, samples) -> Vector

Convenience wrapper that computes the proposal's per-component log-densities on the
fly via [`component_logpdfs`](@ref) and delegates to the cached form. Hot paths
should precompute `proposal_logprob` once (the proposal is fiducial and fixed) and
call the 5-argument method directly.
"""
function logprobdiff(
        model::PopulationModel,
        prior::ProductNamedTupleDistribution,
        proposal::ProductNamedTupleDistribution,
        samples::NamedTuple
)
    proposal_logprob = component_logpdfs(proposal, samples)
    return logprobdiff(model, prior, proposal, proposal_logprob, samples)
end
