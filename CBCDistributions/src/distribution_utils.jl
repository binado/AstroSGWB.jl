using Distributions
using Distributions: ProductNamedTupleDistribution

# Batched log-pdf helpers consumed by likelihood and importance-sampling paths.

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
        d::SourceMassPairDistribution,
        field
)
    values = sample_values(field)
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
    n = validate_samples(d, samples)
    T = _batched_output_eltype(d.dists)
    out = zeros(T, n)
    for key in keys(d.dists)
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
    n = validate_samples(d, samples)
    vals = map(ks) do key
        dk = d.dists[key]
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

Per-sample log-density ratio `log p_target - log p_proposal` between the target
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
    n = validate_samples(prior, samples)
    T = _batched_output_eltype(prior.dists)
    out = zeros(T, n)
    for key in ks
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
