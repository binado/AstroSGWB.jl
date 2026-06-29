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
