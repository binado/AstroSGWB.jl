using Distributions

"""
    IntrinsicPrior(dists::NamedTuple)

Batched intrinsic-prior distribution over a struct-of-arrays sample bundle.
`dists` maps sample field names to component distributions. Extra fields in
sample bundles are ignored.
"""
struct IntrinsicPrior{Names, Dists}
    dists::NamedTuple{Names, Dists}

    function IntrinsicPrior(dists::NamedTuple{Names, Dists}) where {Names, Dists}
        isempty(Names) &&
            throw(ArgumentError("intrinsic prior must contain at least one component"))
        return new{Names, Dists}(dists)
    end
end

"""
    intrinsic_prior(::FullBNS; kwargs...) -> IntrinsicPrior

Build the intrinsic-parameter prior for full-BNS proposal samples. The returned
prior evaluates batches with fields `mass`, `χ₁`, `χ₂`, `Λ₁`, and `Λ₂`;
proposal fields not present in the prior, such as `redshift`, are ignored.
"""
function intrinsic_prior(
        ::FullBNS;
        mass_low::Real = BNS_MASS_LOW,
        mass_high::Real = BNS_MASS_HIGH,
        spin_a_max::Real = BNS_SPIN_A_MAX,
        lambda_high::Real = BNS_LAMBDA_HIGH
)
    lambda_dist = Uniform(0.0, Float64(lambda_high))
    spin_dist = AlignedSpinChiSimple(; a_max = spin_a_max)
    return IntrinsicPrior((
        mass = OrderedUniformSourceMassPair(; low = mass_low, high = mass_high),
        χ₁ = spin_dist,
        χ₂ = spin_dist,
        Λ₁ = lambda_dist,
        Λ₂ = lambda_dist
    ))
end

"""
    validate_batch(prior::IntrinsicPrior, samples::NamedTuple) -> Int

Validate that `samples` contains every field required by `prior` and that all
prior fields describe the same number of samples. Extra sample fields are
ignored. Returns the batch size.
"""
function validate_batch(prior::IntrinsicPrior, samples::NamedTuple)::Int
    first_key = first(keys(prior.dists))
    haskey(samples, first_key) ||
        throw(ArgumentError("samples are missing intrinsic prior field $(repr(first_key))"))
    n = _component_batch_length(prior.dists[first_key], samples[first_key], first_key)
    for key in Iterators.drop(keys(prior.dists), 1)
        haskey(samples, key) ||
            throw(ArgumentError("samples are missing intrinsic prior field $(repr(key))"))
        n_key = _component_batch_length(prior.dists[key], samples[key], key)
        n_key == n ||
            throw(ArgumentError("intrinsic prior sample fields must have matching lengths"))
    end
    return n
end

function _component_batch_length(d::UnivariateDistribution, field::AbstractVector, key)
    return length(field)
end

function _component_batch_length(d::MultivariateDistribution, field::AbstractMatrix, key)
    expected = _event_size(d)
    size(field, 1) == expected ||
        throw(
            ArgumentError(
            "intrinsic prior field $(repr(key)) must have $expected rows, got $(size(field, 1))",
        ),
        )
    return size(field, 2)
end

function _component_batch_length(d, field, key)
    throw(
        ArgumentError(
        "unsupported batch layout for intrinsic prior field $(repr(key)) and distribution $(typeof(d))",
    ),
    )
end

_event_size(d::MultivariateDistribution) = length(d)

function _prior_output_eltype(prior::IntrinsicPrior)
    return promote_type(map(eltype, values(prior.dists))...)
end

function Distributions.logpdf(prior::IntrinsicPrior, samples::NamedTuple)
    n = validate_batch(prior, samples)
    T = _prior_output_eltype(prior)
    out = zeros(T, n)
    return logpdf!(out, prior, samples)
end

"""
    logpdf!(out, prior::IntrinsicPrior, samples::NamedTuple) -> out

Evaluate per-sample intrinsic log-prior terms in place.
"""
function logpdf!(out::AbstractVector, prior::IntrinsicPrior, samples::NamedTuple)
    n = validate_batch(prior, samples)
    length(out) == n ||
        throw(ArgumentError("output length must match the number of samples"))
    fill!(out, zero(eltype(out)))
    for key in keys(prior.dists)
        _add_component_logpdf!(out, prior.dists[key], samples[key])
    end
    return out
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::UnivariateDistribution,
        field::AbstractVector
)
    @inbounds for i in eachindex(out, field)
        out[i] += logpdf(d, field[i])
    end
    return out
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::MultivariateDistribution,
        field::AbstractMatrix
)
    values = logpdf(d, field)
    length(values) == length(out) ||
        throw(ArgumentError("component logpdf length must match output length"))
    @inbounds for i in eachindex(out, values)
        out[i] += values[i]
    end
    return out
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::OrderedUniformSourceMassPair,
        field::AbstractMatrix
)
    size(field, 1) == 2 ||
        throw(ArgumentError("ordered source-mass batch must have two rows"))
    @inbounds for i in eachindex(out)
        out[i] += _ordered_mass_logpdf(d, field[1, i], field[2, i])
    end
    return out
end

function _ordered_mass_logpdf(d::OrderedUniformSourceMassPair, m1::Real, m2::Real)
    m1 >= m2 && m2 >= d.low && m1 <= d.high || return -Inf
    T = typeof(d.low)
    return log(T(2)) - T(2) * log(d.high - d.low)
end
