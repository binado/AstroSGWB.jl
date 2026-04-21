using Distributions
using Random

const BNS_MASS_LOW = 1.1
const BNS_MASS_HIGH = 2.5
const BNS_LAMBDA_HIGH = 5000.0
const BNS_SPIN_A_MAX = 0.99

struct OrderedUniformSourceMassPair{T<:Real} <: ContinuousMultivariateDistribution
    low::T
    high::T
end

function OrderedUniformSourceMassPair(; low::Real=BNS_MASS_LOW, high::Real=BNS_MASS_HIGH)
    low < high || throw(ArgumentError("low must be smaller than high"))
    return OrderedUniformSourceMassPair(Float64(low), Float64(high))
end

Base.length(::OrderedUniformSourceMassPair) = 2
Base.size(::OrderedUniformSourceMassPair) = (2,)

function Distributions.insupport(
    d::OrderedUniformSourceMassPair,
    value::NTuple{2,<:Real},
)
    m1, m2 = value
    return m1 >= m2 && m2 >= d.low && m1 <= d.high
end

function Distributions.insupport(
    d::OrderedUniformSourceMassPair,
    value::AbstractVector{<:Real},
)
    length(value) == 2 || return false
    return insupport(d, (value[1], value[2]))
end

function Distributions.logpdf(
    d::OrderedUniformSourceMassPair,
    value::NTuple{2,<:Real},
)
    return insupport(d, value) ? log(2.0) - 2.0 * log(d.high - d.low) : -Inf
end

function Distributions.logpdf(
    d::OrderedUniformSourceMassPair,
    value::AbstractVector{<:Real},
)
    length(value) == 2 || throw(ArgumentError("ordered mass pair expects two coordinates"))
    return logpdf(d, (value[1], value[2]))
end

function Random.rand(rng::AbstractRNG, d::OrderedUniformSourceMassPair)
    span = d.high - d.low
    x = d.low + span * rand(rng)
    y = d.low + span * rand(rng)
    return x >= y ? [x, y] : [y, x]
end

struct AlignedSpinChiSimple{T<:Real} <: ContinuousUnivariateDistribution
    a_max::T
end

function AlignedSpinChiSimple(; a_max::Real=BNS_SPIN_A_MAX)
    a_max > 0 || throw(ArgumentError("a_max must be positive"))
    return AlignedSpinChiSimple(Float64(a_max))
end

Base.minimum(d::AlignedSpinChiSimple) = -d.a_max
Base.maximum(d::AlignedSpinChiSimple) = d.a_max

Distributions.insupport(d::AlignedSpinChiSimple, value::Real) = abs(value) <= d.a_max

function Distributions.logpdf(d::AlignedSpinChiSimple, value::Real)
    insupport(d, value) || return -Inf
    eps_value = eps(Float64)
    density = -log(max(abs(value), eps_value) / d.a_max) / (2.0 * d.a_max)
    return log(max(density, floatmin(Float64)))
end

function Random.rand(rng::AbstractRNG, d::AlignedSpinChiSimple)
    magnitude = d.a_max * rand(rng) * rand(rng)
    return rand(rng, Bool) ? magnitude : -magnitude
end

struct RedshiftInterpolatedDistribution{B<:RadialInterpolant} <: ContinuousUnivariateDistribution
    bundle::B
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.bundle.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.bundle.x)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real) =
    log_prob_from_bundle(value, d.bundle)

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * d.bundle.norm
    cumulative = d.bundle.cumulative
    x = d.bundle.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx+1]
    x0, x1 = x[idx], x[idx+1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end

"""
    FullBNSIntrinsicPrior{M,Z,C,L}

Intrinsic-parameter prior for the [`FullBNS`](@ref) proposal: an ordered source-frame
mass pair, a redshift distribution tied to a [`RadialInterpolant`](@ref), a shared
aligned-spin distribution applied to `chi_1` and `chi_2`, and a shared uniform
distribution applied to `lambda_1` and `lambda_2`.
"""
struct FullBNSIntrinsicPrior{M,Z,C,L}
    mass::M
    redshift::Z
    spin::C
    lambda::L
end

function build_uniform_priors(
    bounds::AbstractDict{<:AbstractString,<:Tuple{<:Real,<:Real}},
)
    return InferencePriors(
        Uniform(Float64(bounds["H0"][1]), Float64(bounds["H0"][2])),
        Uniform(Float64(bounds["Omega_m"][1]), Float64(bounds["Omega_m"][2])),
        Uniform(Float64(bounds["chi0"][1]), Float64(bounds["chi0"][2])),
        Uniform(Float64(bounds["chin"][1]), Float64(bounds["chin"][2])),
        Uniform(Float64(bounds["gamma"][1]), Float64(bounds["gamma"][2])),
        Uniform(Float64(bounds["kappa"][1]), Float64(bounds["kappa"][2])),
        Uniform(Float64(bounds["z_peak"][1]), Float64(bounds["z_peak"][2])),
    )
end

function logprior(h::HyperParameters, priors::InferencePriors)
    pop = h.population
    pop isa MadauDickinsonParameters || throw(
        ArgumentError("logprior with InferencePriors requires MadauDickinsonParameters"),
    )
    return (
        logpdf(priors.H0, h.cosmological.H0) +
        logpdf(priors.Omega_m, h.cosmological.Omega_m) +
        logpdf(priors.chi0, h.propagation.chi0) +
        logpdf(priors.chin, h.propagation.chin) +
        logpdf(priors.gamma, pop.gamma) +
        logpdf(priors.kappa, pop.kappa) +
        logpdf(priors.z_peak, pop.z_peak)
    )
end

"""
    intrinsic_prior(::FullBNS, bundle; kwargs...) -> FullBNSIntrinsicPrior

Build the intrinsic-parameter prior for the full-BNS proposal. `bundle` supplies the
redshift interpolant used by [`RedshiftInterpolatedDistribution`](@ref).
"""
function intrinsic_prior(
    ::FullBNS,
    bundle::RadialInterpolant;
    mass_low::Real=BNS_MASS_LOW,
    mass_high::Real=BNS_MASS_HIGH,
    spin_a_max::Real=BNS_SPIN_A_MAX,
    lambda_high::Real=BNS_LAMBDA_HIGH,
)
    return FullBNSIntrinsicPrior(
        OrderedUniformSourceMassPair(; low=mass_low, high=mass_high),
        RedshiftInterpolatedDistribution(bundle),
        AlignedSpinChiSimple(; a_max=spin_a_max),
        Uniform(0.0, Float64(lambda_high)),
    )
end

function _full_bns_logpdf_pointwise(
    prior::FullBNSIntrinsicPrior,
    samples::FullBNSSamples,
    i::Integer,
)
    return (
        logpdf(prior.mass, (samples.mass_1_source[i], samples.mass_2_source[i])) +
        logpdf(prior.redshift, samples.redshift[i]) +
        logpdf(prior.spin, samples.chi_1[i]) +
        logpdf(prior.spin, samples.chi_2[i]) +
        logpdf(prior.lambda, samples.lambda_1[i]) +
        logpdf(prior.lambda, samples.lambda_2[i])
    )
end

function _require_full_bns_matching_lengths(samples::FullBNSSamples)
    n = length(samples.redshift)
    (length(samples.mass_1_source) == n &&
     length(samples.mass_2_source) == n &&
     length(samples.chi_1) == n &&
     length(samples.chi_2) == n &&
     length(samples.lambda_1) == n &&
     length(samples.lambda_2) == n) || throw(
        ArgumentError("FullBNSSamples vectors must all have matching lengths"),
    )
    return n
end

"""
    intrinsic_log_prob_samples!(out, samples, prior) -> out

Write per-sample log-prior into `out`. `out` must match the sample count of `samples`.
"""
function intrinsic_log_prob_samples!(
    out::AbstractVector,
    samples::FullBNSSamples,
    prior::FullBNSIntrinsicPrior,
)
    n = _require_full_bns_matching_lengths(samples)
    length(out) == n || throw(
        ArgumentError("output and sample vectors must have matching lengths"),
    )
    @inbounds for i in 1:n
        out[i] = _full_bns_logpdf_pointwise(prior, samples, i)
    end
    return out
end

"""
    intrinsic_log_prob_samples(samples, prior) -> Vector

Per-sample log-prior as a freshly allocated vector.
"""
function intrinsic_log_prob_samples(
    samples::FullBNSSamples,
    prior::FullBNSIntrinsicPrior,
)
    n = _require_full_bns_matching_lengths(samples)
    n == 0 && return Float64[]
    first_val = _full_bns_logpdf_pointwise(prior, samples, 1)
    out = Vector{typeof(first_val)}(undef, n)
    @inbounds out[1] = first_val
    @inbounds for i in 2:n
        out[i] = _full_bns_logpdf_pointwise(prior, samples, i)
    end
    return out
end
