using Distributions
using Random

export RedshiftInterpolatedDistribution

"""
    RedshiftInterpolatedDistribution(dN_dz::CumulativeIntegral1D)

`ContinuousUnivariateDistribution` over redshift backed by a detector-frame `dN/dz`
tabulated on a grid (see [`redshift_density`](@ref)). The cached `norm = normalizer(dN_dz)`
makes [`logpdf`](@ref) the normalized log-density; sampling uses inverse-CDF lookup on
the cumulative table (not performance-critical).
"""
struct RedshiftInterpolatedDistribution{C <: CumulativeIntegral1D, T} <:
       ContinuousUnivariateDistribution
    dN_dz::C
    norm::T
end

function RedshiftInterpolatedDistribution(dN_dz::CumulativeIntegral1D)
    return RedshiftInterpolatedDistribution(dN_dz, normalizer(dN_dz))
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.dN_dz.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.dN_dz.x)
Base.eltype(d::RedshiftInterpolatedDistribution) = redshift_logpdf_eltype(d.dN_dz)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

function Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real)
    insupport(d, value) || return -Inf
    tiny = floatmin(real(eltype(d.dN_dz.y)))
    return _normalized_log_density(interpolate(d.dN_dz, value), d.norm, tiny)
end

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * d.norm
    cumulative = d.dN_dz.cumulative
    x = d.dN_dz.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end
