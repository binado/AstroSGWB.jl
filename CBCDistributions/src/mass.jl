using Distributions
using QuadGK
using Random

export DefaultBBHPrimaryMass, DefaultBBHMassPair, planck_taper

const DEFAULT_BBH_M_HIGH = 300.0
const _SQRT_EPS_FLOAT64 = sqrt(eps(Float64))
const _LOG_FLOATMAX_FLOAT64 = log(floatmax(Float64))
const _LOG_FLOATMIN_FLOAT64 = log(floatmin(Float64))

@inline function _logaddexp(a::Real, b::Real)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

@inline function _logsumexp3(a::Real, b::Real, c::Real)
    return _logaddexp(_logaddexp(a, b), c)
end

@inline function _log_weighted(weight::Real, log_value::Real)
    weight > 0 || return -Inf
    return log(weight) + log_value
end

@inline function _power_integral(low::Real, high::Real, exponent::Real)
    high > low || return zero(promote_type(typeof(low), typeof(high), typeof(exponent)))
    T = promote_type(typeof(low), typeof(high), typeof(exponent))
    a = exponent + one(T)
    if abs(a) <= _SQRT_EPS_FLOAT64
        return log(high / low)
    end
    return (high^a - low^a) / a
end

@inline function _broken_power_integral(
        α::Real,
        low::Real,
        high::Real,
        m_break::Real
)
    high > low ||
        return zero(promote_type(typeof(α), typeof(low), typeof(high), typeof(m_break)))
    T = promote_type(typeof(α), typeof(low), typeof(high), typeof(m_break))
    a = one(T) - α
    if abs(a) <= _SQRT_EPS_FLOAT64
        return m_break * log(high / low)
    end
    return m_break * ((high / m_break)^a - (low / m_break)^a) / a
end

function _broken_power_normalizer(
        α1::Real,
        α2::Real,
        m_break::Real,
        m_low::Real,
        m_high::Real
)
    return _broken_power_integral(α1, m_low, m_break, m_break) +
           _broken_power_integral(α2, m_break, m_high, m_break)
end

@inline function _log_broken_power_pdf(
        m::Real,
        α1::Real,
        α2::Real,
        m_break::Real,
        m_low::Real,
        m_high::Real
)
    (m_low <= m < m_high) || return -Inf
    norm = _broken_power_normalizer(α1, α2, m_break, m_low, m_high)
    norm > 0 || return -Inf
    α = m < m_break ? α1 : α2
    return -α * log(m / m_break) - log(norm)
end

@inline function _left_truncated_normal_logpdf(m::Real, μ::Real, σ::Real, low::Real)
    m >= low || return -Inf
    normal = Normal(μ, σ)
    return logpdf(normal, m) - logccdf(normal, low)
end

"""
    planck_taper(m, low, δ)

Planck taper used by the DEFAULT BBH mass model. It is zero below `low`, smoothly
turns on over `(low, low + δ)`, and is one above the taper interval. `δ == 0`
is treated as a hard step at `low`.
"""
function planck_taper(m::Real, low::Real, δ::Real)
    δ >= 0 || throw(ArgumentError("δ must be non-negative"))
    log_s = _log_planck_taper(m, low, δ)
    log_s == -Inf && return zero(promote_type(typeof(m), typeof(low), typeof(δ)))
    return exp(log_s)
end

@inline function _log_planck_taper(m::Real, low::Real, δ::Real)
    δ >= 0 || throw(ArgumentError("δ must be non-negative"))
    T = promote_type(typeof(m), typeof(low), typeof(δ))
    m < low && return -Inf
    δ == 0 && return zero(T)
    m >= low + δ && return zero(T)

    m′ = m - low
    m′ <= 0 && return -Inf
    a = δ / m′ + δ / (m′ - δ)
    a > _LOG_FLOATMAX_FLOAT64 && return -a
    a < _LOG_FLOATMIN_FLOAT64 && return -exp(a)
    return -log1p(exp(a))
end

struct DefaultBBHPrimaryMass{T <: Real, N <: Real} <: ContinuousUnivariateDistribution
    α1::T
    α2::T
    m_break::T
    μ1::T
    σ1::T
    μ2::T
    σ2::T
    m1_low::T
    δm1::T
    λ0::T
    λ1::T
    λ2::T
    m_high::T
    log_norm::N
end

function DefaultBBHPrimaryMass(;
        α1::Real,
        α2::Real,
        m_break::Real,
        μ1::Real,
        σ1::Real,
        μ2::Real,
        σ2::Real,
        m1_low::Real,
        δm1::Real,
        λ0::Real,
        λ1::Real,
        m_high::Real = DEFAULT_BBH_M_HIGH
)
    T = promote_type(
        Float64,
        typeof(α1), typeof(α2), typeof(m_break), typeof(μ1), typeof(σ1),
        typeof(μ2), typeof(σ2), typeof(m1_low), typeof(δm1), typeof(λ0),
        typeof(λ1), typeof(m_high)
    )
    λ2 = one(T) - T(λ0) - T(λ1)
    primary = _unchecked_primary(
        T(α1), T(α2), T(m_break), T(μ1), T(σ1), T(μ2), T(σ2), T(m1_low),
        T(δm1), T(λ0), T(λ1), λ2, T(m_high), zero(T))
    _validate_primary(primary)
    log_norm = log(_primary_normalizer(primary))
    return _unchecked_primary(
        primary.α1, primary.α2, primary.m_break, primary.μ1, primary.σ1, primary.μ2,
        primary.σ2, primary.m1_low, primary.δm1, primary.λ0, primary.λ1, primary.λ2,
        primary.m_high, log_norm)
end

function _unchecked_primary(
        α1::T,
        α2::T,
        m_break::T,
        μ1::T,
        σ1::T,
        μ2::T,
        σ2::T,
        m1_low::T,
        δm1::T,
        λ0::T,
        λ1::T,
        λ2::T,
        m_high::T,
        log_norm::N
) where {T <: Real, N <: Real}
    return DefaultBBHPrimaryMass{T, N}(
        α1, α2, m_break, μ1, σ1, μ2, σ2, m1_low, δm1, λ0, λ1, λ2,
        m_high, log_norm)
end

function _validate_primary(d::DefaultBBHPrimaryMass)
    0 < d.m1_low < d.m_break < d.m_high ||
        throw(ArgumentError("mass bounds must satisfy 0 < m1_low < m_break < m_high"))
    d.σ1 > 0 || throw(ArgumentError("σ1 must be positive"))
    d.σ2 > 0 || throw(ArgumentError("σ2 must be positive"))
    d.δm1 >= 0 || throw(ArgumentError("δm1 must be non-negative"))
    d.λ0 >= 0 || throw(ArgumentError("λ0 must be non-negative"))
    d.λ1 >= 0 || throw(ArgumentError("λ1 must be non-negative"))
    d.λ2 >= 0 || throw(ArgumentError("1 - λ0 - λ1 must be non-negative"))
    d.λ0 + d.λ1 + d.λ2 > 0 ||
        throw(ArgumentError("at least one mixture weight must be positive"))
    return nothing
end

Base.minimum(d::DefaultBBHPrimaryMass) = d.m1_low
Base.maximum(d::DefaultBBHPrimaryMass) = d.m_high
Base.eltype(::Type{<:DefaultBBHPrimaryMass{T}}) where {T} = T
Base.eltype(d::DefaultBBHPrimaryMass) = typeof(d.m1_low)

function Distributions.insupport(d::DefaultBBHPrimaryMass, value::Real)
    return d.m1_low <= value < d.m_high
end

@inline function _primary_log_untapered_mixture(d::DefaultBBHPrimaryMass, m::Real)
    bp = _log_broken_power_pdf(m, d.α1, d.α2, d.m_break, d.m1_low, d.m_high)
    g1 = m < d.m_high ? _left_truncated_normal_logpdf(m, d.μ1, d.σ1, d.m1_low) : -Inf
    g2 = m < d.m_high ? _left_truncated_normal_logpdf(m, d.μ2, d.σ2, d.m1_low) : -Inf
    return _logsumexp3(
        _log_weighted(d.λ0, bp),
        _log_weighted(d.λ1, g1),
        _log_weighted(d.λ2, g2)
    )
end

@inline function _primary_log_unnormalized(d::DefaultBBHPrimaryMass, m::Real)
    insupport(d, m) || return -Inf
    log_s = _log_planck_taper(m, d.m1_low, d.δm1)
    log_s == -Inf && return -Inf
    return _primary_log_untapered_mixture(d, m) + log_s
end

function _primary_normalizer(d::DefaultBBHPrimaryMass)
    f = m -> exp(_primary_log_unnormalized(d, m))
    taper_high = min(d.m1_low + d.δm1, d.m_high)
    z = zero(promote_type(typeof(d.m1_low), typeof(d.log_norm)))
    if d.δm1 > 0 && taper_high > d.m1_low
        z += first(quadgk(f, d.m1_low, taper_high))
    end
    if taper_high < d.m_high
        z += first(quadgk(f, taper_high, d.m_high))
    end
    return z
end

function Distributions.logpdf(d::DefaultBBHPrimaryMass, value::Real)
    logp = _primary_log_unnormalized(d, value)
    logp == -Inf && return -Inf
    return logp - d.log_norm
end

Distributions.pdf(d::DefaultBBHPrimaryMass, value::Real) = exp(logpdf(d, value))

struct DefaultBBHMassPair{P <: DefaultBBHPrimaryMass, T <: Real} <:
       ContinuousMultivariateDistribution
    primary::P
    βq::T
    m2_low::T
    δm2::T
end

function DefaultBBHMassPair(;
        α1::Real,
        α2::Real,
        m_break::Real,
        μ1::Real,
        σ1::Real,
        μ2::Real,
        σ2::Real,
        m1_low::Real,
        δm1::Real,
        λ0::Real,
        λ1::Real,
        βq::Real,
        m2_low::Real,
        δm2::Real,
        m_high::Real = DEFAULT_BBH_M_HIGH
)
    primary = DefaultBBHPrimaryMass(;
        α1, α2, m_break, μ1, σ1, μ2, σ2, m1_low, δm1, λ0, λ1, m_high)
    T = promote_type(
        Float64, typeof(primary.m1_low), typeof(βq), typeof(m2_low), typeof(δm2))
    d = DefaultBBHMassPair(primary, T(βq), T(m2_low), T(δm2))
    _validate_mass_pair(d)
    return d
end

function _validate_mass_pair(d::DefaultBBHMassPair)
    d.m2_low > 0 || throw(ArgumentError("m2_low must be positive"))
    d.m2_low <= d.primary.m1_low ||
        throw(ArgumentError("m2_low must be less than or equal to m1_low"))
    d.δm2 >= 0 || throw(ArgumentError("δm2 must be non-negative"))
    return nothing
end

Base.length(::DefaultBBHMassPair) = 2
Base.size(::DefaultBBHMassPair) = (2,)
Base.eltype(::Type{<:DefaultBBHMassPair{P, T}}) where {P, T} = promote_type(eltype(P), T)
Base.eltype(d::DefaultBBHMassPair) = promote_type(eltype(d.primary), typeof(d.βq))

function Distributions.insupport(d::DefaultBBHMassPair, value::Tuple{<:Real, <:Real})
    m1, m2 = value
    return insupport(d.primary, m1) && d.m2_low <= m2 <= m1
end

function Distributions.insupport(d::DefaultBBHMassPair, value::AbstractVector{<:Real})
    length(value) == 2 || return false
    return insupport(d, (value[1], value[2]))
end

function _q_power_integral(d::DefaultBBHMassPair, q_low::Real, q_high::Real)
    return _power_integral(q_low, q_high, d.βq)
end

function _q_normalizer(d::DefaultBBHMassPair, m1::Real)
    q_low = d.m2_low / m1
    q_low < 1 || return zero(promote_type(typeof(q_low), typeof(d.βq)))
    d.δm2 == 0 && return _q_power_integral(d, q_low, one(q_low))

    q_taper_high = min((d.m2_low + d.δm2) / m1, one(q_low))
    z = zero(promote_type(typeof(q_low), typeof(d.βq)))
    if q_taper_high > q_low
        f = q -> q^d.βq * planck_taper(q * m1, d.m2_low, d.δm2)
        z += first(quadgk(f, q_low, q_taper_high))
    end
    if q_taper_high < 1
        z += _q_power_integral(d, q_taper_high, one(q_low))
    end
    return z
end

function Distributions.logpdf(d::DefaultBBHMassPair, value::Tuple{<:Real, <:Real})
    insupport(d, value) || return -Inf
    m1, m2 = value
    q = m2 / m1
    log_s2 = _log_planck_taper(m2, d.m2_low, d.δm2)
    log_s2 == -Inf && return -Inf
    zq = _q_normalizer(d, m1)
    zq > 0 || return -Inf
    return logpdf(d.primary, m1) + d.βq * log(q) + log_s2 - log(zq) - log(m1)
end

function Distributions._logpdf(d::DefaultBBHMassPair, x::AbstractVector{<:Real})
    return logpdf(d, (x[1], x[2]))
end

function Distributions.pdf(d::DefaultBBHMassPair, value::Tuple{<:Real, <:Real})
    exp(logpdf(d, value))
end
function Distributions.pdf(d::DefaultBBHMassPair, value::AbstractVector{<:Real})
    exp(logpdf(d, value))
end

function _rand_broken_power(rng::AbstractRNG, d::DefaultBBHPrimaryMass)
    z1 = _broken_power_integral(d.α1, d.m1_low, d.m_break, d.m_break)
    z2 = _broken_power_integral(d.α2, d.m_break, d.m_high, d.m_break)
    target = rand(rng) * (z1 + z2)
    if target <= z1
        return _rand_scaled_power(rng, d.m1_low, d.m_break, -d.α1)
    end
    return _rand_scaled_power(rng, d.m_break, d.m_high, -d.α2)
end

function _rand_scaled_power(rng::AbstractRNG, low::Real, high::Real, exponent::Real)
    u = rand(rng)
    a = exponent + 1
    if abs(a) <= _SQRT_EPS_FLOAT64
        return low * exp(u * log(high / low))
    end
    return (low^a + u * (high^a - low^a))^(1 / a)
end

function _rand_primary_proposal(rng::AbstractRNG, d::DefaultBBHPrimaryMass)
    u = rand(rng)
    if u < d.λ0
        return _rand_broken_power(rng, d)
    elseif u < d.λ0 + d.λ1
        return rand(rng, truncated(Normal(d.μ1, d.σ1), d.m1_low, Inf))
    end
    return rand(rng, truncated(Normal(d.μ2, d.σ2), d.m1_low, Inf))
end

function Random.rand(rng::AbstractRNG, d::DefaultBBHPrimaryMass)
    while true
        m1 = _rand_primary_proposal(rng, d)
        m1 < d.m_high || continue
        rand(rng) <= planck_taper(m1, d.m1_low, d.δm1) && return m1
    end
end

function _rand_q_power(rng::AbstractRNG, βq::Real, q_low::Real)
    q_low < 1 || return one(q_low)
    u = rand(rng)
    a = βq + 1
    if abs(a) <= _SQRT_EPS_FLOAT64
        return q_low * exp(u * log(1 / q_low))
    end
    return (q_low^a + u * (1 - q_low^a))^(1 / a)
end

function _rand_q(rng::AbstractRNG, d::DefaultBBHMassPair, m1::Real)
    q_low = d.m2_low / m1
    while true
        q = _rand_q_power(rng, d.βq, q_low)
        rand(rng) <= planck_taper(q * m1, d.m2_low, d.δm2) && return q
    end
end

function Random.rand(rng::AbstractRNG, d::DefaultBBHMassPair)
    m1 = rand(rng, d.primary)
    q = _rand_q(rng, d, m1)
    return [m1, q * m1]
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::DefaultBBHMassPair,
        x::AbstractVector{<:Real}
)
    length(x) == 2 || throw(ArgumentError("DEFAULT BBH mass pair expects length-2 output"))
    sample = rand(rng, d)
    x[1] = sample[1]
    x[2] = sample[2]
    return x
end

function _add_component_logpdf!(
        out::AbstractVector,
        d::DefaultBBHMassPair,
        field::AbstractMatrix
)
    size(field, 1) == 2 ||
        throw(ArgumentError("DEFAULT BBH source-mass batch must have two rows"))
    @inbounds for i in eachindex(out)
        out[i] += logpdf(d, (field[1, i], field[2, i]))
    end
    return out
end
