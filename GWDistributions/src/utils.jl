const JULIAN_YEAR_SEC = 365.25 * 24 * 3600.0

year_to_second(yr::Real) = Float64(yr) * JULIAN_YEAR_SEC
second_to_year(sec::Real) = Float64(sec) / JULIAN_YEAR_SEC

"""
    _cumtrapz(y, x) -> AbstractVector

Cumulative trapezoidal integral of `y` over nodes `x`, evaluated at each node:
`out[1] = 0` and `out[i+1] = out[i] + (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2`.

The rule is exact for a piecewise-*linear* integrand sampled on `x`, which is why it
backs [`Interpolated1DDistribution`](@ref): it integrates the linear interpolant itself,
not an approximation of it.
"""
function _cumtrapz(y::AbstractVector, x::AbstractVector{<:Real})
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    n >= 1 || throw(ArgumentError("_cumtrapz requires at least one grid point"))
    cumulative = similar(y)
    @inbounds cumulative[1] = zero(y[1])
    acc = @inbounds cumulative[1]
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
        cumulative[i + 1] = acc
    end
    return cumulative
end

"""
    _trapz(y, x) -> Real

Integral of `y` over nodes `x` by the composite trapezoid rule, using the same
accumulation order as [`_cumtrapz`](@ref), so `_trapz(y, x) === last(_cumtrapz(y, x))`.
"""
function _trapz(y::AbstractVector, x::AbstractVector{<:Real})
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    n >= 1 || throw(ArgumentError("_trapz requires at least one grid point"))
    acc = zero(@inbounds y[1])
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
    end
    return acc
end

@inline function _planck_unit_exponent(t::Real)
    return inv(t) - inv(one(t) - t)
end

@inline function _planck_unit_taper(t::Real)
    T = typeof(t)
    t <= 0 && return zero(T)
    t >= 1 && return one(T)
    a = _planck_unit_exponent(t)
    if a > 0
        ea = exp(-a)
        return ea / (one(ea) + ea)
    end
    return inv(one(a) + exp(a))
end

@inline function _log_planck_unit_taper(t::Real)
    T = typeof(t)
    t <= 0 && return T(-Inf)
    t >= 1 && return zero(T)
    a = _planck_unit_exponent(t)
    if a > 0
        return -a - log1p(exp(-a))
    end
    return -log1p(exp(a))
end
