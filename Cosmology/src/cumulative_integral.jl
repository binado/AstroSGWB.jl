using DataInterpolations: LinearInterpolation

"""
    CumulativeIntegral1D(x, f)

Linear interpolant of a scalar function `f` on strictly increasing nodes `x`,
plus a cumulative antiderivative at each node: prefix sum of trapezoids between
neighbors (exact for the `LinearInterpolation` antiderivative at grid points).

Query entry points:
- [`interpolate`](@ref) evaluates `f` via linear interpolation on `[x[1], x[end]]`
  (out-of-domain queries throw `BoundsError`).
- [`cdf`](@ref) evaluates the antiderivative at an arbitrary `x0`, clamping at the
  grid boundaries. The integral is exact under the linear interpolant (analytic
  trapezoidal rule), so no user-supplied `f` is needed at query time.
- [`normalizer`](@ref) returns the full integral `∫ f dx` over the grid
  (`last(cumulative)`).

# Fields
- `x`          : strictly increasing grid nodes (`Float64`)
- `y`          : `f` evaluated at each node
- `cumulative` : cumulative antiderivative at each node, `cumulative[1] = 0`
- `itp`        : cached `LinearInterpolation(y, x)` object
"""
struct CumulativeIntegral1D{
    TX <: AbstractVector{Float64},
    TY <: AbstractVector,
    TC <: AbstractVector,
    TI
}
    x::TX
    y::TY
    cumulative::TC
    itp::TI
end

"""
    CumulativeIntegral1D(x, f)

Build a [`CumulativeIntegral1D`](@ref) by evaluating `f` at each node of `x`,
building a `LinearInterpolation`, and computing nodal cumulative integrals as
the prefix sum of trapezoids ``(x_{i+1}-x_i)(y_i+y_{i+1})/2`` (O(n), identical
to the linear interpolant’s antiderivative on the nodes). `x` must be strictly
increasing with length ≥ 2.

Off-grid [`cdf`](@ref) queries use a direct analytic trapezoid lookup on the
cached nodal values.
"""
function _cumulative_at_nodes_trapezoid(x::AbstractVector{Float64}, y::AbstractVector)
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    cumulative = similar(y)
    @inbounds cumulative[1] = zero(y[1])
    acc = cumulative[1]
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
        cumulative[i + 1] = acc
    end
    return cumulative
end

@inline function _linear_cell_integral(cumulative_at_left, y_lo, y_hi, dx, t)
    return cumulative_at_left + dx * (y_lo * t + 0.5 * (y_hi - y_lo) * t^2)
end

function _cumulative_integral_from_values(
        x::AbstractVector{<:Real},
        y::AbstractVector
)
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    itp = LinearInterpolation(y, x_float)
    cumulative = _cumulative_at_nodes_trapezoid(x_float, y)
    return CumulativeIntegral1D(x_float, y, cumulative, itp)
end

function CumulativeIntegral1D(x::AbstractVector{<:Real}, f)
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    y = map(f, x_float)
    return _cumulative_integral_from_values(x_float, y)
end

"""
    CumulativeIntegral1D(x, y::AbstractVector{<:Real})

Build a [`CumulativeIntegral1D`](@ref) directly from precomputed nodal values `y`
(rather than evaluating a function). `x` must be strictly increasing with length ≥ 2
and `length(y) == length(x)`. `y` may carry `ForwardDiff.Dual` values so the cumulative
integral differentiates through whatever produced `y`.
"""
function CumulativeIntegral1D(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    return _cumulative_integral_from_values(x_float, y)
end

"""
    interpolate(c::CumulativeIntegral1D, x0) -> Real

Linear interpolation of `c.y` at `x0`. Only defined for
`x0 ∈ [c.x[1], c.x[end]]`; otherwise throws `BoundsError`.
"""
interpolate(c::CumulativeIntegral1D, x0::Real) = c.itp(x0)

"""
    cdf(c::CumulativeIntegral1D, x0) -> Real

Evaluate the antiderivative of the linear interpolant from `x[1]` to `x0` using
the analytic (piecewise-trapezoidal) integral. Clamps to `0` for `x0 <= x[1]`
and to `last(cumulative)` for `x0 >= x[end]`.
"""
function cdf(c::CumulativeIntegral1D, x0::Real)
    x_lo = @inbounds c.x[1]
    if x0 <= x_lo
        return zero(eltype(c.cumulative))
    end
    x_hi = @inbounds c.x[end]
    if x0 >= x_hi
        return @inbounds c.cumulative[end]
    end
    idx = searchsortedlast(c.x, x0)
    @inbounds begin
        dx = c.x[idx + 1] - c.x[idx]
        t = (x0 - c.x[idx]) / dx
        y_lo = c.y[idx]
        y_hi = c.y[idx + 1]
        return _linear_cell_integral(c.cumulative[idx], y_lo, y_hi, dx, t)
    end
end

"""
    normalizer(c::CumulativeIntegral1D) -> Real

Total integral of `f` over the grid, `last(c.cumulative)`.
"""
normalizer(c::CumulativeIntegral1D) = @inbounds c.cumulative[end]

"""
    GridQuery

Precomputed query plan for a fixed set of points located on a fixed grid. `bin_idx[i]`
is the lower grid-cell index for query point `i`; `t[i]` is the within-cell fraction.

Built once for a set of points and reused across many [`CumulativeIntegral1D`](@ref)s that
share the same grid but carry different (e.g. parameter-dependent) nodal values, so the
per-point grid search is hoisted out of the hot path. Query a [`CumulativeIntegral1D`](@ref)
with `interpolate(c, q, i)` (value) and `cdf(c, q, i)` (cumulative integral), the batched
counterparts to the scalar `interpolate(c, x0)` / `cdf(c, x0)`.
"""
struct GridQuery
    bin_idx::Vector{Int}
    t::Vector{Float64}
end

"""
    GridQuery(points, x)

Precompute the lower cell index and within-cell fraction of each point in `points`
on grid `x` (strictly increasing, length ≥ 2). Throws if a point lies outside
`[x[1], x[end]]`.
"""
function GridQuery(points::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    n_grid = length(x)
    n_grid >= 2 || throw(ArgumentError("grid must contain at least two points"))
    n = length(points)
    bin_idx = Vector{Int}(undef, n)
    t = Vector{Float64}(undef, n)
    x_min = @inbounds x[1]
    x_max = @inbounds x[end]
    @inbounds for i in 1:n
        z = points[i]
        (x_min <= z <= x_max) || throw(
            ArgumentError("query point $(z) lies outside grid support [$x_min, $x_max]"),
        )
        idx = if z == x_max
            n_grid - 1
        else
            searchsortedlast(x, z)
        end
        idx = max(1, min(idx, n_grid - 1))
        dz = x[idx + 1] - x[idx]
        bin_idx[i] = idx
        t[i] = Float64((z - x[idx]) / dz)
    end
    return GridQuery(bin_idx, t)
end

"""
    interpolate(c::CumulativeIntegral1D, q::GridQuery, i) -> Real

Linear interpolation of `c.y` at the `i`-th point of `q`, reusing the precomputed cell
location. Batched, search-free counterpart to [`interpolate`](@ref)`(c, x0)`.
"""
@inline function interpolate(c::CumulativeIntegral1D, q::GridQuery, i::Integer)
    @inbounds begin
        idx = q.bin_idx[i]
        ti = q.t[i]
        y = c.y
        return y[idx] + ti * (y[idx + 1] - y[idx])
    end
end

"""
    cdf(c::CumulativeIntegral1D, q::GridQuery, i) -> Real

Antiderivative of the linear interpolant from `c.x[1]` to the `i`-th point of `q`, reusing
the precomputed cell location. Batched, search-free counterpart to [`cdf`](@ref)`(c, x0)`.
"""
@inline function cdf(c::CumulativeIntegral1D, q::GridQuery, i::Integer)
    @inbounds begin
        idx = q.bin_idx[i]
        ti = q.t[i]
        dx = c.x[idx + 1] - c.x[idx]
        y_lo = c.y[idx]
        y_hi = c.y[idx + 1]
        return _linear_cell_integral(c.cumulative[idx], y_lo, y_hi, dx, ti)
    end
end
