using QuadGK

# Reference Gauss–Legendre nodes and weights on [-1, 1], computed once per rule.
# The grid path matches astrogwb's composite rule (`GAUSS_LEGENDRE_ORDER = 4`: 4
# nodes per grid interval reach near machine precision for the smooth integrand
# 1/E(z)). The scalar rule covers [0, z] in one shot; order 40 is the smallest
# order with margin below 1e-12 relative error up to z = 20 for ΛCDM and w0waCDM
# (pinned against adaptive quadgk in the tests).
const _GRID_GAUSS_LEGENDRE = QuadGK.gauss(Float64, 4)
const _SCALAR_GAUSS_LEGENDRE = QuadGK.gauss(Float64, 40)

"""
    _gauss_legendre_integral(f, a, b, nodes, weights) -> typeof(f(midpoint))

Integral of `f` over `[a, b]` with the given reference Gauss–Legendre `nodes` and
`weights` on [-1, 1], mapped affinely onto the interval. The accumulation starts
from `f`'s own output type, so `ForwardDiff.Dual` cosmology parameters propagate.
"""
function _gauss_legendre_integral(f, a::Real, b::Real, nodes, weights)
    half = (b - a) / 2
    mid = (a + b) / 2
    acc = zero(half * f(mid))
    @inbounds for k in eachindex(nodes)
        acc += weights[k] * f(mid + half * nodes[k])
    end
    return half * acc
end

function comoving_distance(z::Real, c::AbstractCosmology)
    Ez = E(z, c)
    pref = SPEED_OF_LIGHT_KM_S / (H0(c) * Ez)
    z == zero(z) && return zero(pref)
    nodes, weights = _SCALAR_GAUSS_LEGENDRE
    integral = _gauss_legendre_integral(x -> inv(E(x, c)), zero(z), z, nodes, weights)
    return pref * integral * Ez
end

function luminosity_distance(z::Real, c::AbstractCosmology)
    (1 + z) * comoving_distance(z, c)
end

"""
    hubble_distance(c::AbstractCosmology) -> Real

Hubble distance `d_h = c / H0` in Mpc. The `c` is the speed of light and `H0(c)` the
present-day Hubble parameter.
"""
function hubble_distance(c::AbstractCosmology)
    SPEED_OF_LIGHT_KM_S / H0(c)
end

"""
    differential_comoving_volume(z, c) -> Real

Differential comoving volume element `4π · d_h · d_c(z)² / E(z)`, i.e. the
solid-angle-integrated `dV_c/dz` in Mpc³ per unit redshift. The `4π` is owned here, not
by the redshift-distribution consumers, so a tabulated grid from
[`distance_and_volume_grid`](@ref) and this scalar function agree.
"""
function differential_comoving_volume(z::Real, c::AbstractCosmology)
    d_h = hubble_distance(c)
    d_c = comoving_distance(z, c)
    return 4π * d_h * d_c^2 / E(z, c)
end

"""
    distance_and_volume_grid(c::AbstractCosmology, z)
        -> (; comoving_distance, luminosity_distance, differential_comoving_volume)

Tabulate the three distance quantities on `z` in a single pass, sharing one
`1/E(z)` evaluation pass and one cumulative quadrature between them.

`differential_comoving_volume` is the solid-angle-integrated `4π · dV_c/dz` in
Mpc³ per unit redshift — the `4π` lives here, not in the redshift-distribution
consumers, matching the Python `astrogwb` stack.

The redshift integral is a composite fixed-order Gauss–Legendre rule — 4 nodes
per grid interval, with `0` prepended internally — matching astrogwb's
`distance_and_volume_grid` to rounding error for the smooth integrand `1/E(z)`:
`d_c = d_h · cumsum(∫ 1/E)`, `d_L = (1+z) · d_c`, `dV_c/dz = 4π · d_h · d_c² / E(z)`.
This is the efficient batched path for models that already evaluate and normalize
quantities on a redshift grid. Scalar distance calls use a single fixed-order
Gauss–Legendre rule over `[0, z]` instead. The grid approximation and subsequent
interpolation policy belong to the caller.

Takes the grid **array**, not `(z_min, z_max, n)`, so the caller's grid is the grid used
— there is no way for tabulation and interpolation to disagree about nodes. The grid
must be non-negative and strictly increasing, with at least one node (astrogwb's
contract): a grid that does not start at zero is fine, because the cumulative
comoving-distance integral always starts from the internally prepended `d_c(0) = 0`.
"""
function distance_and_volume_grid(c::AbstractCosmology, z::AbstractVector{<:Real})
    length(z) >= 1 || throw(ArgumentError(
        "distance_and_volume_grid requires at least one grid point"))
    all(>=(0), z) || throw(ArgumentError(
        "distance_and_volume_grid requires a non-negative grid"))
    all(diff(z) .> 0) || throw(ArgumentError(
        "distance_and_volume_grid requires a strictly increasing grid"))
    inv_E = inv.(E.(z, Ref(c)))
    nodes, weights = _GRID_GAUSS_LEGENDRE
    # Cumulative ∫₀^z 1/E over each interval [z_{i-1}, z_i] (z_{-1} = 0 prepended),
    # accumulated left to right so the summation order matches astrogwb's `cumsum`.
    integral = similar(inv_E)
    acc = zero(eltype(integral))
    z_prev = zero(eltype(z))
    @inbounds for (i, z_i) in enumerate(z)
        acc += _gauss_legendre_integral(x -> inv(E(x, c)), z_prev, z_i, nodes, weights)
        integral[i] = acc
        z_prev = z_i
    end
    d_h = hubble_distance(c)
    d_c = d_h .* integral
    return (;
        comoving_distance = d_c,
        luminosity_distance = (1 .+ z) .* d_c,
        differential_comoving_volume = @. 4π * d_h * d_c^2 * inv_E
    )
end
