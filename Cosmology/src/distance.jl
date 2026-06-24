using QuadGK

struct CosmologyCache{C <: AbstractCosmology, I <: CumulativeIntegral1D, TD <: Real}
    cosmology::C
    inv_E_integral::I
    d_h::TD
end

function CosmologyCache(cosmology::AbstractCosmology, z_grid::AbstractVector{<:Real})
    inv_E_integral = CumulativeIntegral1D(z_grid, z -> inv(E(z, cosmology)))
    d_h = SPEED_OF_LIGHT_KM_S / H0(cosmology)
    return CosmologyCache(cosmology, inv_E_integral, d_h)
end

function comoving_distance(z::Real, c::AbstractCosmology)
    Ez = E(z, c)
    pref = SPEED_OF_LIGHT_KM_S / (H0(c) * Ez)
    z == zero(z) && return zero(pref)
    integral, _ = quadgk(x -> inv(E(x, c)), zero(z), z)
    return pref * integral * Ez
end

function luminosity_distance(z::Real, c::AbstractCosmology)
    (1 + z) * comoving_distance(z, c)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c)
    return d_h * d_c^2 / E(z, c)
end

function comoving_distance(z::Real, cache::CosmologyCache)
    cache.d_h * cdf(cache.inv_E_integral, z)
end

function luminosity_distance(z::Real, cache::CosmologyCache)
    (1 + z) * comoving_distance(z, cache)
end

function differential_comoving_volume(z::Real, cache::CosmologyCache)
    d_c = comoving_distance(z, cache)
    return cache.d_h * d_c^2 / E(z, cache.cosmology)
end

"""
    comoving_distance(z, c::AbstractCosmology, dist::CumulativeIntegral1D) -> Real

Comoving distance using a precomputed [`CumulativeIntegral1D`](@ref) of
`w -> 1/E(w, c)`. Uses [`cdf`](@ref) which returns the exact integral under
the linear interpolant (analytic trapezoidal rule).
"""
function comoving_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (SPEED_OF_LIGHT_KM_S / H0(c)) * cdf(dist, z)
end

function luminosity_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (1 + z) * comoving_distance(z, c, dist)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c, dist)
    return d_h * d_c^2 / E(z, c)
end

"""
    gw_em_distance_ratio(z, prop) -> Real

Ratio `Ξ(z) = D_gw / D_L` between the gravitational-wave and electromagnetic luminosity
distances at redshift `z`. [`GR`](@ref) recovers `Ξ ≡ 1`; a [`ModifiedPropagation`](@ref)
applies the `(Ξ₀, Ξₙ)` factor

``\\Xi(z) = \\Xi_0 + (1 - \\Xi_0) / (1 + z)^{\\Xi_n}``.

This is the single source of truth for the propagation factor; `gravitational_wave_distance`
is `gw_em_distance_ratio(z, prop) * D_L`.
"""
gw_em_distance_ratio(z::Real, Ξ₀::Real, Ξₙ::Real) = Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ
gw_em_distance_ratio(z::Real, ::GR) = one(z)
gw_em_distance_ratio(z::Real, p::ModifiedPropagation) = gw_em_distance_ratio(z, p.Ξ₀, p.Ξₙ)

function gravitational_wave_distance(
        z::Real,
        luminosity_distance::Real,
        Ξ₀::Real,
        Ξₙ::Real
)
    return gw_em_distance_ratio(z, Ξ₀, Ξₙ) * luminosity_distance
end

# GW luminosity distance from a precomputed EM luminosity distance `d_l`. The propagation
# dispatch lives entirely in `gw_em_distance_ratio` (1 for `GR`, the (Ξ₀, Ξₙ) factor for
# `ModifiedPropagation`), so this is just `Ξ(z) * d_l`.
function gravitational_wave_distance(z::Real, d_l::Real, prop::AbstractPropagation)
    gw_em_distance_ratio(z, prop) * d_l
end

function gravitational_wave_distance(z::Real, c::AbstractCosmology, prop::AbstractPropagation)
    return gravitational_wave_distance(z, luminosity_distance(z, c), prop)
end
