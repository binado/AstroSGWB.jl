"""Abstract supertype for flat FLRW cosmology models."""
abstract type AbstractCosmology end

"""Flat ΛCDM cosmology (w=-1, radiation-free)."""
struct LambdaCDM{TH0 <: Real, TΩm <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
end

"""Flat wCDM cosmology with constant dark-energy equation of state w0."""
struct W0CDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
end

"""Flat w0waCDM (CPL) cosmology: w(z) = w0 + wa·z/(1+z)."""
struct W0WaCDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real, Twa <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
    wa::Twa
end

H0(c::AbstractCosmology) = c.H0
Ωm(c::AbstractCosmology) = c.Ωm

Base.broadcastable(c::AbstractCosmology) = Ref(c)

"""
    cosmology(::Type{C}, h::NamedTuple) -> C

Build cosmology subtype `C` from the corresponding fields in hyperparameter state `h`.
"""
function cosmology(::Type{C}, h::NamedTuple) where {C <: AbstractCosmology}
    fn = fieldnames(C)
    return C(ntuple(i -> h[fn[i]], Val(length(fn)))...)
end

"""
    cosmology(h::NamedTuple) -> AbstractCosmology

Infer cosmology subtype from keys in `h` (`:wa` → [`W0WaCDM`](@ref), `:w0` → [`W0CDM`](@ref), else [`LambdaCDM`](@ref)).
"""
function cosmology(h::NamedTuple)
    :wa in keys(h) && return cosmology(W0WaCDM, h)
    :w0 in keys(h) && return cosmology(W0CDM, h)
    return cosmology(LambdaCDM, h)
end

function (::Type{C})(h::NamedTuple) where {C <: AbstractCosmology}
    return cosmology(C, h)
end

"""
    dark_energy_eos(c::AbstractCosmology, z) -> Real

Dark energy equation of state w(z).
"""
dark_energy_eos(::LambdaCDM, z) = -one(z)
dark_energy_eos(c::W0CDM, z) = c.w0
dark_energy_eos(c::W0WaCDM, z) = c.w0 + c.wa * z / (1 + z)

"""
    de_density_ratio(c::AbstractCosmology, z) -> Real

Ratio ρ_DE(z)/ρ_DE(0): closed-form integral of `dark_energy_eos` through the Friedmann equation.
"""
de_density_ratio(::LambdaCDM, z) = one(z)
de_density_ratio(c::W0CDM, z) = (1 + z)^(3 * (1 + c.w0))
function de_density_ratio(c::W0WaCDM, z)
    (1 + z)^(3 * (1 + c.w0 + c.wa)) * exp(-3 * c.wa * z / (1 + z))
end

"""
    E(z, c::AbstractCosmology) -> Real

Hubble parameter ratio E(z) = H(z)/H₀ for flat FLRW cosmology.
"""
function E(z::Real, c::AbstractCosmology)
    return sqrt(Ωm(c) * (1 + z)^3 + (1 - Ωm(c)) * de_density_ratio(c, z))
end
