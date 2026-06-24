const SPEED_OF_LIGHT_KM_S = 299792.458

"""1 Mpc in meters (IAU/CODATA convention)."""
const METERS_PER_MPC = 3.085677581e22

"""
    hubble_constant_si(H0_km_s_mpc::Real) -> Float64

Hubble constant in s⁻¹ from ``H_0`` in **km/s/Mpc** (same units as [`LambdaCDM`](@ref).`H0`).
"""
function hubble_constant_si(H0_km_s_mpc::Real)
    return Float64(H0_km_s_mpc) * 1000.0 / METERS_PER_MPC
end

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
Modified gravitational-wave propagation layered over a base FLRW cosmology.
Electromagnetic distances and expansion history are delegated to `base`; the
GW luminosity distance applies the usual `(Ξ₀, Ξₙ)` propagation factor.
"""
struct ModifiedPropagation{C <: AbstractCosmology, TΞ0 <: Real, TΞn <: Real} <:
       AbstractCosmology
    base::C
    Ξ₀::TΞ0
    Ξₙ::TΞn
end

base_cosmology(c::AbstractCosmology) = c
base_cosmology(c::ModifiedPropagation) = c.base
H0(c::ModifiedPropagation) = H0(c.base)
Ωm(c::ModifiedPropagation) = Ωm(c.base)

"""Supported configurable cosmology subtypes (registration order)."""
const SUPPORTED_COSMOLOGIES = (
    LambdaCDM,
    W0CDM,
    W0WaCDM,
    ModifiedPropagation{LambdaCDM},
    ModifiedPropagation{W0CDM},
    ModifiedPropagation{W0WaCDM}
)

"""
    hyperparameters(::Type{C}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols for cosmology subtype `C`, in struct field order.
"""
hyperparameters(::Type{C}) where {C <: AbstractCosmology} = fieldnames(C)
function hyperparameters(::Type{<:ModifiedPropagation{C}}) where {C <: AbstractCosmology}
    return (hyperparameters(C)..., :Ξ₀, :Ξₙ)
end

const cosmology_parameters = hyperparameters

"""
    cosmology(::Type{C}, h::NamedTuple) -> C

Build cosmology subtype `C` from hyperparameter state `h` (fields must match [`hyperparameters`](@ref)(`C`)).
"""
function cosmology(::Type{C}, h::NamedTuple) where {C <: AbstractCosmology}
    fn = fieldnames(C)
    return C(ntuple(i -> h[fn[i]], Val(length(fn)))...)
end

function cosmology(::Type{<:ModifiedPropagation{C}}, h::NamedTuple) where {C <:
                                                                           AbstractCosmology}
    return ModifiedPropagation(cosmology(C, h), h.Ξ₀, h.Ξₙ)
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

cosmology_config_name(::Type{LambdaCDM}) = "LambdaCDM"
cosmology_config_name(::Type{W0CDM}) = "W0CDM"
cosmology_config_name(::Type{W0WaCDM}) = "W0WaCDM"
function cosmology_config_name(::Type{<:ModifiedPropagation{C}}) where {C <:
                                                                        AbstractCosmology}
    return "ModifiedPropagation{$(cosmology_config_name(C))}"
end

const _COSMOLOGY_BY_CONFIG_NAME = Dict(
    cosmology_config_name(C) => C for C in SUPPORTED_COSMOLOGIES
)

"""
    cosmology_type(name::AbstractString) -> Type{<:AbstractCosmology}

Resolve a config/TOML cosmology name to a concrete subtype.
"""
function cosmology_type(name::AbstractString)
    C = get(_COSMOLOGY_BY_CONFIG_NAME, String(name), nothing)
    C === nothing && throw(
        ArgumentError(
        "unknown cosmology \"$(name)\"; valid choices: $(sort(collect(keys(_COSMOLOGY_BY_CONFIG_NAME))))",
    ),
    )
    return C
end

"""
    dark_energy_eos(c::AbstractCosmology, z) -> Real

Dark energy equation of state w(z).
"""
dark_energy_eos(::LambdaCDM, z) = -one(z)
dark_energy_eos(c::W0CDM, z) = c.w0
dark_energy_eos(c::W0WaCDM, z) = c.w0 + c.wa * z / (1 + z)
dark_energy_eos(c::ModifiedPropagation, z) = dark_energy_eos(c.base, z)

"""
    de_density_ratio(c::AbstractCosmology, z) -> Real

Ratio ρ_DE(z)/ρ_DE(0): closed-form integral of `dark_energy_eos` through the Friedmann equation.
"""
de_density_ratio(::LambdaCDM, z) = one(z)
de_density_ratio(c::W0CDM, z) = (1 + z)^(3 * (1 + c.w0))
function de_density_ratio(c::W0WaCDM, z)
    (1 + z)^(3 * (1 + c.w0 + c.wa)) * exp(-3 * c.wa * z / (1 + z))
end
de_density_ratio(c::ModifiedPropagation, z) = de_density_ratio(c.base, z)

"""
    E(z, c::AbstractCosmology) -> Real

Hubble parameter ratio E(z) = H(z)/H₀ for flat FLRW cosmology.
"""
function E(z::Real, c::AbstractCosmology)
    return sqrt(Ωm(c) * (1 + z)^3 + (1 - Ωm(c)) * de_density_ratio(c, z))
end
