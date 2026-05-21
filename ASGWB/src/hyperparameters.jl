using Distributions: ProductNamedTupleDistribution

"""
    hyperparameter_order(prior::ProductNamedTupleDistribution)

Symbols and order used by `Bijectors.link` / HMC unconstrained vectors (`keys(prior.dists)`).
The prior is the single source of truth for flat parameter layout.
"""
hyperparameter_order(prior::ProductNamedTupleDistribution) = keys(prior.dists)

"""
    validate_sample_only!(sample_only, prior::ProductNamedTupleDistribution)

Validate `sample_only` against [`hyperparameter_order`](@ref). Pass `nothing` to sample all
hyperparameters. Throws `ArgumentError` on empty, duplicate, or unknown symbols.
"""
function validate_sample_only!(
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}},
        prior::ProductNamedTupleDistribution
)
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    order = hyperparameter_order(prior)
    for s in sample_only
        s in order || throw(
            ArgumentError(
            "sample_only contains $(repr(s)); expected symbols from $(Tuple(order))",
        ),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

"""
    coerce_hyperparameters(; H0, Ωm, Ξ₀=1.0, Ξₙ=0.0, γ, κ, zpeak) -> NamedTuple

Coerce Madau–Dickinson hyperparameters to `Float64` for init, cache load, and sampler boundaries.
Inner likelihood paths accept any `NamedTuple` (including `ForwardDiff.Dual` fields during AD).
"""
function coerce_hyperparameters(;
        H0::Real,
        Ωm::Real,
        Ξ₀::Real = 1.0,
        Ξₙ::Real = 0.0,
        γ::Real,
        κ::Real,
        zpeak::Real
)
    return (
        H0 = Float64(H0),
        Ωm = Float64(Ωm),
        Ξ₀ = Float64(Ξ₀),
        Ξₙ = Float64(Ξₙ),
        γ = Float64(γ),
        κ = Float64(κ),
        zpeak = Float64(zpeak)
    )
end

"""
    coerce_hyperparameters(nt::NamedTuple) -> NamedTuple

Build a `Float64` hyperparameter `NamedTuple` from any tuple with at least
`:H0, :Ωm, :γ, :κ, :zpeak`. `Ξ₀` / `Ξₙ` default to `1.0` / `0.0` when absent.
"""
function coerce_hyperparameters(nt::NamedTuple)
    return coerce_hyperparameters(;
        H0 = nt.H0,
        Ωm = nt.Ωm,
        Ξ₀ = haskey(nt, :Ξ₀) ? nt.Ξ₀ : 1.0,
        Ξₙ = haskey(nt, :Ξₙ) ? nt.Ξₙ : 0.0,
        γ = nt.γ,
        κ = nt.κ,
        zpeak = nt.zpeak
    )
end
