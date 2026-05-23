using Distributions: ProductNamedTupleDistribution

"""Abstract supertype for ASGWB forward models with explicit hyperparameter contracts."""
abstract type AbstractASGWBModel end

"""
Madau-Dickinson population with modified gravitational-wave propagation.

Type parameter `C <: AbstractCosmology` selects the cosmology model.
`MadauDickinsonModifiedPropagation()` defaults to `LambdaCDM`.
"""
struct MadauDickinsonModifiedPropagation{C <: AbstractCosmology} <: AbstractASGWBModel end

MadauDickinsonModifiedPropagation() = MadauDickinsonModifiedPropagation{LambdaCDM}()

"""
    hyperparameters(model::AbstractASGWBModel) -> Tuple{Vararg{Symbol}}

Symbols and order used by a model's flat hyperparameter state.
"""
function hyperparameters end

function hyperparameters(::MadauDickinsonModifiedPropagation{LambdaCDM})
    (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
end
function hyperparameters(::MadauDickinsonModifiedPropagation{W0CDM})
    (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
end
function hyperparameters(::MadauDickinsonModifiedPropagation{W0WaCDM})
    (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
end

"""
    build_cosmology(model::AbstractASGWBModel, h::NamedTuple) -> AbstractCosmology

Construct the cosmology subtype for `model` from the live hyperparameter state `h`.
"""
function build_cosmology(::MadauDickinsonModifiedPropagation{LambdaCDM}, h::NamedTuple)
    LambdaCDM(h.H0, h.Ωm)
end
function build_cosmology(::MadauDickinsonModifiedPropagation{W0CDM}, h::NamedTuple)
    W0CDM(h.H0, h.Ωm, h.w0)
end
function build_cosmology(::MadauDickinsonModifiedPropagation{W0WaCDM}, h::NamedTuple)
    W0WaCDM(h.H0, h.Ωm, h.w0, h.wa)
end

function _check_unique_hyperparameters(model::AbstractASGWBModel)
    order = hyperparameters(model)
    isempty(order) && throw(
        ArgumentError("$(typeof(model)) must define at least one hyperparameter"),
    )
    length(unique(order)) == length(order) || throw(
        ArgumentError("$(typeof(model)) defines duplicate hyperparameters: $(order)"),
    )
    return order
end

"""
    validate_subset(subset::Union{Tuple{Vararg{Symbol}}, NamedTuple}, order) -> subset

Validate that `subset` (either a tuple of symbols or a NamedTuple) contains only
unique symbols that are a subset of `order`. Allows empty subsets. Throws `ArgumentError`
on duplicates or unknown symbols.
"""
function validate_subset(
        subset::Tuple{Vararg{Symbol}},
        order::Union{Tuple{Vararg{Symbol}}, Base.KeySet, AbstractVector{Symbol}}
)
    for s in subset
        s in order || throw(
            ArgumentError(
            "subset contains $(repr(s)); expected symbols from $(Tuple(order))",
        ),
        )
    end
    length(unique(subset)) == length(subset) ||
        throw(ArgumentError("subset must not repeat symbols"))
    return subset
end

function validate_subset(
        subset::NamedTuple,
        order::Union{Tuple{Vararg{Symbol}}, Base.KeySet, AbstractVector{Symbol}}
)
    validate_subset(keys(subset), order)
    return subset
end

"""
    validate_subset(subset, model::AbstractASGWBModel) -> subset
"""
function validate_subset(subset, model::AbstractASGWBModel)
    validate_subset(subset, _check_unique_hyperparameters(model))
end

"""
    validate_subset(subset, prior::ProductNamedTupleDistribution) -> subset
"""
function validate_subset(subset, prior::ProductNamedTupleDistribution)
    validate_subset(subset, keys(prior.dists))
end


"""
    validate_hyperparameters(model, Λ; context="hyperparameters")

Require `Λ` to contain exactly the model hyperparameters.
"""
function validate_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters"
)
    validate_subset(Λ, model)
    order = hyperparameters(model)
    if length(keys(Λ)) != length(order)
        missing = Tuple(s for s in order if s ∉ keys(Λ))
        throw(ArgumentError("$(context) must match $(typeof(model)); missing $(missing)"))
    end
    return nothing
end

"""
    canonical_hyperparameters(model, Λ; context="hyperparameters", eltype=Float64) -> NamedTuple

Validate a hyperparameter tuple against `model`, reorder it to the model's canonical
hyperparameter order, and convert each value to `eltype`.
"""
function canonical_hyperparameters(
        model::AbstractASGWBModel,
        Λ::NamedTuple;
        context::AbstractString = "hyperparameters",
        eltype::Type = Float64
)
    validate_hyperparameters(model, Λ; context = context)
    return (; (k => eltype(Λ[k]) for k in hyperparameters(model))...)
end

"""
    validate_prior(model, prior)

Require a product prior's named sites and order to match the model hyperparameters.
"""
function validate_prior(
        model::AbstractASGWBModel,
        prior::ProductNamedTupleDistribution
)
    order = _check_unique_hyperparameters(model)
    prior_order = keys(prior.dists)
    prior_order == order || throw(
        ArgumentError(
        "prior hyperparameters must match $(typeof(model)); expected $(order), got $(prior_order)",
    ),
    )
    return nothing
end
