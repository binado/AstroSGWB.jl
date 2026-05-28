using Distributions: Uniform, product_distribution

"""
    BNSPopulationModel <: PopulationModel

Full binary-neutron-star population model: Madau–Dickinson source-frame
redshift distribution plus uniform mass, spin, and tidal-deformability priors.
Implements the three-method [`PopulationModel`](@ref) contract.  This is the
production caller model; the framework owns no concrete population types.
"""
struct BNSPopulationModel <: PopulationModel end

hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

function hyperprior(::BNSPopulationModel)
    return product_distribution((
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
end

function single_event_prior(::BNSPopulationModel, cosmo::AbstractCosmology, Λ::NamedTuple)
    z_d = redshift_prior(MadauDickinsonSourceFrame(), cosmo, Λ)
    spin = AlignedSpinChiSimple()
    return product_distribution((
        mass = OrderedUniformSourceMassPair(),
        redshift = z_d,
        χ₁ = spin,
        χ₂ = spin,
        Λ₁ = Uniform(0.0, BNS_LAMBDA_HIGH),
        Λ₂ = Uniform(0.0, BNS_LAMBDA_HIGH)
    ))
end

"""
    POPULATION_REGISTRY

Maps `[model].population` names to concrete [`PopulationModel`](@ref) instances.
Passed into [`load_problem`](@ref)/`load_model_toml` by the inference CLI.
"""
const POPULATION_REGISTRY = Dict{String, PopulationModel}("bns" => BNSPopulationModel())
