using CBCDistributions: AbstractCosmology

"""
Madau-Dickinson population with modified gravitational-wave propagation.

Type parameter `C <: AbstractCosmology` selects the cosmology model.
`MadauDickinsonModifiedPropagation()` defaults to `LambdaCDM`.
"""
struct MadauDickinsonModifiedPropagation{C <: AbstractCosmology} <: AbstractASGWBModel end

MadauDickinsonModifiedPropagation() = MadauDickinsonModifiedPropagation{LambdaCDM}()

"""
    model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols owned by the Madau–Dickinson modified-propagation forward model
(excluding cosmology parameters).
"""
model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) = (:Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

"""
    cosmology_type(model::MadauDickinsonModifiedPropagation{C}) -> Type{C}

Return the cosmology subtype baked into the forward model's type parameter.
"""
cosmology_type(::MadauDickinsonModifiedPropagation{C}) where {C <: AbstractCosmology} = C

function external_model_parameter_names(::MadauDickinsonModifiedPropagation)
    return (
        Ξ₀ = "Xi_0",
        Ξₙ = "Xi_n",
        γ = "gamma",
        κ = "kappa",
        zpeak = "z_peak"
    )
end

"""
    gravitational_wave_distance(m::MadauDickinsonModifiedPropagation, z, d_l, Λ)

Modified GW luminosity distance: destructures `Λ.Ξ₀`/`Λ.Ξₙ` and delegates to the
scalar CBCDistributions hook.
"""
function gravitational_wave_distance(
        ::MadauDickinsonModifiedPropagation,
        z::Real,
        d_l::Real,
        Λ::NamedTuple
)
    return gravitational_wave_distance(z, d_l, Λ.Ξ₀, Λ.Ξₙ)
end

"""
    model_hyperparameters(data, model::MadauDickinsonModifiedPropagation) -> NamedTuple

Build the canonical fiducial hyperparameter state for `model` from parsed
`model.toml` data.
"""
function model_hyperparameters(
        data::AbstractDict,
        model::MadauDickinsonModifiedPropagation{C}
) where {C <: AbstractCosmology}
    cosmology_table = _require_table(data, "cosmology")
    mg_table = _require_table(data, "modified_gravity")
    population_table = _require_table(data, "population")

    raw = (;
        _external_values(cosmology_table, external_parameter_names(C), "cosmology")...,
        _external_values(mg_table, external_model_parameter_names(model)[(:Ξ₀, :Ξₙ)],
            "modified_gravity")...,
        _external_values(
            population_table, external_model_parameter_names(model)[(:γ, :κ, :zpeak)],
            "population")...
    )
    return canonical_hyperparameters(model, raw; context = "fiducial hyperparameters")
end

"""
    redshift_prior_spec(data, model::MadauDickinsonModifiedPropagation) -> RedshiftPriorSpec

Build the redshift prior spec for the model from parsed `model.toml` data.
"""
function redshift_prior_spec(
        data::AbstractDict,
        ::MadauDickinsonModifiedPropagation
)
    redshift_table = _require_table(data, "redshift")
    tdm = _parse_time_delay_model(get(redshift_table, "time_delay_model", "none"))
    return RedshiftPriorSpec(
        MadauDickinson,
        _require_real(redshift_table, "z_min", "redshift"),
        _require_real(redshift_table, "z_max", "redshift"),
        Int(redshift_table["num_interp"]),
        tdm
    )
end

function model_config_dict(config::ModelConfig{<:MadauDickinsonModifiedPropagation})
    model = config.model
    Λ = config.fiducial_hyperparameters
    model_mapping = external_model_parameter_names(model)
    spec = config.redshift_prior_spec
    return Dict{String, Any}(
        "model" => Dict{String, Any}(
            "name" => MADAU_DICKINSON_MODIFIED_PROPAGATION_CONFIG_NAME,
            "cosmology" => cosmology_config_name(cosmology_type(model))
        ),
        "cosmology" => _cosmology_config_dict(model, Λ),
        "modified_gravity" => _model_parameter_config_dict(model_mapping[(:Ξ₀, :Ξₙ)], Λ),
        "population" => _model_parameter_config_dict(model_mapping[(:γ, :κ, :zpeak)], Λ),
        "redshift" => Dict{String, Any}(
            "z_min" => spec.z_min,
            "z_max" => spec.z_max,
            "num_interp" => spec.num_interp,
            "time_delay_model" => spec.time_delay_model === nothing ? "none" :
                                  spec.time_delay_model
        )
    )
end
