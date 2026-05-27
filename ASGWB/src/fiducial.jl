using SHA: sha256
using TOML

"""
    ModifiedGravity

Fiducial modified-gravity propagation parameters: `Ξ₀` (overall amplitude) and
`Ξₙ` (running). These scale gravitational-wave luminosity distance relative to
electromagnetic luminosity distance via `d_gw = d_L × (Ξ₀ + (1 - Ξ₀)(1+z)^(-Ξₙ))`.
"""
struct ModifiedGravity
    Ξ₀::Float64
    Ξₙ::Float64
end

"""
    PopulationParams

Fiducial merger-rate population scalars. `family` selects the redshift-prior
functional form. For `MadauDickinson`: `γ`, `κ`, `zpeak` characterize the
star-formation–like rate. For `PowerLaw`: `Λ` is the power-law index.
`z_min`, `z_max`, `num_interp`, `time_delay_model` are grid/integration settings
shared with [`RedshiftPriorSpec`](@ref).
"""
struct PopulationParams
    family::RedshiftPriorFamily
    γ::Float64
    κ::Float64
    zpeak::Float64
    Λ::Union{Nothing, Float64}
    z_min::Float64
    z_max::Float64
    num_interp::Int
    time_delay_model::Union{Nothing, String}
end

"""
    ObservationParams

Fiducial observation metadata: local BNS merger rate (Gpc⁻³ yr⁻¹) and
analysis observation time (years). These are part of `cosmology.toml` so the bundle
records which observation scenario the samples were drawn for.
"""
struct ObservationParams
    local_merger_rate::Float64
    observation_time_yr::Float64
end

"""
    FiducialParameters

Fiducial physics snapshot loaded from `cosmology.toml`. Groups four orthogonal
concerns into one typed struct:
- `cosmology::AbstractCosmology` — the background expansion model (`LambdaCDM`, `W0CDM`, or `W0WaCDM`)
- `modified_gravity::ModifiedGravity` — GW-propagation Ξ₀/Ξₙ scalars
- `population::PopulationParams` — the merger-rate redshift prior family and scalars
- `observation::ObservationParams` — local merger rate and observation time
"""
struct FiducialParameters
    cosmology::AbstractCosmology
    modified_gravity::ModifiedGravity
    population::PopulationParams
    observation::ObservationParams
end

"""
    cosmology_type(fid::FiducialParameters) -> Type{<:AbstractCosmology}

Concrete cosmology subtype encoded in the fiducial.
"""
cosmology_type(fid::FiducialParameters) = Base.typename(typeof(fid.cosmology)).wrapper

"""
    propagation_model(fid::FiducialParameters) -> MadauDickinsonModifiedPropagation

[`MadauDickinsonModifiedPropagation`](@ref) with cosmology type parameter matching `fid`.
"""
propagation_model(fid::FiducialParameters) = MadauDickinsonModifiedPropagation{cosmology_type(fid)}()

fiducial_cosmology(fid::FiducialParameters) = fid.cosmology

"""
    redshift_prior_spec(fid::FiducialParameters) -> RedshiftPriorSpec

Build a [`RedshiftPriorSpec`](@ref) from the population parameters of `fid`.
"""
function redshift_prior_spec(fid::FiducialParameters)
    return RedshiftPriorSpec(
        fid.population.family,
        fid.population.z_min,
        fid.population.z_max,
        fid.population.num_interp,
        fid.population.time_delay_model
    )
end

"""
    cosmology_sha256_of_file(path) -> String

SHA-256 hex digest of the raw bytes of `path`.
"""
function cosmology_sha256_of_file(path::AbstractString)::String
    return bytes2hex(sha256(read(path)))
end

function _cosmology_nt(c::AbstractCosmology)
    fn = fieldnames(typeof(c))
    return NamedTuple{fn}(Tuple(getfield(c, f) for f in fn))
end

"""
    hyperparameters_from_fiducial(fid::FiducialParameters, spec::RedshiftPriorSpec) -> NamedTuple

Build model-validated fiducial hyperparameters from `fid`. Used when reconstructing
per-sample proposal log-density or redshift integrals from the fiducial population.
Requires a `MadauDickinson` population family.
"""
function hyperparameters_from_fiducial(fid::FiducialParameters, spec::RedshiftPriorSpec)
    spec.family == MadauDickinson || throw(
        ArgumentError(
            "live hyperparameter reconstruction supports MadauDickinson only; " *
            "PowerLaw caches are metadata-only",
        ),
    )
    model = propagation_model(fid)
    return canonical_hyperparameters(
        model,
        (;
            _cosmology_nt(fid.cosmology)...,
            Ξ₀ = fid.modified_gravity.Ξ₀,
            Ξₙ = fid.modified_gravity.Ξₙ,
            γ = fid.population.γ,
            κ = fid.population.κ,
            zpeak = fid.population.zpeak
        );
        context = "fiducial hyperparameters"
    )
end

"""
    fiducial_redshift_integral(fid::FiducialParameters, spec::RedshiftPriorSpec) -> Float64

Trapezoid integral ``\\int p(z)\\,dz`` of the detector-frame merger-rate density on the
redshift grid defined by `spec` and the population in `hyperparameters_from_fiducial(fid, spec)`.
"""
function fiducial_redshift_integral(fid::FiducialParameters, spec::RedshiftPriorSpec)::Float64
    Λ = hyperparameters_from_fiducial(fid, spec)
    redshift_prior = build_redshift_prior(Λ, spec, cosmology(propagation_model(fid), Λ))
    return Float64(redshift_integral(redshift_prior))
end

function _parse_cosmology(dict::Dict)::AbstractCosmology
    ctype_str = get(dict, "type", "LambdaCDM")::String
    H0 = Float64(dict["H0"])
    Omega_m = Float64(dict["Omega_m"])
    if ctype_str == "LambdaCDM"
        return LambdaCDM(H0, Omega_m)
    elseif ctype_str == "W0CDM"
        haskey(dict, "w0") || throw(
            ArgumentError("cosmology.toml [cosmology] type=W0CDM requires w0"),
        )
        return W0CDM(H0, Omega_m, Float64(dict["w0"]))
    elseif ctype_str == "W0WaCDM"
        haskey(dict, "w0") && haskey(dict, "wa") || throw(
            ArgumentError("cosmology.toml [cosmology] type=W0WaCDM requires w0 and wa"),
        )
        return W0WaCDM(H0, Omega_m, Float64(dict["w0"]), Float64(dict["wa"]))
    else
        throw(ArgumentError("unknown cosmology type $(repr(ctype_str)); expected LambdaCDM, W0CDM, or W0WaCDM"))
    end
end

function _parse_population(dict::Dict)::PopulationParams
    family_str = get(dict, "family", "madau_dickinson")::String
    family = parse_redshift_prior_family(family_str)
    tdm = get(dict, "time_delay_model", "none")::String
    tdm_val = (tdm == "none" || isempty(strip(tdm))) ? nothing : tdm
    if family == MadauDickinson
        return PopulationParams(
            family,
            Float64(dict["gamma"]),
            Float64(dict["kappa"]),
            Float64(dict["z_peak"]),
            nothing,
            Float64(dict["z_min"]),
            Float64(dict["z_max"]),
            Int(dict["num_interp"]),
            tdm_val
        )
    else
        return PopulationParams(
            family,
            0.0,
            0.0,
            0.0,
            Float64(dict["lamb"]),
            Float64(dict["z_min"]),
            Float64(dict["z_max"]),
            Int(dict["num_interp"]),
            tdm_val
        )
    end
end

"""
    load_cosmology_toml(path) -> FiducialParameters

Parse a sectioned `cosmology.toml` file into a [`FiducialParameters`](@ref)
(`cosmology`, `modified_gravity`, `population`, `observation` sections).
"""
function load_cosmology_toml(path::AbstractString)::FiducialParameters
    dict = TOML.parsefile(path)

    cosmo = _parse_cosmology(get(dict, "cosmology", Dict{String, Any}()))

    mg_dict = get(dict, "modified_gravity", Dict{String, Any}())
    mg = ModifiedGravity(
        Float64(mg_dict["Xi_0"]),
        Float64(mg_dict["Xi_n"])
    )

    pop = _parse_population(get(dict, "population", Dict{String, Any}()))

    obs_dict = get(dict, "observation", Dict{String, Any}())
    obs = ObservationParams(
        Float64(obs_dict["local_merger_rate"]),
        Float64(obs_dict["observation_time_yr"])
    )

    return FiducialParameters(cosmo, mg, pop, obs)
end

"""
    save_cosmology_toml(path, fid::FiducialParameters)

Write `fid` to a sectioned `cosmology.toml` file readable by [`load_cosmology_toml`](@ref).
"""
function save_cosmology_toml(path::AbstractString, fid::FiducialParameters)
    cosmo = fid.cosmology
    C = cosmology_type(fid)
    cosmo_dict = Dict{String, Any}(
        "type" => string(nameof(C)),
        "H0" => cosmo.H0,
        "Omega_m" => cosmo.Ωm,
    )
    if C <: Union{W0CDM, W0WaCDM}
        cosmo_dict["w0"] = cosmo.w0
    end
    if C <: W0WaCDM
        cosmo_dict["wa"] = cosmo.wa
    end

    mg = fid.modified_gravity
    mg_dict = Dict{String, Any}("Xi_0" => mg.Ξ₀, "Xi_n" => mg.Ξₙ)

    pop = fid.population
    pop_dict = Dict{String, Any}(
        "family" => string(pop.family == MadauDickinson ? "madau_dickinson" : "power_law"),
        "z_min" => pop.z_min,
        "z_max" => pop.z_max,
        "num_interp" => pop.num_interp,
        "time_delay_model" => pop.time_delay_model === nothing ? "none" : pop.time_delay_model,
    )
    if pop.family == MadauDickinson
        pop_dict["gamma"] = pop.γ
        pop_dict["kappa"] = pop.κ
        pop_dict["z_peak"] = pop.zpeak
    else
        pop_dict["lamb"] = pop.Λ
    end

    obs = fid.observation
    obs_dict = Dict{String, Any}(
        "local_merger_rate" => obs.local_merger_rate,
        "observation_time_yr" => obs.observation_time_yr,
    )

    doc = Dict{String, Any}(
        "cosmology" => cosmo_dict,
        "modified_gravity" => mg_dict,
        "population" => pop_dict,
        "observation" => obs_dict,
    )
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return nothing
end
