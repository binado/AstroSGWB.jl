using SHA: sha256
using TOML

model_sha256_of_file(path::AbstractString)::String = bytes2hex(sha256(read(path)))

function _require_table(data::AbstractDict, key::AbstractString)
    value = get(data, key, nothing)
    value isa AbstractDict || throw(ArgumentError("model.toml requires [$key] table"))
    return value
end

"""
    read_cosmology(data) -> Type{<:AbstractCosmology}

Parse `[model].cosmology` from a TOML dict and return the corresponding
cosmology type.
"""
function read_cosmology(data::AbstractDict)
    model_table = _require_table(data, "model")
    name = get(model_table, "cosmology", nothing)
    name isa AbstractString ||
        throw(ArgumentError("model.toml [model] requires a string \"cosmology\" key"))
    return cosmology_type(name)
end

"""
    read_population(data, registry) -> PopulationModel

Parse the population model from a TOML dict.  `[model].population` names the
population; `registry` maps that name to a concrete [`PopulationModel`](@ref).
The framework owns no population types: callers supply the registry.
"""
function read_population(data::AbstractDict, registry::AbstractDict)
    model_table = _require_table(data, "model")
    name = get(model_table, "population", nothing)
    name isa AbstractString ||
        throw(ArgumentError("model.toml [model] requires a string \"population\" key"))
    haskey(registry, name) || throw(
        ArgumentError(
        "unknown population $(repr(name)); registered: $(sort!(collect(keys(registry))))",
    ),
    )
    return registry[name]
end

"""
    population_name(registry, pop) -> String

Reverse-lookup the registry name for `pop`.  Used for serialisation.
"""
function population_name(registry::AbstractDict, pop::PopulationModel)
    for (name, candidate) in registry
        candidate === pop && return String(name)
    end
    for (name, candidate) in registry
        typeof(candidate) === typeof(pop) && return String(name)
    end
    throw(ArgumentError("population $(typeof(pop)) is not present in the registry"))
end

"""
    read_parameters(data, C, pop) -> NamedTuple

Parse `[parameters]` from a TOML dict.  Keys are symbol names (`Ωm`, `γ`, …);
values are converted to `Float64`.  The returned NamedTuple is ordered by
`full_hyperparameters(C, pop)` and validated via `canonical_hyperparameters`.
"""
function read_parameters(data::AbstractDict, ::Type{C}, pop::M) where {
        C, M <: PopulationModel}
    params_table = _require_table(data, "parameters")
    order = full_hyperparameters(C, pop)
    raw = NamedTuple{order}(
        ntuple(Val(length(order))) do i
        k = order[i]
        v = get(params_table, String(k), nothing)
        v === nothing &&
            throw(ArgumentError("model.toml [parameters] missing key \"$(String(k))\""))
        Float64(v)
    end
    )
    return canonical_hyperparameters(order, raw; context = "model parameters")
end

"""
    dump_parameters(Λ) -> Dict{String,Any}

Serialise a hyperparameter NamedTuple to a string-keyed dict suitable for
`TOML.print`.  Keys are `string(sym)` (Julia symbol names).
"""
function dump_parameters(Λ::NamedTuple)
    return Dict{String, Any}(String(k) => v for (k, v) in pairs(Λ))
end

"""
    dump_model(C, population_name) -> Dict{String,Any}

Serialise the cosmology type and population name to a `[model]` section dict.
"""
function dump_model(::Type{C}, population_name::AbstractString) where {C <:
                                                                       AbstractCosmology}
    return Dict{String, Any}(
        "cosmology" => cosmology_config_name(C),
        "population" => String(population_name)
    )
end

"""
    load_model_toml(path, registry) -> (C::Type, pop::PopulationModel, Λ::NamedTuple)

Load a `model.toml` file and return the cosmology type, population model, and
canonical fiducial hyperparameters.  `registry` resolves `[model].population`.
"""
function load_model_toml(path::AbstractString, registry::AbstractDict)
    data = TOML.parsefile(path)
    C = read_cosmology(data)
    pop = read_population(data, registry)
    Λ = read_parameters(data, C, pop)
    return C, pop, Λ
end

"""
    save_model_toml(path, C, pop, Λ, registry)

Write cosmology type, population name, and hyperparameters to `path` as a
`model.toml` file.  The population name is resolved from `registry`.
"""
function save_model_toml(
        path::AbstractString,
        ::Type{C},
        pop::PopulationModel,
        Λ::NamedTuple,
        registry::AbstractDict
) where {C}
    data = Dict{String, Any}(
        "model" => dump_model(C, population_name(registry, pop)),
        "parameters" => dump_parameters(Λ)
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
    return nothing
end
