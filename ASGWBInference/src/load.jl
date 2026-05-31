"""
    load_problem_context(catalog_path, model_path, detectors, registry; local_merger_rate, observation_time_yr)
        -> (; problem, cosmology_type, ctx)

Orchestrate the full load-time flow for importance-sampling inference:

1. read the waveform catalog ([`load_catalog`](@ref)) and verify its model fingerprint,
2. read `model.toml` ([`load_model_toml`](@ref)) → cosmology family `C`, population model, and
   fiducial hyperparameters,
3. restructure the catalog samples into the BNS proposal layout
   ([`bns_samples_from_catalog`](@ref)),
4. construct the pure [`ImportanceSamplingProblem`](@ref), and
5. build the `Λ`-independent [`ModelContext`](@ref) ([`build_model_context`](@ref)).

`registry` maps the `[model].population` name to a concrete `PopulationModel`. `detectors`
must contain at least two `Detector`s.
"""
function load_problem_context(
        catalog_path::AbstractString,
        model_path::AbstractString,
        detectors::AbstractVector{<:Detector},
        registry::AbstractDict;
        local_merger_rate::Real,
        observation_time_yr::Real
)
    loaded = load_catalog(catalog_path)
    verify_model_fingerprint(loaded, model_path)
    C, pop, Λ = load_model_toml(model_path, registry)
    catalog = loaded.catalog
    samples = bns_samples_from_catalog(catalog.samples)
    problem = ImportanceSamplingProblem(pop, catalog.fluxes, samples, Λ)
    ctx = build_model_context(
        problem,
        C,
        loaded.metadata.grid,
        detectors,
        observation_time_yr,
        local_merger_rate
    )
    return (; problem = problem, cosmology_type = C, ctx = ctx)
end
