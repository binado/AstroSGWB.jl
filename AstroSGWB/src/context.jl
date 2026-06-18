"""
    build_model_context(problem, C, grid, detectors, observation_time_yr, local_merger_rate; z_grid) -> ModelContext

Build the `Λ`-independent [`ModelContext`](@ref) for `problem` at cosmology family `C`.

`grid` is the catalog [`FrequencyGrid`](@ref) (supplies the frequency axis and analysis-band
mask). `detectors` must contain at least two [`Detector`](@ref)s; the network effective PSD
and Gaussian bin scales are computed from tabulated PSDs and overlap-reduction functions.

The proposal caches (`proposal_prior`, the per-component `proposal_log_prob`, `dl_fid_sq`)
are recomputed at the fiducial cosmology `cosmology(C, Λ_fid)`, so stale on-disk values are
never trusted. The raw catalog `fluxes` are used directly (no `(D_L/D_gw)²` pre-scaling).
"""
function build_model_context(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        grid::FrequencyGrid,
        detectors::AbstractVector{D},
        observation_time_yr::Real,
        local_merger_rate::Real;
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
)::ModelContext where {C <: AbstractCosmology, D <: Detector}
    length(detectors) < 2 && throw(
        ArgumentError(
        "build_model_context: at least two detectors are required to build effective_psd and sgwb_scale",
    ),
    )
    C ∈ SUPPORTED_COSMOLOGIES || throw(
        ArgumentError(
        "unsupported cosmology type $(C); supported: $(join(SUPPORTED_COSMOLOGIES, ", "))",
    ),
    )

    Λ_fid = problem.fiducial_hyperparameters
    z = problem.samples.redshift
    n_samp = length(z)

    all_freq = frequencies(grid)
    mask = in_band_mask(grid)
    n_freq_full = length(all_freq)

    size(problem.fluxes, 2) == n_samp || throw(
        ArgumentError(
        "fluxes column count ($(size(problem.fluxes, 2))) does not match sample count ($n_samp)",
    ),
    )
    size(problem.fluxes, 1) == n_freq_full || throw(
        ArgumentError(
        "fluxes row count ($(size(problem.fluxes, 1))) does not match frequency grid length ($n_freq_full)",
    ),
    )

    obs_yr = Float64(observation_time_yr)
    det_vec = Vector{Detector}(collect(detectors))
    observation = build_observation_context(all_freq, det_vec, mask, obs_yr)

    c_fid = cosmology(C, Λ_fid)
    # Per-sample squared EM luminosity distance at the fiducial cosmology. The (Ξ₀, Ξₙ)
    # propagation factor is applied live in the importance weights as `dl_fid_sq / (D_L,θ² · Ξ_θ²)`.
    dl_fid_sq = luminosity_distance.(z, c_fid) .^ 2

    redshift_grid = collect(Float64, z_grid)
    interp = SampleInterpolant(z, redshift_grid)
    local_rate = Float64(local_merger_rate)

    # Fiducial proposal prior and its per-component log-densities, computed with the same
    # interpolant the hot path uses for the target redshift logpdf, so the redshift
    # log-ratio at Λ_fid is exactly zero (bitwise-identical arithmetic on both sides).
    cosmology_cache_fid = CosmologyCache(c_fid, redshift_grid)
    proposal_prior = single_event_prior(
        problem.population_model, cosmology_cache_fid, Λ_fid)
    samples = _with_redshift_interpolant(problem.samples, interp)
    proposal_log_prob = component_logpdfs(proposal_prior, samples)

    return ModelContext(
        proposal_prior,
        proposal_log_prob,
        dl_fid_sq,
        redshift_grid,
        interp,
        observation,
        local_rate
    )
end
