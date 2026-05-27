using HDF5

const BUNDLE_COMMAND_ATTR = "command"
const BUNDLE_GIT_REVISION_ATTR = "git_revision"

"""
    load_problem(bundle_path, cosmology_path, detectors) -> ImportanceSamplingProblem

Load a [`WaveformCatalog`](@ref) bundle and a `cosmology.toml` fiducial file, verify
their cosmology fingerprint matches, and build an in-memory
[`ImportanceSamplingProblem`](@ref) ready for MCMC.

`detectors` must contain at least two [`Detector`](@ref) values; the network effective
PSD and SGWB scale are computed from tabulated PSDs and overlap-reduction functions.

The fiducial spectral density stored in [`ObservationConfig`](@ref) is computed via
[`fiducial_spectral_density`](@ref) after the problem is assembled, so it always
reflects the current Julia forward model rather than any stale on-disk value.
"""
function load_problem(
        bundle_path::AbstractString,
        cosmology_path::AbstractString,
        detectors::AbstractVector{D}
)::ImportanceSamplingProblem where {D <: Detector}
    isempty(detectors) && throw(ArgumentError("load_problem: detectors must be non-empty"))
    length(detectors) < 2 && throw(
        ArgumentError(
            "load_problem: at least two detectors are required to build effective_psd and sgwb_scale",
        ),
    )

    catalog = load_bundle(bundle_path)
    fid = load_cosmology_toml(cosmology_path)
    verify_cosmology_fingerprint(catalog, cosmology_path)

    cache_C = cosmology_type(fid)
    cache_C ∈ SUPPORTED_COSMOLOGIES || throw(
        ArgumentError(
            "cosmology.toml specifies unsupported cosmology type $(cache_C); " *
            "supported: $(join(SUPPORTED_COSMOLOGIES, ", "))",
        ),
    )

    meta = catalog.metadata
    grid = meta.grid
    all_freq = frequencies(grid)
    mask = in_band_mask(grid)
    n_freq_full = length(all_freq)

    obs_yr = fid.observation.observation_time_yr
    obs_sec = obs_yr * 365.25 * 24 * 3600.0

    det_vec = Vector{Detector}(collect(detectors))
    placeholder_fid_sd = zeros(Float64, n_freq_full)
    observation = build_observation_config(
        collect(Float64, all_freq),
        det_vec,
        mask,
        placeholder_fid_sd,
        obs_sec,
        obs_yr
    )

    z = catalog.samples.redshift
    n_samp = length(z)
    size(catalog.fluxes, 2) == n_samp || throw(
        ArgumentError(
            "bundle fluxes column count ($(size(catalog.fluxes, 2))) " *
            "does not match sample count ($n_samp)",
        ),
    )
    size(catalog.fluxes, 1) == n_freq_full || throw(
        ArgumentError(
            "bundle fluxes row count ($(size(catalog.fluxes, 1))) " *
            "does not match frequency grid length ($n_freq_full)",
        ),
    )

    samples = _catalog_samples_to_bns_soa(catalog.samples)
    cached_flux_over_dgw2 = reconstruct_cached_flux_over_dgw2(catalog.fluxes, z, fid)
    dgw_fid_sq = reconstruct_dgw_fid_sq(z, fid)

    intrinsic_site_order = _bns_intrinsic_site_order(catalog.samples)
    lp = reconstruct_proposal_log_prob(samples, redshift_prior_spec(fid), fid)
    intrinsic_vector = _build_intrinsic_vector(catalog.samples, intrinsic_site_order)

    proposal = ProposalData(
        intrinsic_site_order,
        samples,
        lp,
        intrinsic_vector,
        cached_flux_over_dgw2,
        dgw_fid_sq
    )

    spec = redshift_prior_spec(fid)
    ri = fiducial_redshift_integral(fid, spec)

    p = importance_sampling_problem(
        proposal,
        observation,
        spec,
        fid.observation.local_merger_rate,
        ri,
        fid
    )

    fs = try
        fiducial_spectral_density(p)
    catch err
        throw(
            ArgumentError(
                "fiducial_spectral_density recomputation failed after loading bundle; " *
                "Underlying error: " * sprint(showerror, err),
            ),
        )
    end
    observation2 = ObservationConfig(
        p.observation.frequencies,
        p.observation.effective_psd,
        p.observation.sgwb_scale,
        p.observation.in_band_mask,
        fs,
        p.observation.observation_time_sec,
        p.observation.observation_time_yr
    )
    return importance_sampling_problem(
        p.proposal,
        observation2,
        p.redshift_prior_spec,
        p.local_merger_rate,
        p.redshift_integral_fiducial,
        p.fiducial_parameters
    )
end

const _BNS_INTRINSIC_KEYS = (
    :mass_1_source, :mass_2_source, :redshift,
    :chi_1, :chi_2, :lambda_1, :lambda_2
)

function _bns_intrinsic_site_order(samples::NamedTuple)
    for k in _BNS_INTRINSIC_KEYS
        haskey(samples, k) || throw(
            ArgumentError("bundle samples missing required BNS column $(repr(k))"),
        )
    end
    return [String(k) for k in _BNS_INTRINSIC_KEYS]
end

function _catalog_samples_to_bns_soa(samples::NamedTuple)::FullBNSSamplesSoA
    for k in _BNS_INTRINSIC_KEYS
        haskey(samples, k) || throw(
            ArgumentError("bundle samples missing required BNS column $(repr(k))"),
        )
    end
    return (
        mass = stack_source_masses(samples.mass_1_source, samples.mass_2_source),
        redshift = copy(samples.redshift),
        χ₁ = copy(samples.chi_1),
        χ₂ = copy(samples.chi_2),
        Λ₁ = copy(samples.lambda_1),
        Λ₂ = copy(samples.lambda_2)
    )
end

function _build_intrinsic_vector(
        samples::NamedTuple,
        site_order::Vector{String}
)::Matrix{Float64}
    n = length(samples.redshift)
    ncols = length(site_order)
    mat = Matrix{Float64}(undef, n, ncols)
    for (j, key) in enumerate(site_order)
        k = Symbol(key)
        haskey(samples, k) || throw(ArgumentError("samples missing column $(repr(key))"))
        col = samples[k]
        for i in 1:n
            mat[i, j] = col[i]
        end
    end
    return mat
end
