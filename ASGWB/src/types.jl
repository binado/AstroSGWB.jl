"""
    ObservationConfig

Detector-side SGWB observation layout: frequency grid, per-bin effective strain PSD
amplitude from the detector network (ORFs and tabulated PSDs; square matches network
variance), Gaussian bin scales for the likelihood, analysis band mask, and observation
time metadata.
"""
struct ObservationConfig
    frequencies::Vector{Float64}
    effective_psd::Vector{Float64}
    sgwb_scale::Vector{Float64}
    in_band_mask::BitVector
    fiducial_spectral_density::Vector{Float64}
    observation_time_sec::Float64
    observation_time_yr::Float64
    sgwb_scale_in_band::Vector{Float64}
    fiducial_spectral_density_in_band::Vector{Float64}
end

function ObservationConfig(
        frequencies::Vector{Float64},
        effective_psd::Vector{Float64},
        sgwb_scale::Vector{Float64},
        in_band_mask::BitVector,
        fiducial_spectral_density::Vector{Float64},
        observation_time_sec::Float64,
        observation_time_yr::Float64
)
    return ObservationConfig(
        frequencies,
        effective_psd,
        sgwb_scale,
        in_band_mask,
        fiducial_spectral_density,
        observation_time_sec,
        observation_time_yr,
        sgwb_scale[in_band_mask],
        fiducial_spectral_density[in_band_mask]
    )
end

"""HDF5 `proposal_samples` group attribute naming the compact-object proposal class."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR = "source_type"

"""`proposal_samples` / [`PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`](@ref) value for BNS importance samples."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_BNS = "BNS"
