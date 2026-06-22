"""
    ObservationContext

Detector-side SGWB observation layout: frequency grid, per-bin effective strain PSD
amplitude from the detector network (ORFs and tabulated PSDs; square matches network
variance), Gaussian bin scales for the likelihood, analysis band mask, and observation
time metadata (`observation_time`, in years). The in-band Gaussian scale is precomputed for the likelihood hot path.

The observed spectral density is intentionally *not* part of this object; callers pass it
explicitly to [`loglikelihood`](@ref) or let [`build_turing_model`](@ref) synthesize it
via [`fiducial_spectral_density`](@ref) when `observed` is omitted.
"""
struct ObservationContext{
    F <: AbstractVector{<:Real},
    E <: AbstractVector{<:Real},
    S <: AbstractVector{<:Real},
    M <: AbstractVector{Bool},
    T <: Real,
    B <: AbstractVector{<:Real}
}
    frequencies::F
    effective_psd::E
    sgwb_scale::S
    in_band_mask::M
    observation_time::T
    sgwb_scale_in_band::B
end

function ObservationContext(
        frequencies::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        sgwb_scale::AbstractVector{<:Real},
        in_band_mask::AbstractVector{Bool},
        observation_time::Real
)
    return ObservationContext(
        frequencies,
        effective_psd,
        sgwb_scale,
        in_band_mask,
        observation_time,
        sgwb_scale[in_band_mask]
    )
end
