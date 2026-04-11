struct RedshiftPriorSpec
    family::String
    z_min::Float64
    z_max::Float64
    num_interp::Int
    time_delay_model::Union{String,Nothing}
end

abstract type IntrinsicPriorStrategy end
struct RedshiftOnly <: IntrinsicPriorStrategy end
struct FullBNS <: IntrinsicPriorStrategy end

struct ProposalData
    intrinsic_site_order::Vector{String}
    samples::Dict{String,Vector{Float64}}
    log_prob::Vector{Float64}
    intrinsic_vector::Matrix{Float64}
    cached_flux_over_dgw2::Matrix{Float64}
    dgw_fid_sq::Vector{Float64}
end

struct ObservationConfig
    frequencies::Vector{Float64}
    covariance::Vector{Float64}
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
    covariance::Vector{Float64},
    sgwb_scale::Vector{Float64},
    in_band_mask::BitVector,
    fiducial_spectral_density::Vector{Float64},
    observation_time_sec::Float64,
    observation_time_yr::Float64,
)
    return ObservationConfig(
        frequencies,
        covariance,
        sgwb_scale,
        in_band_mask,
        fiducial_spectral_density,
        observation_time_sec,
        observation_time_yr,
        sgwb_scale[in_band_mask],
        fiducial_spectral_density[in_band_mask],
    )
end

struct ImportanceSamplingProblem{S<:IntrinsicPriorStrategy}
    proposal::ProposalData
    observation::ObservationConfig
    redshift_prior_spec::RedshiftPriorSpec
    local_merger_rate::Float64
    redshift_integral_fiducial::Float64
    hyperparameters::Dict{String,Float64}
    strategy::S
end

redshift(problem::ImportanceSamplingProblem) = problem.proposal.samples["redshift"]

const ImportanceCache = ImportanceSamplingProblem
