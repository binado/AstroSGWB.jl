"""
    reconstruct_dgw_fid_sq(z, cosmology_type, Λ) -> Vector{Float64}

Per-sample squared gravitational-wave luminosity distance at fiducial cosmology
and propagation, from source-frame redshifts `z`.
"""
function reconstruct_dgw_fid_sq(
        z::AbstractVector{<:Real},
        ::Type{C},
        Λ::NamedTuple
)::Vector{Float64} where {C <: AbstractCosmology}
    c = cosmology(C, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = _dgw_from_cached_dl.(z, d_l, Ref(c))
    return Float64.(d_gw .^ 2)
end

"""
    reconstruct_cached_flux_over_dgw2(cached_flux, z, cosmology_type, Λ) -> Matrix{Float64}

Apply the squared ratio of electromagnetic to gravitational-wave luminosity distance
sample-wise. Inputs and outputs use the `(n_freq, n_samples)` layout (column-major
friendly; each proposal sample is a contiguous column).
"""
function reconstruct_cached_flux_over_dgw2(
        cached_flux::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real},
        ::Type{C},
        Λ::NamedTuple
)::Matrix{Float64} where {C <: AbstractCosmology}
    size(cached_flux, 2) == length(z) ||
        throw(ArgumentError("cached_flux column count must match redshift sample count"))
    c = cosmology(C, Λ)
    d_l = luminosity_distance.(z, c)
    d_gw = _dgw_from_cached_dl.(z, d_l, Ref(c))
    scale_row = reshape(Float64.((d_l ./ d_gw) .^ 2), 1, :)
    return Matrix{Float64}(cached_flux) .* scale_row
end

"""
    reconstruct_proposal_log_prob(samples, cosmology_type, population, Λ) -> Vector{Float64}

Proposal log-density per sample: full single-event prior log-density on
[`FullBNSSamplesSoA`](@ref) under the given hyperparameters.
"""
function reconstruct_proposal_log_prob(
        samples::FullBNSSamplesSoA,
        ::Type{C},
        population::M,
        Λ::NamedTuple
)::Vector{Float64} where {C <: AbstractCosmology, M <: PopulationModel}
    c = cosmology(C, Λ)
    prior = single_event_prior(population, c, Λ)
    return batched_logpdf(prior, samples)
end
