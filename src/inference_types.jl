using Distributions: Distribution

"""Ordering of parameters in the flat constrained `NamedTuple` used by Bijectors / HMC."""
const DEFAULT_PARAMETER_ORDER = (:H0, :Omega_m, :chi0, :chin, :gamma, :kappa, :z_peak)

abstract type PopulationParameters end

struct MadauDickinsonParameters{Tγ<:Real,Tκ<:Real,Tz<:Real} <: PopulationParameters
    gamma::Tγ
    kappa::Tκ
    z_peak::Tz
end

struct PowerLawRedshiftParameters{Tλ<:Real} <: PopulationParameters
    lamb::Tλ
end

struct CosmologicalParameters{TH0<:Real,TΩ<:Real}
    H0::TH0
    Omega_m::TΩ
end

struct ModifiedPropagationParameters{Tχ0<:Real,Tχn<:Real}
    chi0::Tχ0
    chin::Tχn
end

"""
    HyperParameters

Nested inference state: cosmology, modified GW propagation, and a
[`PopulationParameters`](@ref) branch (Madau–Dickinson or power-law redshift population).

For HMC / [`build_prior_distribution`](@ref), only [`MadauDickinsonParameters`](@ref) is
supported today (seven stochastic parameters). Use [`as_flat_constrained`](@ref) for
the Bijectors bridge; [`PowerLawRedshiftParameters`](@ref) is for redshift-grid tests
that do not use the seven-parameter product prior.
"""
struct HyperParameters{
    C<:CosmologicalParameters,
    P<:ModifiedPropagationParameters,
    Pop<:PopulationParameters,
}
    cosmological::C
    propagation::P
    population::Pop
end

function _getnt(nt::NamedTuple, key::Symbol, default)
    haskey(nt, key) ? nt[key] : default
end

function HyperParameters(;
    H0::Real,
    Omega_m::Real,
    chi0::Real=1.0,
    chin::Real=0.0,
    gamma=nothing,
    kappa=nothing,
    z_peak=nothing,
    lamb=nothing,
)
    if !isnothing(lamb) && isnothing(gamma)
        return HyperParameters((
            H0=Float64(H0),
            Omega_m=Float64(Omega_m),
            chi0=Float64(chi0),
            chin=Float64(chin),
            lamb=Float64(lamb),
        ))
    end
    return HyperParameters((
        H0=Float64(H0),
        Omega_m=Float64(Omega_m),
        chi0=Float64(chi0),
        chin=Float64(chin),
        gamma=Float64(something(gamma)),
        kappa=Float64(something(kappa)),
        z_peak=Float64(something(z_peak)),
    ))
end

function HyperParameters(nt::NamedTuple)
    c = CosmologicalParameters(nt.H0, nt.Omega_m)
    p = ModifiedPropagationParameters(_getnt(nt, :chi0, 1.0), _getnt(nt, :chin, 0.0))
    pop = if haskey(nt, :lamb) && !haskey(nt, :gamma)
        PowerLawRedshiftParameters(nt.lamb)
    else
        MadauDickinsonParameters(nt.gamma, nt.kappa, nt.z_peak)
    end
    return HyperParameters(c, p, pop)
end

function validate_redshift_spec_population(spec::RedshiftPriorSpec, pop::PopulationParameters)
    if spec.family == MadauDickinson && !(pop isa MadauDickinsonParameters)
        throw(ArgumentError("MadauDickinson redshift prior requires MadauDickinsonParameters"))
    end
    if spec.family == PowerLaw && !(pop isa PowerLawRedshiftParameters)
        throw(ArgumentError("PowerLaw redshift prior requires PowerLawRedshiftParameters"))
    end
    return nothing
end

"""
    as_flat_constrained(h::HyperParameters)

Build the flat constrained `NamedTuple` in [`DEFAULT_PARAMETER_ORDER`](@ref) for
`product_distribution` / Bijectors. Requires [`MadauDickinsonParameters`](@ref).
"""
function as_flat_constrained(h::HyperParameters)
    pop = h.population
    pop isa MadauDickinsonParameters || throw(
        ArgumentError(
            "as_flat_constrained requires MadauDickinsonParameters; use PowerLaw only for redshift-grid code paths without the seven-parameter prior",
        ),
    )
    return NamedTuple{DEFAULT_PARAMETER_ORDER}((
        h.cosmological.H0,
        h.cosmological.Omega_m,
        h.propagation.chi0,
        h.propagation.chin,
        pop.gamma,
        pop.kappa,
        pop.z_peak,
    ))
end

struct InferencePriors
    H0::Distribution
    Omega_m::Distribution
    chi0::Distribution
    chin::Distribution
    gamma::Distribution
    kappa::Distribution
    z_peak::Distribution
end

abstract type ProposalSampleBundle end

"""
    FullBNSSamplesSoA

Struct-of-arrays proposal-sample container matching the NamedTuple returned by
`rand(prior, n)` when `prior = intrinsic_prior(FullBNS(), bundle)`:

- `mass::Matrix{Float64}` of size `(2, n)`; row 1 is `mass_1_source`, row 2 is `mass_2_source`.
- `redshift`, `chi_1`, `chi_2`, `lambda_1`, `lambda_2` are `Vector{Float64}` of length `n`.

Matches `keys(prior.dists)` for the full-BNS intrinsic prior.
"""
const FullBNSSamplesSoA = @NamedTuple{
    mass::Matrix{Float64},
    redshift::Vector{Float64},
    chi_1::Vector{Float64},
    chi_2::Vector{Float64},
    lambda_1::Vector{Float64},
    lambda_2::Vector{Float64},
}

"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into the `2 × n` matrix expected by
[`FullBNSSamplesSoA`](@ref)`.mass` (row 1 = `mass_1_source`, row 2 = `mass_2_source`).
"""
function stack_source_masses(
    mass_1_source::AbstractVector{<:Real},
    mass_2_source::AbstractVector{<:Real},
)::Matrix{Float64}
    n = length(mass_1_source)
    length(mass_2_source) == n || throw(
        ArgumentError("mass_1_source and mass_2_source must have matching lengths"),
    )
    return permutedims(hcat(collect(Float64, mass_1_source), collect(Float64, mass_2_source)))
end

"""
    ProposalData

Proposal-sample bundle for the importance-sampling problem.

Matrix layouts:
- `intrinsic_vector` is `(n_samples, n_intrinsic)` (rows = samples, columns = intrinsic sites).
- `cached_flux_over_dgw2` is `(n_freq, n_samples)` (column-major friendly: each proposal
  sample is a contiguous column; `fluxes * weights` contracts to a per-frequency vector).
"""
struct ProposalData
    intrinsic_site_order::Vector{String}
    samples::FullBNSSamplesSoA
    log_prob::Vector{Float64}
    intrinsic_vector::Matrix{Float64}
    cached_flux_over_dgw2::Matrix{Float64}
    dgw_fid_sq::Vector{Float64}
end

"""
    ImportanceSamplingProblem

In-memory importance-sampling context. See [`importance_sampling_problem`](@ref) and
[`load_cache`](@ref). `fiducial_parameters` merges population scalars from HDF5
`hyperparameters` and `redshift_prior_spec` ([`ProposalFiducialParameters`](@ref)), not the live
[`HyperParameters`](@ref) state. `redshift_integral_fiducial` is carried for cache round-trip and
may differ from [`fiducial_redshift_integral`](@ref) when the file’s optional
`redshift_integral_fiducial` attribute overrides the recomputed value; likelihood evaluation uses
the integral implied by the live [`HyperParameters`](@ref), not this field.
"""
struct ImportanceSamplingProblem
    proposal::ProposalData
    observation::ObservationConfig
    redshift_prior_spec::RedshiftPriorSpec
    local_merger_rate::Float64
    redshift_integral_fiducial::Float64
    fiducial_parameters::ProposalFiducialParameters
    strategy::FullBNS
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.proposal.samples)

function _validate_strategy_bundle(strategy::FullBNS, proposal::ProposalData)
    proposal.samples isa FullBNSSamplesSoA || throw(
        ArgumentError("proposal samples must match the FullBNSSamplesSoA layout"),
    )
    return nothing
end

"""
    importance_sampling_problem(
        proposal, observation, redshift_prior_spec,
        local_merger_rate, fiducial_parameters,
    ) -> ImportanceSamplingProblem

Five-argument form: the stored fiducial integral is
[`fiducial_redshift_integral`](@ref) applied to `fiducial_parameters` and `redshift_prior_spec`.

    importance_sampling_problem(
        proposal, observation, redshift_prior_spec,
        local_merger_rate, redshift_integral_fiducial, fiducial_parameters,
    ) -> ImportanceSamplingProblem

Six-argument form: supply a precomputed fiducial redshift integral (e.g. read from a cache file).

Both forms validate [`IntrinsicPriorStrategy`](@ref) against the proposal sample bundle type.
"""
function importance_sampling_problem(
    proposal::ProposalData,
    observation::ObservationConfig,
    redshift_prior_spec::RedshiftPriorSpec,
    local_merger_rate::Real,
    redshift_integral_fiducial::Real,
    fiducial_parameters::ProposalFiducialParameters,
)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    _validate_strategy_bundle(strategy, proposal)
    return ImportanceSamplingProblem(
        proposal,
        observation,
        redshift_prior_spec,
        Float64(local_merger_rate),
        Float64(redshift_integral_fiducial),
        fiducial_parameters,
        strategy,
    )
end

"""
    ImportanceCache

Deprecated alias for [`ImportanceSamplingProblem`](@ref); use `ImportanceSamplingProblem` in new code.
"""
const ImportanceCache = ImportanceSamplingProblem
