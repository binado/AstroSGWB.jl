struct RedshiftGridBundle{TX<:AbstractVector,TY<:AbstractVector,T<:Real}
    x::TX
    pdf_unnorm::TY
    norm::T
end

function trapezoid_integral(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))
    length(x) >= 2 || throw(ArgumentError("at least two grid points are required"))
    @views sum((y[1:end-1] .+ y[2:end]) .* (x[2:end] .- x[1:end-1])) / 2
end

function _interp_linear(
    x_grid::AbstractVector{<:Real},
    y_grid::AbstractVector{<:Real},
    x::Real;
    left::Real=0.0,
    right::Real=0.0,
)
    length(x_grid) == length(y_grid) || throw(ArgumentError("grid and values must align"))
    T = promote_type(eltype(y_grid), typeof(left), typeof(right))
    x < x_grid[1] && return convert(T, left)
    x >= x_grid[end] && return y_grid[end]

    idx = searchsortedlast(x_grid, x)
    idx == 0 && return convert(T, left)
    x0, y0 = x_grid[idx], y_grid[idx]
    (x == x0 || idx == length(x_grid)) && return y0
    x1, y1 = x_grid[idx+1], y_grid[idx+1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
end

function detector_frame_merger_rate_density(
    z::Real,
    differential_comoving_volume::Real,
    source_frame_distribution::Real,
)
    return 4π * differential_comoving_volume * source_frame_distribution / (1 + z)
end

function expected_number_of_events(
    local_merger_rate_gpc3_yr::Real,
    redshift_integral_mpc3::Real,
    observation_time_yr::Real,
)
    return 1e-9 * local_merger_rate_gpc3_yr * redshift_integral_mpc3 * observation_time_yr
end

"""
    merger_rate_per_sec(
        bundle, local_merger_rate_gpc3_yr, observation_time_yr, observation_time_sec,
    ) -> Float64

Detector-frame merger rate in events/sec: `expected_number_of_events(local_rate, bundle.norm,
observation_time_yr) / observation_time_sec`. `observation_time_yr` sets the events count,
`observation_time_sec` converts to per-second units; they are taken independently rather than
assuming a fixed seconds-per-year so the cache's stored pair of times round-trips exactly.
"""
function merger_rate_per_sec(
    bundle::RedshiftGridBundle,
    local_merger_rate_gpc3_yr::Real,
    observation_time_yr::Real,
    observation_time_sec::Real,
)
    n_events = expected_number_of_events(
        local_merger_rate_gpc3_yr, bundle.norm, observation_time_yr,
    )
    return n_events / observation_time_sec
end

function madau_dickinson_source_frame_distribution(
    z::Real;
    gamma::Real,
    kappa::Real,
    z_peak::Real,
)
    one_plus_z = 1 + z
    return (
        (one_plus_z^gamma) / (1 + (one_plus_z / (1 + z_peak))^kappa)
    ) * (1 + (1 + z_peak)^(-kappa))
end

power_law_source_frame_distribution(z::Real; lamb::Real) = (1 + z)^lamb

function _build_redshift_grid(
    source_frame_fn,
    H0::Real,
    Omega_m::Real,
    z_min::Real,
    z_max::Real,
    num_interp::Integer,
)
    z_grid = collect(LinRange(Float64(z_min), Float64(z_max), Int(num_interp)))
    dvc_dz = differential_comoving_volume.(z_grid, H0, Omega_m)
    source_frame = source_frame_fn.(z_grid)
    pdf_unnorm = detector_frame_merger_rate_density.(z_grid, dvc_dz, source_frame)
    return z_grid, pdf_unnorm
end

function _source_frame_fn(spec::RedshiftPriorSpec, pop::MadauDickinsonParameters)
    spec.family == MadauDickinson || throw(
        ArgumentError("MadauDickinson population requires MadauDickinson redshift prior family"),
    )
    return z -> madau_dickinson_source_frame_distribution(
        z; gamma=pop.gamma, kappa=pop.kappa, z_peak=pop.z_peak,
    )
end

function _source_frame_fn(spec::RedshiftPriorSpec, pop::PowerLawRedshiftParameters)
    spec.family == PowerLaw || throw(
        ArgumentError("PowerLawRedshiftParameters require PowerLaw redshift prior family"),
    )
    return z -> power_law_source_frame_distribution(z; lamb=pop.lamb)
end

function build_redshift_grid_bundle(h::HyperParameters, spec::RedshiftPriorSpec)
    isnothing(spec.time_delay_model) || throw(
        ArgumentError("time-delay redshift models are not supported in the Julia v0 port"),
    )
    validate_redshift_spec_population(spec, h.population)
    sfn = _source_frame_fn(spec, h.population)
    z_grid, pdf_unnorm = _build_redshift_grid(
        sfn,
        h.cosmological.H0,
        h.cosmological.Omega_m,
        spec.z_min,
        spec.z_max,
        spec.num_interp,
    )
    return RedshiftGridBundle(z_grid, pdf_unnorm, trapezoid_integral(z_grid, pdf_unnorm))
end

function log_prob_from_bundle(value::Real, bundle::RedshiftGridBundle)
    T = promote_type(eltype(bundle.pdf_unnorm), typeof(bundle.norm))
    tiny = floatmin(T)
    pdf_at_value = _interp_linear(bundle.x, bundle.pdf_unnorm, value; left=0.0, right=0.0)
    return log(max(pdf_at_value / max(bundle.norm, tiny), tiny))
end
