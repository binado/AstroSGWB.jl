export source_frame_distribution,
       unnormalized_merger_rate_density,
       redshift_logpdf_unnormalized,
       redshift_logpdf_normalized,
       redshift_density,
       integrated_merger_rate,
       redshift_logpdf_eltype,
       _normalized_log_density,
       DEFAULT_Z_GRID

"""
    DEFAULT_Z_GRID

Default redshift integration grid: 256 uniformly-spaced points on [1e-3, 20].
"""
const DEFAULT_Z_GRID = collect(LinRange(1e-3, 20.0, 256))

"""
    source_frame_distribution(sf, z, Λ) -> Real

Source-frame merger-rate density ``ψ(z)`` for source-frame model `sf` and
hyperparameters `Λ`.

CONTRACT: implementations must be normalized so that ``ψ(0) = 1``. The local
merger rate then sets the absolute scale through [`integrated_merger_rate`](@ref).
Concrete models live in `distance/source_frame/`.
"""
function source_frame_distribution end

"""
    unnormalized_merger_rate_density(z, dvc, sf, Λ) -> Real

Detector-frame merger-rate density ``dN/dz = 4π · dvc · ψ(z) / (1 + z)``, where
`dvc` is the differential comoving volume ``dV_c/dz`` evaluated at `z`. This is
the only cosmology-dependent input and is passed in as a plain number, keeping
this kernel decoupled from any cosmology implementation.
"""
function unnormalized_merger_rate_density(z::Real, dvc::Real, sf, Λ)
    return 4π * dvc * source_frame_distribution(sf, z, Λ) / (1 + z)
end

"""
    redshift_logpdf_unnormalized(z, dvc, sf, Λ) -> Real

Log of the unnormalized detector-frame redshift density at `z`
(`log` of [`unnormalized_merger_rate_density`](@ref)).
"""
function redshift_logpdf_unnormalized(z::Real, dvc::Real, sf, Λ)
    return log(unnormalized_merger_rate_density(z, dvc, sf, Λ))
end

"""
    redshift_logpdf_normalized(z, dvc, sf, Λ, norm) -> Real

Normalized log-density `redshift_logpdf_unnormalized(z, dvc, sf, Λ) - log(norm)`,
where `norm = ∫ dN/dz dz` over the integration range (see
[`integrated_merger_rate`](@ref) / `normalizer`). `norm` is supplied explicitly so
it is computed once rather than per call.
"""
function redshift_logpdf_normalized(z::Real, dvc::Real, sf, Λ, norm::Real)
    return redshift_logpdf_unnormalized(z, dvc, sf, Λ) - log(norm)
end

"""
    redshift_density(z_grid, dvc_grid, sf, Λ) -> CumulativeIntegral1D

Detector-frame `dN/dz` tabulated on `z_grid` from the matching differential
comoving volume values `dvc_grid`, wrapped in a [`CumulativeIntegral1D`](@ref) so
its cumulative table backs both normalization and inverse-CDF sampling. The
density is a `CumulativeIntegral1D` directly; there is no wrapper type.
"""
function redshift_density(
        z_grid::AbstractVector{<:Real},
        dvc_grid::AbstractVector{<:Real},
        sf,
        Λ
)
    vals = map(
        (z, dvc) -> unnormalized_merger_rate_density(z, dvc, sf, Λ),
        z_grid, dvc_grid)
    return CumulativeIntegral1D(z_grid, vals)
end

"""
    integrated_merger_rate(dN_dz::CumulativeIntegral1D, local_rate) -> Real

Physical detector-frame merger rate in **events/sec**:

``\\dot N = 10^{-9} · R_0 · ∫ dN/dz\\, dz / T_{yr}``,

where `R_0 = local_rate` is the local merger rate in ``\\mathrm{Gpc^{-3}\\,yr^{-1}}``,
the ``10^{-9}`` factor converts ``\\mathrm{Mpc^{-3}} → \\mathrm{Gpc^{-3}}`` against the
``\\mathrm{Mpc^3}`` redshift integral, and ``T_{yr}`` is the Julian year in seconds.
The bare redshift integral used for pdf normalization is `normalizer(dN_dz)`.
"""
function integrated_merger_rate(dN_dz::CumulativeIntegral1D, local_rate::Real)
    return 1e-9 * local_rate * normalizer(dN_dz) / JULIAN_YEAR_SEC
end

"""
    integrated_merger_rate(z_grid, dvc_grid, sf, Λ, local_rate) -> Real

Convenience form that builds the density on the grid via [`redshift_density`](@ref)
and returns the physical merger rate.
"""
function integrated_merger_rate(
        z_grid::AbstractVector{<:Real},
        dvc_grid::AbstractVector{<:Real},
        sf,
        Λ,
        local_rate::Real
)
    return integrated_merger_rate(redshift_density(z_grid, dvc_grid, sf, Λ), local_rate)
end

@inline function _normalized_log_density(pdf_at_value, norm, tiny)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

"""
    redshift_logpdf_eltype(dN_dz::CumulativeIntegral1D) -> Type

Element type of the normalized redshift log-density backed by `dN_dz`. Useful for
preallocating output vectors that promote with the redshift contribution (for
example `ForwardDiff.Dual` when `dN_dz` was built under AD).
"""
function redshift_logpdf_eltype(dN_dz::CumulativeIntegral1D)
    return promote_type(eltype(dN_dz.y), typeof(normalizer(dN_dz)))
end
