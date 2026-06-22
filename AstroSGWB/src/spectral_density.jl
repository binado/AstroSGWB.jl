using ForwardDiff

"""
    spectral_density(fluxes, merger_rate_per_sec; weights=nothing) -> Vector

Collapse per-sample flux contributions into a spectral density vector.

`fluxes` is a `(nfreq, nsamples)` matrix (column-major friendly). When `weights`
is `nothing`, samples are averaged uniformly: `mean_flux = sum(fluxes; dims=2) / nsamples`.
When `weights` is supplied, the contraction is `fluxes * weights / nsamples`
(no normalization of `weights`). The `0.4 = 2/5` prefactor captures the
average over the inclination angle.
"""
function spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
)
    return _spectral_density(fluxes, merger_rate_per_sec, weights)
end

function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        ::Nothing
)
    nsamples = size(fluxes, 2)
    mean_flux = vec(sum(fluxes; dims = 2)) ./ nsamples
    return 0.4 .* merger_rate_per_sec .* mean_flux
end

function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:Real}
)
    nsamples = size(fluxes, 2)
    mean_flux = (fluxes * weights) ./ nsamples
    return 0.4 .* merger_rate_per_sec .* mean_flux
end

# Avoid `Matrix{Float64} * Vector{Dual}` here: on realistic caches the generic
# Dual matvec dominated ForwardDiff/Turing gradient profiles. Splitting primal
# values and partials lets BLAS handle the two dense contractions (see
# `_spectral_density_forwarddiff`). The rate may itself be a same-tag Dual.
function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}}
) where {Tag, V, N}
    rate_value, rate_partials = _rate_value_partials(
        merger_rate_per_sec, Tag, V, Val(N))
    return _spectral_density_forwarddiff(fluxes, rate_value, rate_partials, weights)
end

# Extract `(value, partials)` from a rate that is either a plain `Real` (zero
# partials) or a `Dual` whose tag/lane count match the weights'. Mismatched-tag
# Duals are not supported in this dispatch family.
function _rate_value_partials(x::Real, ::Type, ::Type{V}, ::Val{N}) where {V, N}
    (V(x), ntuple(_ -> zero(V), Val(N)))
end

function _rate_value_partials(
        x::ForwardDiff.Dual{Tag, V, N}, ::Type{Tag}, ::Type{V}, ::Val{N}
) where {Tag, V, N}
    (ForwardDiff.value(x), Tuple(ForwardDiff.partials(x)))
end

# See comment above `_spectral_density` for the Dual-weighted dispatch rationale.
function _spectral_density_forwarddiff(
        fluxes::AbstractMatrix{<:Real},
        rate_value::V,
        rate_partials::NTuple{N, V},
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}}
) where {Tag, V, N}
    nsamples = size(fluxes, 2)
    length(weights) == nsamples ||
        throw(DimensionMismatch("weight length must match flux sample dimension"))

    # Pack value + partials into one contiguous `(nsamples, N+1)` buffer so a single
    # gemm `fluxes * weight_block` yields the primal sum (column 1) and every partial
    # sum (columns 2:N+1) at once, instead of a separate gemv + gemm. Built with
    # broadcasts + `hcat` (no scalar indexing) so the pack, gemm, and recombine all
    # dispatch unchanged on device arrays.
    val = ForwardDiff.value.(weights)
    pvecs = ForwardDiff.partials.(weights)
    parts = ntuple(j -> getindex.(pvecs, j), Val(N))
    weight_block = hcat(val, parts...)

    sums = fluxes * weight_block
    scale = V(0.4) / V(nsamples)
    primal = @view sums[:, 1]
    value = scale .* rate_value .* primal
    pcols = ntuple(
        j -> scale .* (rate_partials[j] .* primal .+ rate_value .* @view(sums[:, j + 1])),
        Val(N)
    )
    make_dual(v, p::Vararg{V, N}) = ForwardDiff.Dual{Tag, V, N}(v, ForwardDiff.Partials(p))
    return make_dual.(value, pcols...)
end

"""
    Ωgw(spectral_density, frequency, H0::Real)

Dimensionless gravitational-wave energy density per logarithmic frequency,

``\\Omega_{\\mathrm{GW}}(f) = \\frac{4\\pi^2}{3 H_0^2} f^3 S_h(f)``,

where ``S_h(f)`` is the strain spectral density (same units as [`spectral_density`](@ref) on fluxes)
and ``H_0`` is the Hubble constant in **s⁻¹**.

``H_0`` is passed in **km/s/Mpc** (matching hyperparameter `H0` and [`LambdaCDM`](@ref).`H0`)
and converted internally via [`CBCDistributions.hubble_constant_si`](@ref).

`frequency` and `spectral_density` may be scalars or arrays; they broadcast together (e.g. same-length
vectors for one spectrum per frequency bin).
"""
function Ωgw(spectral_density, frequency, H0::Real)
    h0_si = CBCDistributions.hubble_constant_si(H0)
    pre = 4 * pi^2 / (3 * h0_si^2)
    return @. pre * frequency^3 * spectral_density
end
