export MadauDickinsonSourceFrame, madau_dickinson_source_frame_distribution

"""
    madau_dickinson_source_frame_distribution(z; γ, κ, zpeak) -> Real

Source-frame merger-rate density at redshift `z` under the Madau–Dickinson (2014)
star-formation-rate model. The denominator exponent is `γ + κ` (so `κ` is the
increment beyond `γ`). The trailing `(1 + (1 + zpeak)^(-(γ + κ)))` factor normalizes
the density so that `ψ(0) = 1`, satisfying the [`source_frame_distribution`](@ref)
contract.
"""
function madau_dickinson_source_frame_distribution(
        z::Real;
        γ::Real,
        κ::Real,
        zpeak::Real
)
    one_plus_z = 1 + z
    denom_exp = γ + κ
    return ((one_plus_z^γ) / (1 + (one_plus_z / (1 + zpeak))^denom_exp)) *
           (1 + (1 + zpeak)^(-denom_exp))
end

"""
    MadauDickinsonSourceFrame

Dispatch tag for the Madau–Dickinson source-frame merger-rate model. Pass to
[`source_frame_distribution`](@ref) together with hyperparameters `Λ` carrying
`γ`, `κ`, `zpeak`.
"""
struct MadauDickinsonSourceFrame end

"""
    source_frame_distribution(::MadauDickinsonSourceFrame, z, Λ) -> Real

Madau–Dickinson source-frame density at `z`, reading `γ`, `κ`, `zpeak` from `Λ`.
Normalized to `ψ(0) = 1`.
"""
function source_frame_distribution(::MadauDickinsonSourceFrame, z::Real, Λ)
    return madau_dickinson_source_frame_distribution(z; γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
end
