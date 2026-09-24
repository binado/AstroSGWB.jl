# GWDistributions

Population-distribution building blocks for gravitational-wave source modeling: interpolated 1-D
distributions with cached normalizers (`Interpolated1DDistribution`), the DEFAULT BBH mass law
(`DefaultBBHPrimaryMass` / `DefaultBBHMassPair` with `planck_taper`), and the Madau–Dickinson
source-frame merger rate (`MadauDickinsonSourceFrame`) composed into a detector-frame redshift
distribution (`RedshiftInterpolatedDistribution`) whose `normalizer` is the total merger rate in
events/sec once the local rate `R₀` is included.

The package is cosmology-independent by design: redshift distributions take solid-angle-integrated
`dV_c/dz` **arrays**, not cosmology types. Pair it with
[BackgroundCosmology](https://github.com/binado/AstroSGWB.jl/tree/main/BackgroundCosmology) (not a
dependency) for real cosmology.

## Installation

GWDistributions lives in the [AstroSGWB.jl](https://github.com/binado/AstroSGWB.jl) monorepo and
installs from a git URL with `subdir`:

```julia
using Pkg
Pkg.add(url = "https://github.com/binado/AstroSGWB.jl", subdir = "GWDistributions")
```

## Example

A Madau–Dickinson redshift distribution from a volume grid, with `logpdf` and `rand`:

```julia
using GWDistributions, QuadGK

# Flat-ΛCDM dV_c/dz grid (Mpc³ per unit redshift), computed inline — swap in
# BackgroundCosmology.distance_and_volume_grid if you have it
H0, Ωm = 67.0, 0.315
d_h = 299_792.458 / H0                 # Hubble distance, Mpc
E(z) = sqrt(Ωm * (1 + z)^3 + 1 - Ωm)
d_c(z) = d_h * quadgk(x -> 1 / E(x), 0, z)[1]
dVc(z) = 4π * d_h * d_c(z)^2 / E(z)

z = collect(LinRange(0.0, 2.0, 201))
sf = MadauDickinsonSourceFrame(; γ = 2.7, κ = 3.0, zpeak = 2.5, R₀ = 161.0)  # R₀ in Gpc⁻³ yr⁻¹
d = RedshiftInterpolatedDistribution(sf, dVc.(z), z)

logpdf(d, 0.5)    # detector-frame density at z = 0.5
rand(d, 10)       # 10 redshift samples
normalizer(d)     # total detector-frame merger rate, events/sec
```
