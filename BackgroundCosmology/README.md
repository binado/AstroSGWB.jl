# BackgroundCosmology

Flat FLRW cosmology kernels for gravitational-wave background modeling: the dimensionless Hubble
parameter `E(z)` for ΛCDM, wCDM (`W0CDM`) and w0waCDM (`W0WaCDM`), and AD-friendly distance and
volume functions — scalar `comoving_distance`, `luminosity_distance`,
`differential_comoving_volume`, and the single-pass `distance_and_volume_grid` tabulation consumed
by redshift-population codes. Distances are in Mpc; `differential_comoving_volume` is the
solid-angle-integrated `4π · dV_c/dz` in Mpc³ per unit redshift. All quadrature is fixed
Gauss–Legendre (no adaptive branching), so ForwardDiff duals propagate through every call, and the
results match the Python `astrogwb` stack to rounding error.

## Installation

BackgroundCosmology lives in the [AstroSGWB.jl](https://github.com/binado/AstroSGWB.jl) monorepo
and installs from a git URL with `subdir`:

```julia
using Pkg
Pkg.add(url = "https://github.com/binado/AstroSGWB.jl", subdir = "BackgroundCosmology")
```

## Example

ΛCDM distances and the volume grid:

```julia
using BackgroundCosmology

c = LambdaCDM(67.0, 0.315)            # H0 in km/s/Mpc, Ωm
comoving_distance(1.0, c)             # Mpc
luminosity_distance(1.0, c)           # Mpc
differential_comoving_volume(1.0, c)  # 4π · dV_c/dz, Mpc³ per unit redshift

# Tabulate all three on your own grid in a single pass
z = collect(LinRange(0.0, 2.0, 101))
grid = distance_and_volume_grid(c, z)
# -> (; comoving_distance, luminosity_distance, differential_comoving_volume)
```

Dark-energy variants only change the constructor — `W0CDM(67.0, 0.315, -1.0)` or
`W0WaCDM(67.0, 0.315, -1.0, 0.0)` — every distance and volume call is generic over the cosmology.
