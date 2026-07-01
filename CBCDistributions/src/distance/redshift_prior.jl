export redshift_prior

"""
    redshift_prior(sf, Λ, cache::CosmologyCache) -> RedshiftInterpolatedDistribution

Cosmology bridge: evaluate the differential comoving volume on the cache's redshift
grid, build the detector-frame `dN/dz` via [`redshift_density`](@ref), and wrap it in a
[`RedshiftInterpolatedDistribution`](@ref). This is the only redshift-prior entry point
that touches a cosmology; the underlying density/kernel API is cosmology-agnostic.
"""
function redshift_prior(sf, Λ, cache::CosmologyCache)
    z_grid = cache.inv_E_integral.x
    dvc_grid = differential_comoving_volume.(z_grid, Ref(cache))
    return RedshiftInterpolatedDistribution(redshift_density(z_grid, dvc_grid, sf, Λ))
end
