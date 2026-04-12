# Julia importance cache HDF5 specification

This document describes the HDF5 layout consumed by `load_cache` in **ASGWB.jl** (see `src/io.jl`). Only this layout is guaranteed to load; other files are rejected via root attributes.

## Root attributes

All attributes live on the HDF5 **root group** (`/`).

| Attribute | Type | Required | Meaning |
|-----------|------|----------|---------|
| `format_name` | string | yes | Must be exactly `asgwb.julia.importance_cache`. |
| `format_version` | integer | yes | `1`, `2`, or `3` (see [Format versions](#format-versions)). |
| `local_merger_rate` | real | yes | Local merger rate scale passed to `importance_sampling_problem`. |
| `redshift_integral_fiducial` | real | no | When omitted, set to `fiducial_redshift_integral` (the `norm` from `build_redshift_grid_bundle` at fiducial population parameters). Requires the same population scalars on `hyperparameters` as for reconstructing an omitted `proposal_log_prob`. When present, the stored value is used as-is (legacy caches may use a different normalization than the recomputed integral). |
| `observation_time_sec` | real | yes | Observation duration in seconds (also used when reconstructing covariance from detectors). |
| `observation_time_yr` | real | yes | Observation duration in years. |

## Root groups (required)

### `hyperparameters`

HDF5 group of **scalar datasets** (0-dimensional or single-element), read as `Float64`.

**Always required (all format versions):**

| Dataset | Meaning |
|---------|---------|
| `H0` | Hubble constant (km/s/Mpc style consistent with rest of package). |
| `Omega_m` | Matter density parameter. |
| `chi0` | EM–GW distance ratio parameter \( \chi_0 \). |
| `chin` | EM–GW distance ratio parameter \( \chi_\mathrm{in} \). |

**Optional population scalars** (any format version; required for certain reconstructions):

| Dataset | When needed |
|---------|-------------|
| `gamma`, `kappa`, `z_peak` | Madau–Dickinson redshift prior, if `proposal_log_prob` and/or `dgw_fid_sq` are omitted (format 3), and/or if `fiducial_spectral_density` or `redshift_integral_fiducial` is omitted. |
| `lamb` | Power-law redshift prior, under the same omission rules. |

**Key whitelist:** readers reject unknown dataset names in this group. Allowed names are the four fiducial cosmology keys above plus any of `gamma`, `kappa`, `z_peak`, `lamb`.

### `redshift_prior_spec`

Subgroup (under root) with:

| Entry | Required | Type | Notes |
|-------|----------|------|-------|
| `family` | yes | string | Exactly `madau_dickinson` or `power_law` (snake_case; see `parse_redshift_prior_family` in `src/types.jl`). |
| `z_min` | yes | real | |
| `z_max` | yes | real | |
| `num_interp` | yes | integer | Grid size for redshift prior interpolation. |
| `time_delay_model` | no | string | If absent or empty, loads as `nothing`. Reserved for future use. |

### `proposal_samples`

Group containing one **1D float dataset per name** in `intrinsic_site_order`. Each vector must have the same length \(n_\mathrm{samples}\), the number of importance samples.

**Required layout (full BNS):** `intrinsic_site_order` must match this exact order, with matching datasets under `proposal_samples/`:

`mass_1_source`, `mass_2_source`, `redshift`, `chi_1`, `chi_2`, `lambda_1`, `lambda_2`.

Redshift-only caches (`intrinsic_site_order == ["redshift"]` only) are no longer supported. Any other `intrinsic_site_order` is rejected.

**Group attributes (on `proposal_samples`):**

| Attribute | Type | Required | Meaning |
|-----------|------|----------|---------|
| `source_type` | string | no | Compact-object class for the proposal samples. If omitted, loaders treat the file as **BNS** (backward compatibility). If present, must be exactly `BNS` for this package version; other values (for example `BBH`) are reserved for future layouts. |

## Root datasets

### Always required

| Dataset | Kind | Constraints |
|---------|------|----------------|
| `intrinsic_site_order` | 1D array of strings | Must equal the full BNS order in [`proposal_samples`](#proposal_samples). |
| `proposal_intrinsic_vector` | 2D float | After load: shape `(n_samples, length(intrinsic_site_order))`. See [2D array storage](#2d-array-storage). |
| `frequencies` | 1D float, non-empty | Length \(n_f\). |
| `in_band_mask` | 1D bool | Length \(n_f\). |

### Observation: `covariance` and `sgwb_scale`

- **Format 1:** both **`covariance`** and **`sgwb_scale`** must exist as 1D float datasets of length \(n_f\).
- **Format 2 or 3:** they may be **omitted together**. If omitted, `load_cache(path; detectors=[...])` must be called with **at least two** `Detector` values (see `src/detector/detector.jl`); covariance and scales are rebuilt from tabulated PSDs and overlap reduction functions using `frequencies`, `in_band_mask`, `observation_time_sec` / `observation_time_yr`, and the spectral-density vector used for that reconstruction (zeros or on-disk values; see `fiducial_spectral_density` below).

### `fiducial_spectral_density`

- **Optional** for all format versions.
- If present: 1D float, length \(n_f\), used as the fiducial \(\Omega_\mathrm{GW}(f)\) in `ObservationConfig`.
- If absent: a placeholder of zeros is used to build the problem, then the package recomputes the spectrum via `fiducial_spectral_density` in `src/posterior.jl` (cached flux + fiducial hyperparameters). That step requires the same population entries in `hyperparameters` as for reconstructing an omitted `proposal_log_prob` (Madau–Dickinson: `gamma`, `kappa`, `z_peak`; power-law: `lamb`).

### `proposal_log_prob`

- **Format 1–2:** required; 1D float, length \(n_\mathrm{samples}\).
- **Format 3:** optional. If omitted, values are recomputed from `proposal_samples`, `redshift_prior_spec`, and `hyperparameters` (including population keys as in `hyperparameters_from_fiducial` in `src/cache.jl`).

### `dgw_fid_sq`

- **Format 1–2:** required; 1D float, length \(n_\mathrm{samples}\). Per-sample squared GW luminosity distance at fiducial cosmology (as used by the importance likelihood).
- **Format 3:** optional. If omitted, recomputed from `proposal_samples/redshift` and `hyperparameters` (`reconstruct_dgw_fid_sq` in `src/cache.jl`).

### Cached flux: `cached_flux_over_dgw2` vs `cached_flux`

- **Format 1–2:** dataset **`cached_flux_over_dgw2`** is required (2D float). Semantics match in-memory `ProposalData`: rows = samples, columns = frequency bins — after the transpose described in [2D array storage](#2d-array-storage).
- **Format 3:** dataset **`cached_flux`** is required instead. It stores flux **before** multiplying by \((D_L / D_\mathrm{gw})^2\) under the fiducial cosmology/propagation in `hyperparameters`. On load, the package forms `cached_flux_over_dgw2` as `cached_flux .* ((D_L./D_gw).^2)` per row (`reconstruct_cached_flux_over_dgw2` in `src/cache.jl`). Row count must equal `length(proposal_samples/redshift)`.

## 2D array storage

Readers use `permutedims` on the raw HDF5 array so that **logical** matrices have shape **(number of samples, number of columns)**:

- **`proposal_intrinsic_vector`:** columns align with `intrinsic_site_order`.
- **`cached_flux_over_dgw2` (v1–2)** or **`cached_flux` (v3):** columns align with `frequencies`.

**Julia reference when writing files:** if `M` is `n_samples × n_columns` in memory, write `Matrix(permutedims(M))` so the on-disk dataset has dimensions `n_columns × n_samples` (see `test/test_io.jl`).

## Consistency checks (enforced on load)

The loader requires, among others:

- If `proposal_samples` has attribute `source_type`, it must be `BNS` for this version.
- `proposal_intrinsic_vector` rows = `n_samples`; columns = `length(intrinsic_site_order)`.
- Flux matrix rows = `n_samples`; columns = `n_f`.
- `dgw_fid_sq`, `proposal_log_prob`, and every `proposal_samples/*` vector length = `n_samples`.
- `covariance`, `sgwb_scale`, `in_band_mask`, and fiducial spectral vector (whether from disk or placeholder) length = `n_f`.

## Format versions

| Version | Summary |
|---------|---------|
| **1** | `covariance`, `sgwb_scale`, `cached_flux_over_dgw2`, `proposal_log_prob`, and `dgw_fid_sq` are all required on disk. |
| **2** | Like 1, but `covariance` and `sgwb_scale` may be omitted and reconstructed with `detectors=`. |
| **3** | Uses `cached_flux` instead of `cached_flux_over_dgw2`. May omit `proposal_log_prob`, `dgw_fid_sq`, and/or `fiducial_spectral_density` with the reconstruction rules above. `hyperparameters` may include population scalars (`gamma`, `kappa`, `z_peak` or `lamb`) required for those reconstructions. |

## API

```julia
load_cache(path::AbstractString; detectors=nothing) -> ImportanceSamplingProblem
```

Pass `detectors` when `covariance` and `sgwb_scale` are absent (format ≥ 2).

## Reference implementation

Authoritative behavior is `load_cache` and helpers in `src/io.jl`, `src/cache.jl`, and tests under `test/test_io.jl` (including synthetic v3 files and `test/fixtures/importance_context_julia.h5`). New caches should set `proposal_samples` attribute `source_type` to `BNS` (see exported constants `PROPOSAL_SAMPLES_SOURCE_TYPE_BNS` and `PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`).
