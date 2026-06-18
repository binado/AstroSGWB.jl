# AstroSGWB.jl

Julia workspace for modeling and inferring the **astrophysical stochastic gravitational-wave background** (AstroSGWB): detector networks and responses, spectral density calculation, and MCMC with Turing / AdvancedHMC.

## Workspace layout

The root repository is organized as a monorepo comprised of different small packages:

| Path | Role |
|------|------|
| [`AstroSGWB/`](AstroSGWB/) | Core library: cosmology-aware hyperparameters, redshift and spectral-density evaluation, detector PSDs/ORFs, likelihoods, catalog I/O |
| [`AstroSGWBInference/`](AstroSGWBInference/) | Inference layer on top of `AstroSGWB`: Turing model construction, log-posterior helpers, chain I/O |
| [`CBCDistributions/`](CBCDistributions/) | Shared building blocks: ΛCDM / *w*CDM cosmology, redshift distributions, `PopulationModel` contract, and related `Distributions.jl` helpers used by `AstroSGWB`. |
| [`notebooks/`](notebooks/) | **Canonical MCMC workflows** (Pluto / Jupytext): inline population model, `load_catalog`, NUTS sampling, diagnostics. |
| [`config/`](config/) | TOML for developer scripts and headless MCMC runs (e.g. [`config/mcmc/example.toml`](config/mcmc/example.toml)). |
| [`scripts/`](scripts/) | Developer utilities (profiling, chain tools, benchmarks) and [`scripts/run_mcmc.jl`](scripts/run_mcmc.jl) for config-driven cluster runs. |


## Installation

Clone the repository and instantiate the workspace from the repo root:

```bash
cd AstroSGWB.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That resolves all workspace members (`AstroSGWB`, `AstroSGWBInference`, `CBCDistributions`, `notebooks`) and their shared manifest.

Run tests:

```bash
just test
# or
julia --project=AstroSGWB -e 'using Pkg; Pkg.test()'
julia --project=AstroSGWBInference -e 'using Pkg; Pkg.test()'
```

## MCMC inference

### Data and model assembly

1. Provide a waveform **catalog** HDF5 file (`catalog.h5`) at the repo root or set `catalog_path` in the notebook. Catalogs store per-sample intrinsic parameters and a `(n_freq, n_samples)` flux matrix `|h₊|² + |h×|²` (before fiducial `(D_L/D_gw)²` scaling). Use [`AstroSGWB.load_catalog`](AstroSGWB/src/catalog/io.jl) / [`AstroSGWB.save_catalog`](AstroSGWB/src/catalog/io.jl).
2. Define a concrete [`PopulationModel`](CBCDistributions/src/population_model.jl) subtype in Julia (`hyperparameters`, `single_event_prior`) and build hyperparameter priors with `product_distribution(...)` (see the notebook cells).
3. Restructure catalog columns into the `NamedTuple` expected by `single_event_prior` (see `bns_samples_from_catalog` in the notebook).
4. Build [`ImportanceSamplingProblem`](AstroSGWB/src/inference_types.jl) with fiducial hyperparameters, then [`build_model_context`](AstroSGWB/src/context.jl) for detector PSDs and fiducial caches.
5. Sample with `AstroSGWBInference.build_turing_model`, `condition_turing_model`, and Turing NUTS; save chains via `AstroSGWBInference.atomic_save_chain`.

Waveform generation is not part of the Julia packages; see [scripts/generate_waveforms.py](./scripts/generate_waveforms.py) for a standalone Python accumulator (legacy layout).

### Launch Pluto MCMC

From the repository root:

```bash
just pluto
# or
julia --project=notebooks -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook="notebooks/mcmc.jl")'
```

Edit fiducials, hyperprior bounds, detectors, and sampler settings in the notebook cells.

### Headless MCMC (config-driven)

[`scripts/run_mcmc.jl`](scripts/run_mcmc.jl) mirrors the sampling cells of [`notebooks/mcmc_pluto.jl`](notebooks/mcmc_pluto.jl) but reads run-specific settings from a TOML file. Cosmology (`ModifiedPropagation{W0CDM}`), population model (`BNSPopulationModel`), and hyperprior bounds are fixed in the script, matching the notebook.

**One-time setup** (separate Julia project at [`scripts/run/`](scripts/run/)):

```bash
just setup-run
# or
julia --project=scripts/run -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Copy [`config/mcmc/example.toml`](config/mcmc/example.toml) per experiment and edit catalog path, detectors, fiducials, `sample_only`, sampler settings, and output paths. Catalog paths are resolved relative to the repository root unless absolute. Use ASCII keys in `[fiducials]` (e.g. `Omega_m`, `Xi_0`, `gamma`).

**Run locally** (one Turing chain per Julia thread via `MCMCThreads()`):

```bash
just run-mcmc config/mcmc/my_run.toml
# or
julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/my_run.toml
```

`sampler.num_chains` defaults to `0`, which uses `Base.Threads.nthreads()`. If set explicitly, it must equal the thread count passed to `-t` (or `SLURM_CPUS_PER_TASK` on a cluster). The runner currently supports `ad_backend = "ForwardDiff"` only. Chains are written as JLD2 under `output_dir` (default `chains/`).

**Submit on SLURM** from the repository root (pre-instantiate on the login node with `just setup-run`; the batch script does not run `Pkg.instantiate()` on compute nodes):

```bash
just submit-mcmc config/mcmc/my_run.toml
# or
mkdir -p logs
sbatch scripts/submit_mcmc.sbatch config/mcmc/my_run.toml
```

Set `#SBATCH --cpus-per-task` in [`scripts/submit_mcmc.sbatch`](scripts/submit_mcmc.sbatch) to the number of chains you want; adjust the Julia module load line for your cluster.

### Profiling the log-density

To profile a NUTS gradient evaluation without running a full notebook:

```bash
julia --project=AstroSGWBInference scripts/profile_turing.jl --config-file=config/profile_turing.toml
```

## Notebooks

Notebooks live under [`notebooks/`](notebooks/) as Pluto (`.jl` with Pluto cell markers) or **Jupytext** “percent” Julia scripts. They activate the `notebooks/` project (`Pkg.activate(@__DIR__)`) and pull in `AstroSGWB` / `AstroSGWBInference` via path dependencies.

| Notebook | Purpose |
|----------|---------|
| [`notebooks/mcmc.jl`](notebooks/mcmc.jl) | **Canonical** end-to-end catalog load, Ω_GW plots, Turing NUTS, chain save/load. |
| [`notebooks/plots.jl`](notebooks/plots.jl) | MCMC diagnostics and figures from saved chains (`FlexiChains`, `PairPlots`, `CairoMakie`). |
| [`notebooks/amplitude_posterior_gaussian_approximation.jl`](notebooks/amplitude_posterior_gaussian_approximation.jl) | Compare a 1D posterior to a Fisher / SNR Gaussian approximation (single-parameter chains). |

### Setup

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

For Jupyter, register a kernel (once) from the `notebooks/` directory:

```bash
cd notebooks
julia --project=. -e 'using IJulia; IJulia.installkernel("AstroSGWB notebooks"; "--project=$(abspath("."))")'
```

Then open the `.jl` files in Jupyter Lab, VS Code, or Cursor with the Julia/IJulia extension (Jupytext notebooks).

To sync paired `.ipynb` files if you use them:

```bash
just sync-notebook
# jupytext 'notebooks/*.ipynb' --to jl:percent
```

Notebook outputs and shared plotting helpers use [`notebooks/src/NotebookSupport.jl`](notebooks/src/NotebookSupport.jl); figures default under `output-test-figures/` unless `AstroSGWB_FIGURES_DIR` is set.

## Further reading

- [`AGENTS.md`](AGENTS.md) — contributor conventions, testing, and architecture notes.
