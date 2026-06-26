# Repository Guidelines

## Project Structure & Module Organization

AstroSGWB.jl is a Julia package for astrophysical stochastic gravitational-wave background modeling and inference. The package entry point is `AstroSGWB/src/AstroSGWB.jl`, which includes and exports the public API (physics, likelihoods, importance caches). Turing/AdvancedHMC model wrappers and sampling helpers live in the sibling package `AstroSGWBInference/` (see `AstroSGWBInference/src/InferenceImpl.jl`). Core modules live in `AstroSGWB/src/`, with detector-specific code under `AstroSGWB/src/detector/`. Tests live in `AstroSGWB/test/` and are listed from `AstroSGWB/test/runtests.jl`. Detector definitions and noise curves are stored in `AstroSGWB/assets/detector/`. **Production MCMC is notebook-first** (see `notebooks/mcmc.jl`): callers define a concrete [`PopulationModel`](AstroSGWB/src/models/base.jl), load a waveform catalog with [`load_catalog`](AstroSGWB/src/catalog/io.jl), build a pure [`ImportanceSamplingProblem`](AstroSGWB/src/inference_types.jl) (raw fluxes + fiducial hyperparameters, no cosmology or detector state), then assemble a caller-owned **prepared model** plus an [`ObservationContext`](AstroSGWB/src/detector/observation.jl) (the cosmology-specific proposal caches + detector/observation state), and sample via `AstroSGWBInference.build_turing_model(model, problem, observation, prior)`. Developer utilities (profiling, chain stacking, benchmarks) live in `scripts/` (for example `stack_partial_chains.jl` and `profile_turing.jl`). Pluto and Jupytext notebooks are under `notebooks/`.

Note: the package is still in the prototyping phase, so breaking changes are allowed and backwards compatibility is not a concern.

## Build, Test, and Development Commands

- `just fmt`: format the repository using JuliaFormatter.
- `just test`: run the package test suite through `Pkg.test()`.
- `just pluto`: instantiate the notebook environment and launch Pluto.
- `julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'` then open `notebooks/mcmc.jl` in Pluto for the canonical MCMC workflow.

If `just` is unavailable, use the Julia commands in the `justfile`, for example `julia --project=AstroSGWB -e 'using Pkg; Pkg.test()'`.

## Coding Style & Naming Conventions

Target Julia version is 1.12. Formatting is controlled by `.JuliaFormatter.toml` with `style = "sciml"`; run `just fmt` before committing. Keep code type-stable where practical, especially in likelihood, redshift, and spectral-density paths.

Use existing Unicode scientific identifiers, including `Ωm`, `Ξ₀`, `Ξₙ`, `γ`, `κ`, `Λ`, `χ₁`, and `χ₂`. Live inference hyperparameters are a flat `NamedTuple`; `canonical_hyperparameters` validates, orders, and converts boundary inputs. Unconstrained HMC/Turing vector layout is owned by the prepared model via `full_hyperparameters(model)`, which delegates to the `full_hyperparameters(C, P, pop)` helper. The cosmology/propagation families `C`/`P` are model-internal type parameters, so the package inference surface never takes a `::Type{C}`/`::Type{P}` token; callers implement the single `merger_rate_and_log_weights(model, Λ, samples)` joint (see [`contract.jl`](AstroSGWB/src/contract.jl)) on their prepared type. Test files follow `test_<area>.jl`, such as `test_redshift.jl`.

## Testing Guidelines

Tests use Julia's standard `Test` framework and are orchestrated by `AstroSGWB/test/runtests.jl`. Add new test files to `runtests.jl`; otherwise they will not run in CI or `Pkg.test()`. Prefer focused numerical tests with explicit tolerances for cosmology, redshift integrals, likelihoods, and interpolation behavior. Minimal catalog fixtures for tests are written on demand via `parity_catalog_dir` (see `AstroSGWB/test/parity_test_cache.jl`); shared smoke-test hyperparameters live in `AstroSGWB/test/parity_fixtures.jl`. The reference BNS population for tests is `ParityBNSPopulation` in `AstroSGWB/test/fixture_population.jl`.

Run `just test` before opening a pull request. For narrow changes, also run the closest individual test file during development with `julia --project=AstroSGWB AstroSGWB/test/test_<area>.jl`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects: `feat:`, `chore:`, `refactor:`, `perf:`, and `refactor!:` for breaking changes. Keep subjects imperative and scoped to one logical change.

Pull requests should include a short summary, testing performed, and links to relevant issues or discussions. Note behavior changes, fixture updates, performance-sensitive changes, and any compatibility impact. Include plots or screenshots only when changing notebooks or user-facing visual outputs.

## Architecture Notes

The package is **cosmology-agnostic**. The main inference flow is: load `catalog.h5` with `load_catalog`, restructure per-event samples to match `single_event_prior`, construct a pure [`ImportanceSamplingProblem`](AstroSGWB/src/inference_types.jl) (raw fluxes + fiducial hyperparameters), assemble a caller-owned **prepared model** (proposal prior + per-component log-prob, `dl_fid_sq`, redshift grid/interpolant, `local_merger_rate`/`observation_time`) alongside an [`ObservationContext`](AstroSGWB/src/detector/observation.jl) built by `build_observation_context`, then run importance weighting, spectral-density evaluation, and posterior sampling. The single dispatch boundary is `merger_rate_and_log_weights(model, Λ, samples)` ([`contract.jl`](AstroSGWB/src/contract.jl)), which model authors implement on their prepared type and which fuses the detector-frame `merger_rate` with the per-sample `importance_log_weights`. Everything *above* this boundary (cosmology cache, prior, propagation factor `Ξ(z)`, distances) is cosmology-specific and lives on the model; everything *below* it (`spectral_density`, the Gaussian likelihood, SNR tracking) is cosmology-agnostic and lives in the package. The cosmology family `C` is a model-internal type parameter, never stored on the problem. Respect matrix layout conventions: cached spectral flux arrays are `(nfreq, nsamples)` in memory, and intrinsic proposal vectors are `(nsamples, n_intrinsic)`.
