fmt:
    julia -e 'using JuliaFormatter; format(".")'

test:
    julia --project=ASGWB -e 'using Pkg; Pkg.test()'
    julia --project=ASGWBInference -e 'using Pkg; Pkg.test()'
    julia --project=CBCDistributions -e 'using Pkg; Pkg.test()'

pluto threads='"auto"':
    julia -e 'using Pluto; Pluto.run(threads={{threads}})'

run-mcmc config="config/mcmc/example.toml" threads="auto":
    julia --project=scripts/run -e 'using Pkg; Pkg.instantiate()'
    julia --project=scripts/run -t {{threads}} scripts/run_mcmc.jl {{config}}

setup-run:
    julia --project=scripts/run -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

submit-mcmc config="config/mcmc/example.toml":
    mkdir -p logs
    sbatch scripts/submit_mcmc.sbatch {{config}}

resolve package="ASGWB":
    julia --project={{package}} -e 'using Pkg; Pkg.resolve()'

repl project=".":
    julia --project={{project}}

sync-notebook:
    jupytext 'notebooks/*.ipynb' --to jl:percent
