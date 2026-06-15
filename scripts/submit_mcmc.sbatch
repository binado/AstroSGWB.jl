#!/bin/bash
#SBATCH --job-name=asgwb-mcmc
#SBATCH --output=logs/mcmc-%j.out
#SBATCH --error=logs/mcmc-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Headless ASGWB NUTS run on SLURM. One config file == one experiment.
#
# Submit from the repository root with the config as the single positional argument:
#   cd /path/to/ASGWB.jl
#   sbatch scripts/submit_mcmc.sbatch config/mcmc/example.toml
#
# One Turing chain runs per allocated CPU thread (MCMCThreads), so set
# --cpus-per-task to the number of chains you want.

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/submit_mcmc.sbatch <config.toml>}"

# Slurm executes a copied batch script from its spool directory on some
# clusters, so BASH_SOURCE/dirname can point at /var/spool/slurmd. Anchor all
# relative paths to the directory where sbatch was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ ! -d ASGWB || ! -d ASGWBInference || ! -f scripts/run_mcmc.jl ]]; then
    echo "submit_mcmc.sbatch must be submitted from the ASGWB.jl repository root" >&2
    echo "current directory: $PWD" >&2
    exit 2
fi

# Adjust this to your cluster's module system / Julia install.
# module load julia/1.12

# Each Julia thread runs one MCMC chain.
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Keep BLAS single-threaded; the runner also enforces this.
export OPENBLAS_NUM_THREADS=1

echo "[submit_mcmc] config=$CONFIG threads=$JULIA_NUM_THREADS"

# Instantiate the slim runner env once (cheap if already resolved).
julia --project=scripts/run -e 'using Pkg; Pkg.instantiate()'

julia --project=scripts/run -t "$JULIA_NUM_THREADS" \
    scripts/run_mcmc.jl "$CONFIG"
