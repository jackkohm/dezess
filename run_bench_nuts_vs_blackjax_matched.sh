#!/bin/bash
#SBATCH --job-name=bench_nuts_m
#SBATCH --output=slurm_bench_nuts_m_%j.out
#SBATCH --error=slurm_bench_nuts_m_%j.err
#SBATCH --time=0:25:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
module load conda/Miniforge3-25.3.1-3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/scrubbed/jackkohm/.jax_cache
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "${SLURM_SUBMIT_DIR}"; fi
cd GIVEMEPotential/third_party/dezess
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
conda run --no-capture-output -p /gpfs/scrubbed/jackkohm/conda-envs/Astro python -u bench_nuts_vs_blackjax_matched.py
