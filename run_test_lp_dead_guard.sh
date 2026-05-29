#!/bin/bash
#SBATCH --job-name=test_lp_dead
#SBATCH --output=slurm_test_lp_dead_%j.out
#SBATCH --error=slurm_test_lp_dead_%j.err
#SBATCH --time=0:15:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
module load conda/Miniforge3-25.3.1-3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/scrubbed/jackkohm/.jax_cache
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

CONDA_ENV="/gpfs/scrubbed/jackkohm/conda-envs/Astro"
echo "=== test_lp_dead_guard.py — $(date) ==="

cd GIVEMEPotential/third_party/dezess
conda run --no-capture-output -p "$CONDA_ENV" python -u dezess/tests/test_lp_dead_guard.py
