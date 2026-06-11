#!/bin/bash
#SBATCH --job-name=bench_tuner
#SBATCH --output=slurm_bench_tuner_%j.out
#SBATCH --error=slurm_bench_tuner_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
module load conda/Miniforge3-25.3.1-3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/scrubbed/jackkohm/.jax_cache
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "${SLURM_SUBMIT_DIR}"; fi
cd GIVEMEPotential/third_party/dezess
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CONDA_ENV="/gpfs/scrubbed/jackkohm/conda-envs/Astro"

echo "=== regression: block-gibbs hybrid + lp_dead guard tests ==="
conda run --no-capture-output -p "$CONDA_ENV" python -m pytest \
  dezess/tests/test_complementary_hybrid.py dezess/tests/test_lp_dead_guard.py \
  -x -q --no-header
echo "=== A/B: tuner target ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u bench_tuner_target.py
