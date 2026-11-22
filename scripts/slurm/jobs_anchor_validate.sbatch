#!/bin/bash
#SBATCH --job-name=jobs-anchor-validate
#SBATCH --partition=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
JOBS_PYTHON="${JOBS_PYTHON:-/n/home05/kkl/.conda/envs/boltzgen/bin/python}"
"${JOBS_PYTHON}" -m pytest -q tests/test_jobs_coordinated_policies.py \
    tests/test_jobs_anchor_access_metrics.py tests/test_jobs_anchor_access_runner.py
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access synthetic-packet \
    --out results/validation/panel.npz --markets 40
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access evaluate \
    --packet results/validation/panel.npz --out results/validation/replay \
    --ties 20 --allow-synthetic
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access_analysis \
    --runs results/validation/replay --out results/validation/analysis --allow-synthetic
