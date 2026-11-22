#!/bin/bash
#SBATCH --job-name=rental-anchor-access
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rental-access-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
: "${JOBS_PYTHON:?Set project Python interpreter}"
: "${JOBS_PANEL:?Set verified anonymous replay packet}"
: "${JOBS_RESULTS:?Set output directory}"
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access_batch \
    --packet "${JOBS_PANEL}" --out "${JOBS_RESULTS}" --workers "${SLURM_CPUS_PER_TASK}" --shards 20
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access_analysis \
    --runs "${JOBS_RESULTS}" --out results/rental-anchor-access-analysis
