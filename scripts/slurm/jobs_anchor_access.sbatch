#!/bin/bash
#SBATCH --job-name=jobs-anchor-access
#SBATCH --partition=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/access-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
JOBS_PYTHON="${JOBS_PYTHON:-/n/home05/kkl/.conda/envs/boltzgen/bin/python}"
: "${JOBS_PANEL:?Set JOBS_PANEL to the exported frozen jobs replay packet}"
: "${JOBS_RESULTS:?Set JOBS_RESULTS to the output directory}"
"${JOBS_PYTHON}" -m experiments_jobs.anchor_access evaluate \
    --packet "${JOBS_PANEL}" --out "${JOBS_RESULTS}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --shards 20 --ties 20
