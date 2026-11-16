#!/bin/bash
#SBATCH --job-name=cs2880-conditional
#SBATCH --account=kempner_gil_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1
exec "$HOME/.conda/envs/boltzgen/bin/python" -B run.py inputs.npz draw-partners.npz
