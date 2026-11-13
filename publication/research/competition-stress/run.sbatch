#!/bin/bash
#SBATCH --job-name=cs2880-match
#SBATCH --account=kempner_gil_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
/usr/bin/nvidia-smi --query-gpu=name,memory.total --format=csv
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for runtime in "$HOME/.conda/envs/boltzgen/bin/python" "$HOME/.conda/envs/drug/bin/python"; do
    if test -x "$runtime" && "$runtime" -c 'import torch; assert torch.cuda.is_available()'; then
        if [[ "${CROSS_PRICES:-0}" == "1" ]]; then
            exec "$runtime" -B run_gpu.py --ranks ranks.npz --out crossed --chunk 16 --cross-prices
        fi
        exec "$runtime" -B run_gpu.py --ranks ranks.npz --out result --chunk 16
    fi
done
exit 1
