#!/bin/bash
# Open-model recommendations (experiments/open_model_recommend.py) on one H200 per array task.
#
#     sbatch --account=YOUR_LAB --array=0-3 --export=ALL,CONFIG=configs/stage2_main.json,FAMILY=qwen,NUM_SHARDS=4 slurm/gpu_openmodels.sbatch
#     # after every task ends (or chain it: sbatch --dependency=afterok:<jobid> ...), merge once, e.g. on a test node:
#     python -m experiments.open_model_recommend configs/stage2_main.json --family qwen --merge-only
#
# Profile first (protocol section 8: the same fixed 500 requests on each card type, before sizing runs):
#     sbatch --account=YOUR_LAB --array=0 --export=ALL,CONFIG=configs/stage2_main.json,FAMILY=qwen,NUM_SHARDS=1,EXTRA="--limit 500 --tag profile_h200 --no-merge" slurm/gpu_openmodels.sbatch
#
# Sizing: Qwen2.5-32B in bf16 is about 65 GB of weights and Mistral-Small-3.1-24B about 48 GB, so each fits one 141 GB
# H200 with room for the KV cache: one GPU per replica (protocol: prefer single-GPU replicas), replicas = array tasks.
# -t is a placeholder until the 500-request profile gives requests per GPU-hour; tasks resume where they stopped.
#   gpu_h200   FASRC partition with 4 x H200 per node (docs: Running Jobs). gpu (A100) and gpu_requeue are alternatives;
#              keep one card type per model for the primary comparison (protocol section 8).
#   --gres=gpu:1   the docs' syntax for gpu/gpu_h200 (--gres=gpu:n, n = 1..4).
#   -c 8, --mem=64G   vLLM tokenizes and schedules on the CPU; neither value comes from FASRC guidance (the docs give no
#              per-GPU ratio). Adjust after the profile run (seff <jobid>).
#SBATCH -J ho-openmodels
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
## FILL IN (or pass on the command line): your lab's fairshare account.
##SBATCH --account=YOUR_LAB
set -euo pipefail

: "${CONFIG:?set CONFIG}"
: "${FAMILY:?set FAMILY to qwen or mistral}"
NUM_SHARDS="${NUM_SHARDS:-1}"
EXTRA="${EXTRA:-}"
ENV_ROOT="${ENV_ROOT:-/n/holylabs/YOUR_LAB/Lab/$USER/conda}"            # FILL IN: same value as in setup_env.sh
VLLM_ENV_NAME="${VLLM_ENV_NAME:-hiddenoptions-vllm}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
export HF_HOME="${HF_HOME:-${SCRATCH:-/n/netscratch}/YOUR_LAB/Lab/$USER/hf}"  # FILL IN: where setup_env.sh put the weights
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights were pre-downloaded by setup_env.sh --with-vllm at the pinned revision
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load "$PYTHON_MODULE"
# CUDA: vLLM's pip wheels bundle their CUDA runtime. If they cannot find the driver, load a CUDA module
# (`module-query cuda`, then `module load cuda/<version>`; FASRC GPU docs) - unverified whether this is needed.
export CONDA_ENVS_PATH="$ENV_ROOT/envs" CONDA_PKGS_DIRS="$ENV_ROOT/pkgs"
eval "$(mamba shell hook --shell bash 2>/dev/null || conda shell.bash hook)"
mamba activate "$VLLM_ENV_NAME"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "$(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} task=$SHARD/$NUM_SHARDS family=$FAMILY config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if [[ "$NUM_SHARDS" -gt 1 ]]; then
    shard_args=(--shard "$SHARD" --num-shards "$NUM_SHARDS")
else
    shard_args=()
fi
# shellcheck disable=SC2086  # EXTRA is a deliberate word list
python -m experiments.open_model_recommend "$CONFIG" --family "$FAMILY" ${shard_args[@]+"${shard_args[@]}"} $EXTRA
echo "$(date -Is) done"
