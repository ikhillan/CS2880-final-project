#!/bin/bash
# Submit a config as a CPU job array of N shards (array indices 0..N-1).
#
#     FASRC_ACCOUNT=your_lab bash slurm/submit_array.sh configs/stage3_protection.json 300
#     FASRC_ACCOUNT=your_lab THROTTLE=100 bash slurm/submit_array.sh configs/stage2_main.json 200 -t 0-02:00
#
# Extra arguments go to sbatch and override the defaults in run_array.sbatch (e.g. -t, --mem, -p, --test-only).
# Environment: FASRC_ACCOUNT (required), FASRC_PARTITION (default: the sbatch file's serial_requeue), THROTTLE (max tasks
# running at once, optional). Run from the repo root, outside any active environment (FASRC guidance).
# Rerunning the same command after a partial run is safe: each shard skips markets it already finished.
# Afterwards: python -m experiments.analyze <name> ... reads results/<name>/shards/*.jsonl directly; no merge step.
set -euo pipefail

CONFIG="${1:?usage: submit_array.sh CONFIG NUM_SHARDS [sbatch args...]}"
NUM_SHARDS="${2:?usage: submit_array.sh CONFIG NUM_SHARDS [sbatch args...]}"
shift 2
: "${FASRC_ACCOUNT:?set FASRC_ACCOUNT to your lab Slurm account (see sshare -U)}"
[[ -f "$CONFIG" ]] || { echo "no config $CONFIG" >&2; exit 2; }
(( NUM_SHARDS >= 1 && NUM_SHARDS <= 10000 )) || { echo "NUM_SHARDS must be 1..10000 (FASRC max array size)" >&2; exit 2; }

# Optional sanity check when the project environment is active: more shards than markets only creates empty tasks.
if python -c "import experiments.config" 2>/dev/null; then
    n_markets=$(python -c "import sys; from experiments.config import load_config, expand_markets; print(len(expand_markets(load_config(sys.argv[1]))))" "$CONFIG")
    echo "$CONFIG: $n_markets markets over $NUM_SHARDS shards (about $(( (n_markets + NUM_SHARDS - 1) / NUM_SHARDS )) per task)"
    (( NUM_SHARDS <= n_markets )) || echo "warning: more shards than markets; some tasks will have nothing to do" >&2
fi

mkdir -p logs
range="0-$((NUM_SHARDS - 1))${THROTTLE:+%$THROTTLE}"
args=(--account="$FASRC_ACCOUNT" --array="$range" --export="ALL,CONFIG=$CONFIG,NUM_SHARDS=$NUM_SHARDS"
      -J "ho-$(basename "$CONFIG" .json)")
[[ -n "${FASRC_PARTITION:-}" ]] && args+=(-p "$FASRC_PARTITION")
set -x
sbatch "${args[@]}" "$@" slurm/run_array.sbatch
