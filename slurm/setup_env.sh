#!/bin/bash
# Build the Python environment(s) and the git-ignored data on FASRC Cannon.
#
# Run it from the repo root inside an interactive job, not on a login node (FASRC builds environments inside salloc):
#     salloc --partition test --nodes=1 --cpus-per-task=2 --mem=8GB --time=0-02:00:00
#     bash slurm/setup_env.sh                # CPU env + data
#     bash slurm/setup_env.sh --with-vllm    # also the GPU env for the open-model arms, plus the pinned weights
#     bash slurm/setup_env.sh --env-only     # skip the data rebuild
#
# Settings (environment variables; see slurm/README.md):
#     ENV_ROOT        where environments and the package cache live. Default: the lab directory placeholder below.
#                     FASRC advises against ~/.conda (slow home filesystem).
#     ENV_NAME        CPU environment name (default hiddenoptions)
#     VLLM_ENV_NAME   GPU environment name (default hiddenoptions-vllm)
#     PYTHON_VERSION  default 3.12
#     VLLM_VERSION    vLLM release for the GPU env (default 0.10.1.1; >= 0.8.1 is needed for Mistral Small 3.1). Pin it once
#                     and record it; experiments/open_model_recommend.py also writes the installed version to its state file.
#     HF_HOME         Hugging Face cache for the model weights (default $SCRATCH placeholder below)
#     PYTHON_MODULE   FASRC module that provides mamba (default "python"; FASRC lists e.g. python/3.14.10-fasrc01 and
#                     Miniforge3/26.9.4-fasrc01 - pin one with `module spider python` first)
set -euo pipefail

WITH_VLLM=0
ENV_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --with-vllm) WITH_VLLM=1 ;;
        --env-only) ENV_ONLY=1 ;;
        *) echo "unknown argument $arg" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- placeholders the team must fill (or export before running) -------------------------------------------------
ENV_ROOT="${ENV_ROOT:-/n/holylabs/YOUR_LAB/Lab/$USER/conda}"           # FILL IN: lab directory path
HF_HOME="${HF_HOME:-${SCRATCH:-/n/netscratch}/YOUR_LAB/Lab/$USER/hf}"  # FILL IN: netscratch lab folder
# ------------------------------------------------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-hiddenoptions}"
VLLM_ENV_NAME="${VLLM_ENV_NAME:-hiddenoptions-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.10.1.1}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"

if [[ "$ENV_ROOT" == *YOUR_LAB* || "$HF_HOME" == *YOUR_LAB* && $WITH_VLLM == 1 ]]; then
    echo "Set ENV_ROOT (and HF_HOME for --with-vllm) to your lab paths first; see slurm/README.md." >&2
    exit 2
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "Deactivate the current environment first (FASRC: do not run setup from inside an active env)." >&2
    exit 2
fi

module load "$PYTHON_MODULE"
export CONDA_ENVS_PATH="$ENV_ROOT/envs" CONDA_PKGS_DIRS="$ENV_ROOT/pkgs"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

# ---- CPU environment: the runner, analysis and data scripts ------------------------------------------------------
if [[ ! -d "$CONDA_ENVS_PATH/$ENV_NAME" ]]; then
    mamba create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip wheel
fi
# `mamba activate` needs the shell hook in non-interactive shells.
eval "$(mamba shell hook --shell bash 2>/dev/null || conda shell.bash hook)"
mamba activate "$ENV_NAME"
pip install -r requirements.txt
python -c "import highspy, numpy, pandas; print('CPU env ok:', highspy.__name__, numpy.__version__, pandas.__version__)"

# ---- data --------------------------------------------------------------------------------------------------------
if [[ $ENV_ONLY == 0 ]]; then
    MONTHS=$(python -c "from hiddenoptions.data import MONTHS; print(' '.join(MONTHS))")
    [[ -f census_pums/acs_pums_2020_2024_nyc_households.csv ]] || python scripts/fetch_census_pums.py
    for m in $MONTHS; do
        # The committed manifest records the sha256 the study used; the download overwrites the manifest, so read it first.
        expected=$(git show "HEAD:data/firstmover/raw/$m.manifest.json" 2>/dev/null \
                   | python -c "import json,sys; print(json.load(sys.stdin)['sha256'])" 2>/dev/null || echo "")
        [[ -f "data/firstmover/raw/$m.csv" ]] || python scripts/firstmover_download.py --month "$m"
        actual=$(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "data/firstmover/raw/$m.csv")
        if [[ -n "$expected" && "$expected" != "$actual" ]]; then
            echo "ERROR: data/firstmover/raw/$m.csv differs from the committed manifest (sha256 $actual, expected $expected)." >&2
            echo "       Upstream changed the file. Do not run experiments on it until the team decides; see slurm/README.md." >&2
            [[ "${ALLOW_DATA_DRIFT:-0}" == 1 ]] || exit 1
        fi
        [[ -f "data/firstmover/clean/firstmover_listings_$m.csv.gz" ]] || python scripts/firstmover_clean.py --month "$m"
    done
    python scripts/firstmover_coverage.py
    # Restore committed provenance files if the downloads rewrote them only with a new timestamp.
    git diff --stat -- data/firstmover || true
    echo "If the only manifest changes are downloaded_at_utc, run: git checkout -- data/firstmover/raw/*.manifest.json"
    echo "coverage_*.csv must be unchanged (git diff above); a change means the cleaned data differ from the study's."
    python -c "from hiddenoptions.data import load_renters; r = load_renters(); print('renters', len(r), 'recent movers', int(r.recent_mover.sum()))"
    python -m pytest tests -q -p no:cacheprovider
fi
mamba deactivate

# ---- GPU environment: vLLM for the open-model arms ---------------------------------------------------------------
if [[ $WITH_VLLM == 1 ]]; then
    if [[ ! -d "$CONDA_ENVS_PATH/$VLLM_ENV_NAME" ]]; then
        mamba create -y -n "$VLLM_ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip wheel
    fi
    mamba activate "$VLLM_ENV_NAME"
    pip install -r requirements.txt "vllm==$VLLM_VERSION" "huggingface_hub[cli]"
    export HF_HOME
    mkdir -p "$HF_HOME"
    # Pre-download the pinned weights here, so GPU jobs do not depend on network access from compute nodes.
    python - <<'EOF'
from huggingface_hub import snapshot_download
from experiments.open_model_recommend import ARM_MODELS
for family, (model, revision, _) in ARM_MODELS.items():
    path = snapshot_download(model, revision=revision)
    print(f"{family}: {model}@{revision} -> {path}")
EOF
    python -c "import vllm; print('vLLM', vllm.__version__)"
    mamba deactivate
fi
echo "done. Environments in $CONDA_ENVS_PATH"
