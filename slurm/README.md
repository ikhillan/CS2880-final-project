# Running on FASRC Cannon

These scripts run the experiments on Harvard's FASRC Cannon cluster: CPU job arrays for the audit runs and H200 jobs for
the open-model arms. They have **not been run on the cluster yet**. Do the checks in "Before the first real run" first.

| File | What it does |
|---|---|
| `setup_env.sh` | Builds the CPU environment (and, with `--with-vllm`, the GPU environment plus the pinned model weights). Rebuilds the git-ignored data: Census PUMS, every FirstMover month in `hiddenoptions.data.MONTHS` (downloaded, then cleaned) and the coverage files. Checks each raw file against the sha256 in its committed manifest, then runs the tests. |
| `run_array.sbatch` | One CPU array task, which runs `python -m experiments.run <config> --workers 1 --shard $SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS`. |
| `submit_array.sh` | Submits a config as an array of N shards. |
| `gpu_openmodels.sbatch` | One H200 per array task, running `experiments/open_model_recommend.py` (Qwen or Mistral). |

## Placeholders to fill in

You can set each of these as an environment variable or edit the default in the script.

| Setting | Where | What to put |
|---|---|---|
| Slurm account | `FASRC_ACCOUNT` for `submit_array.sh`, or `--account=` on `sbatch` (commented `##SBATCH --account=YOUR_LAB` line in both sbatch files) | Your lab's fairshare account. `sshare -U` lists yours. The docs say to pass `--account` if you belong to more than one lab. |
| `ENV_ROOT` | `setup_env.sh`, `run_array.sbatch`, `gpu_openmodels.sbatch` | A folder under your lab directory, e.g. `/n/holylabs/<LAB>/Lab/$USER/conda`. FASRC advises against keeping environments in `~/.conda`, because the home filesystem is slow. |
| `HF_HOME` | `setup_env.sh --with-vllm`, `gpu_openmodels.sbatch` | The model-weight cache, about 115 GB for both models. Suggested: `$SCRATCH/<LAB>/Lab/$USER/hf`. Netscratch is purged, so re-run `setup_env.sh --with-vllm` if the weights disappear. |
| Partition | `-p` in the sbatch files, or `FASRC_PARTITION` | Defaults are `serial_requeue` (CPU) and `gpu_h200` (GPU). Both appear in the FASRC partition table. See "Verified vs unverified". |
| `PYTHON_MODULE` | all scripts | Defaults to `python`. Pin a version (e.g. `python/3.14.10-fasrc01`, or a `Miniforge3/...` module) after checking `module spider python`. |
| `VLLM_VERSION` | `setup_env.sh` | Defaults to `0.10.1.1`. Mistral Small 3.1 needs at least 0.8.1 (model card). Pin one version for the whole study. The open-model script records the installed version and refuses to mix versions in one work directory. |
| `-t` for GPU jobs | `gpu_openmodels.sbatch` | A placeholder (12 h) until the 500-request profile gives requests per GPU-hour. |

## Where to put things

FASRC says home and lab directories are not suited to I/O-heavy work or large numbers of jobs, and recommends scratch.

- **Repository and `results/`:** clone into netscratch, e.g. `$SCRATCH/<LAB>/Lab/$USER/CS2880-final-project`. The data is about 650 MB, and every shard appends to its own JSONL file.
- **Purge rule:** netscratch deletes files older than 90 days, and the docs say this drops to **60 days from the December maintenance**. Copy `results/` to the lab directory after each stage, e.g. `rsync -a results/ /n/holylabs/<LAB>/Lab/results/`.
- **Environments:** keep them in the lab directory (`ENV_ROOT`), not in home.

## Usage

```bash
# 1. Once: environment and data, inside an interactive job (FASRC builds environments inside salloc)
salloc --partition test --nodes=1 --cpus-per-task=2 --mem=8GB --time=0-02:00:00
export ENV_ROOT=/n/holylabs/<LAB>/Lab/$USER/conda HF_HOME=$SCRATCH/<LAB>/Lab/$USER/hf
bash slurm/setup_env.sh                 # CPU env, data, tests
bash slurm/setup_env.sh --with-vllm --env-only   # GPU env + pinned weights (run on a node with enough disk and network)
exit

# 2. CPU array: N shards of a config. Submit from the repo root with no environment active.
FASRC_ACCOUNT=<lab> bash slurm/submit_array.sh configs/stage3_protection.json 300
FASRC_ACCOUNT=<lab> bash slurm/submit_array.sh configs/stage3_protection.json 300 --test-only   # scheduling check only
squeue -u $USER ; sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
# Rerunning the same command after timeouts or failures resumes: finished (market, exposure, method) rows are skipped.

# 3. Analysis: experiments/analyze.py already reads results/<name>/shards/{runs,markets}-*.jsonl. No merge step is needed.
python -m experiments.analyze stage3_protection --treated claude_k5 --control uniform_k5

# 4. Open models: create the Slurm log directory before sbatch opens its log files.
mkdir -p logs
# Profile first (protocol section 8: the same fixed 500 requests per card type), then the full run.
sbatch --account=<lab> --array=0 \
  --export=ALL,CONFIG=configs/stage1_pilot.json,FAMILY=qwen,NUM_SHARDS=1,EXTRA="--limit 500 --tag profile_h200 --no-merge" \
  slurm/gpu_openmodels.sbatch
#    read results/llm/stage1_pilot/open_qwen_profile_h200/state.json (req/s, tokens/s, statuses, nvidia-smi), then size -t and N
sbatch --account=<lab> --array=0-3 --export=ALL,CONFIG=configs/stage2_main.json,FAMILY=qwen,NUM_SHARDS=4 slurm/gpu_openmodels.sbatch
#    once every task has finished (or chain it with --dependency=afterok:<jobid>):
python -m experiments.open_model_recommend configs/stage2_main.json --family qwen --merge-only
#    repeat with FAMILY=mistral. The merged recs file is the config's llm_results; then run the CPU array.
```

Run `python -m experiments.open_model_recommend configs/stage1_pilot.json --family qwen --limit 500 --dry-run` first (no GPU, no vLLM). It prints
request counts and prompt lengths.

Use the same development config and limit on each GPU type when comparing throughput. For the one-block-per-design smoke sample, add `--seeds 1` to each profiling command. Qwen and Mistral then saw the same 500 renters, candidate lists and prompt texts. Sorting their provider-specific request IDs in the old sampler gave only 228 shared cases. [Input comparison](development-profile-check.json) A different tag gives each card its own profiling directory. A sharded production merge verifies every shard receipt, the expected request identities, and the model and generation settings. If the generation used `--seeds` or `--limit`, pass the same selection to `--merge-only`. Missing or mismatched shards must complete under their original settings before publishing results.

For the Stage 2 primary comparison the original five-borough planning rule gave **400 seed blocks**. The separate 2025 replication remains at 100. The calculation and denominator coverage are in [the pilot report](../findings/claude-haiku-pilot/paired-outcomes.md). The profiling command above uses development renters.

**Order matters for configs with llm arms.** `experiments.run` skips an llm exposure for a market with no saved replies
("skipped (no saved recommendations)"). Merge every model's replies before submitting the CPU array.

## Why these defaults

The CPU defaults come from the measured timings in `WIP.md`:

- **Solve times:** an exact solve takes about 1 s (median) at 64 per side and under 0.2 s at 24. Timeouts at the 900 s cap are rare (3 of 1,728 runs).
- **Stage-3 total:** about 100 core-hours estimated, 250 planned, over 1,200 markets. That averages 5–13 core-minutes per market.

| Setting | Value | Reason |
|---|---|---|
| `-c 1` | 1 core per task | HiGHS is pinned to one thread (`hiddenoptions/audit.py`), and the runner pins BLAS/OpenMP to one thread. The protocol (section 8) asks for one thread per HiGHS worker. |
| `--mem=2G` | 2 GB | One shard peaked at about 220 MB RSS on a laptop with the three development months loaded. Held-out configs load a few more months. |
| `-t 0-08:00` | 8 h | With 300 shards (about 4 markets per task), the typical task takes well under an hour. The worst single 64-per-side market costs 6 exposures × 3 exact objectives × 900 s = 4.5 h. A timed-out task loses only its unfinished market. |
| N | ~4 markets per task | FASRC asks for jobs of at least 5–10 minutes and a cap of about 1,000 submitted jobs. Array size is capped at 10,000. |
| `serial_requeue` | partition | FASRC's partition for small single-core array tasks under a day. Preemption is harmless here because the runner resumes, and `--open-mode=append` keeps the requeue messages. Use `shared` or `sapphire` (3-day limit, no preemption) if requeues waste too much time. |
| GPU: one H200 per replica | `--gres=gpu:1` | Qwen2.5-32B in bf16 is about 65 GB and Mistral-Small-3.1-24B about 48 GB. Each fits one 141 GB H200 (the protocol prefers single-GPU replicas). Keep one card type per model for the primary comparison. |

## Verified vs unverified (checked against the FASRC docs on 2026-11-13)

**Verified in the docs:**

- **Partitions:** `serial_requeue` (default partition; for single-core jobs and small arrays under a day; jobs may be requeued; GPU jobs rejected), `shared`, `sapphire`, `test` and `gpu_test`.
- **H200 partition:** `gpu_h200` has 4 × H200 per node, a 3-day limit, and takes `--gres=gpu:n` with n = 1–4.
- **Other GPU partitions:** `gpu` (A100) and `gpu_requeue`.
- **Defaults:** 1 core, 100 MB and `serial_requeue`, with no default time ("users must always declare time").
- **Array limits:** max array size 10,000, and 10,100 jobs per user. The docs recommend about 1,000 jobs and runs of at least 5–10 minutes.
- **Filename patterns:** `%A` and `%a`.
- **Mail:** `--mail-type` set to FAIL or END, not ALL.
- **Environments:**
  - Use `module load python` (Miniforge, conda-forge) and `mamba create` / `mamba activate`.
  - Keep `CONDA_ENVS_PATH` and `CONDA_PKGS_DIRS` in a lab directory, never pip outside an environment, and don't submit jobs from inside an active environment.
  - `repo.anaconda.com` is blocked.
- **Storage:**
  - `$SCRATCH` resolves to `/n/netscratch`, with 50 TB per lab and the 90-day purge (60 days from the December maintenance).
  - Lab directories are 4 TiB under `/n/holylabs`.
  - Home is 100 GB.
  - Node-local `/scratch` lasts only for the job.

**Unverified (check before relying on it):**

- **Account and paths:** your lab's account name and exact lab and netscratch paths.
- **Shell hook:** whether `mamba activate` works in batch scripts without the shell hook. The scripts run `eval "$(mamba shell hook --shell bash)"`, with a conda fallback.
- **Python version:** whether `python=3.12` is the best choice next to the module's default Python. vLLM wheels lag the newest Python.
- **CUDA module:** whether vLLM's bundled CUDA runtime works with the node driver, or whether you need `module load cuda/<version>`.
- **Network:** whether compute nodes can reach Hugging Face. The GPU job sets `HF_HUB_OFFLINE=1` and expects the weights already downloaded by `setup_env.sh --with-vllm`.
- **H200 counts:** a 2025 training deck lists 24 `gpu_h200` nodes, while the Running Jobs table lists 22. Check `scontrol show partition gpu_h200` (its TRES field lists the GPU types).
- **H100 access:** the docs list an H100 GPU type, but no public partition with H100s. Some GPU types are reachable only through `gpu_requeue`. If the team has H100 access through a lab or FAS partition, add it for the protocol's H100/H200 profile.
- **GPU CPU and memory:** `-c 8 --mem=64G` per GPU job is my guess (the docs give no ratio). Check with `seff <jobid>` after the profile.
- **FAS group partitions:** none are assumed.

## Checks before the first real run

1. **Data:** `setup_env.sh` must finish with all tests passing. `git diff data/firstmover` should show at most new `downloaded_at_utc` timestamps in the manifests, and `coverage_*.csv` must be unchanged. A sha256 mismatch means upstream changed a month: stop and ask the team. (`ALLOW_DATA_DRIFT=1` overrides, but don't use it for study runs.)
2. **Same results as on a laptop:** run a few shards of a development config interactively and compare them with laptop results. Example: `python -m experiments.run configs/stage1_solver_timing.json --workers 1 --shard 0 --num-shards 96`, then compare its `runs-0000.jsonl` costs and `market_id`s.
3. **Scheduling:** `sbatch --test-only ...` with your account and partition.
4. **Small array first:** e.g. `submit_array.sh <config> 300 --array=0-4`. A later `--array` overrides the helper's. Check `sacct` MaxRSS and Elapsed, then submit the rest.
5. **Open models:**
   - Run `--dry-run` on the cluster, then the 500-request profile, before any full run.
   - Look at `stats.status` (should be almost all `ok`) and the throughput.
   - Check `pinned.engine_version` and `pinned.json_constraint` in the state file.
6. **Held-out months:** keep them unopened until the design is frozen (`WIP.md`). Only submit stage-2/3 configs after that.

## Sources (FASRC documentation)

- Running Jobs: sbatch directives, partition table, GPU types, arrays, fairshare. https://docs.rc.fas.harvard.edu/kb/running-jobs/
- Submitting Large Numbers of Jobs: arrays, job-count etiquette, minimum run times. https://docs.rc.fas.harvard.edu/kb/submitting-large-numbers-of-jobs/
- Python package installation: `module load python`, mamba environments, `CONDA_ENVS_PATH`, pip rules. https://docs.rc.fas.harvard.edu/kb/python-package-installation/
- GPU computing on the cluster: `--gpus`, `gpu_test`, `gpu_requeue`, CUDA modules. https://docs.rc.fas.harvard.edu/kb/gpgpu-computing-on-the-cluster/
- Home and lab directories: quotas, holylabs, "use scratch for I/O-heavy work". https://docs.rc.fas.harvard.edu/kb/cluster-storage/
- Scratch policy: `$SCRATCH`, `/n/netscratch`, 50 TB, the 90/60-day purge, node-local `/scratch`. https://docs.rc.fas.harvard.edu/kb/policy-scratch/
- FASRC training materials (2025 decks with partition and GPU counts). https://docs.rc.fas.harvard.edu/kb/training-materials/

Model facts come from the model cards at the pinned revisions:

- https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
