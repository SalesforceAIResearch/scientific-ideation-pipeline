# Insight Synthesis — GRPO-v6-SI Reward/Training + Impact/Diversity/Creative-Utility Eval

This is a code release repo for Training Reasoning Models to Generate Diverse Research Ideas. 
This code has the training objective along with the evaluation harness that scores impact, diversity, and
creative utility (CU) on generated ideas.

## At a glance

| Component | Purpose | Entry point |
| --- | --- | --- |
| GRPO-v6-SI reward | impact x distance + card-only length penalty, SI (4-field) card schema | `grpo_training/rewards/f9b_impdist_len_v6_si_bedrock.py` |
| SciJudge impact judge | pairwise citation-impact scoring | `grpo_training/common/scijudge.py` |
| Eval harness | impact / diversity scoring over a set of generations | `grpo_training/eval/run_all_evals.py` |
| Pool diversity + creative utility | pool-level QD tradeoff + CU metric/plot | `qd_tradeoff/score_pool_diversity.py`, `qd_tradeoff/plot_pool_creative_utility.py` |
| GRPO-v6-SI training | end-to-end training job | `scripts/train_grpo_v6_si.sh` (no scheduler) or `grpo_training/train/launch_grpo_v6_si_onenode_h200.sbatch` (SLURM) |

## Requirements

- Python 3.10, an NVIDIA GPU node for training (this was trained on 8x H200
  nodes; smaller GPUs are untested).
- **verl** is not vendored here — it's a pinned external clone with a small
  patch on top:
  ```bash
  git clone https://github.com/verl-project/verl.git grpo_training/vendor/verl
  cd grpo_training/vendor/verl && git checkout 86e8123643c982343b11fa21dc1df96ab07ced3d
  git apply ../vendor_patches/verl_custom.patch
  ```
  (see `grpo_training/vendor_patches/README.md` for details, or just run
  `bash grpo_training/setup/setup_env.sh`, which does this for you.)
- **giants** (data prep dependency, for the public GiantsBench-train dataset):
  cloned automatically by `scripts/prepare_rl_data.sh`.

## Step 1 — Environment setup

```bash
cp .env.example .env
# fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / HF_TOKEN / WANDB_API_KEY as needed
source .env
bash grpo_training/setup/setup_env.sh
```

## Step 2 — Prepare data

```bash
bash scripts/prepare_rl_data.sh
```

This builds the GiantsBench-derived RL parquet. Our own dataset/checkpoint
Hugging Face repo (`HF_DATASET_REPO` in `.env`) isn't public yet — set it once
it is; until then this step only prepares the GiantsBench data.

## Step 3 — Train GRPO-v6-SI

This is the only reward/training path in this release.

```bash
# no scheduler:
BASE_MODEL=/path/to/sft/checkpoint GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_grpo_v6_si.sh

# or on SLURM:
sbatch --partition=<yours> --account=<yours> \
  grpo_training/train/launch_grpo_v6_si_onenode_h200.sbatch
```

Both co-locate a self-hosted SciJudge (impact judge) server on a few GPUs and
route the distance judge through AWS Bedrock (`openai.gpt-oss-120b-1:0`); see
the header comment in the `.sbatch` file for the full design rationale (GPU
split, batch-size divisibility, why co-location, why Bedrock for distance).

You'll need an SI-format SFT checkpoint as `BASE_MODEL` — a v0-format policy
under the SI reward hits the format gate on every rollout. See the comment
block in the launcher for what that checkpoint needs to look like.

## Step 4 — Run the eval harness (impact / diversity / creative utility)

```bash
bash scripts/run_eval.sh --dataset preds.jsonl --out-prefix runs/my_model --input-format ideation-si
```

Produces `<out-prefix>.evals.jsonl` (per-idea impact/diversity), `.pool_diversity.jsonl`
(pool-level pairwise distances), and `.cu.png`/`.cu.json` (the creative-utility
curve). `run_all_evals.py` also supports soundness/feasibility/grounding
judges (`--help` for the full list) that are outside this release's documented
pipeline but present in the code if you want to experiment.

## Self-hosted judges

The diversity judge (`gpt-oss-120b`) and SciJudge (impact) default to
self-hosted vLLM endpoints (`localhost:8001` / `:8002`). To route a judge
through a hosted provider instead, set `SCIJUDGE_MODEL`/`SCIJUDGE_BASE_URL` (or
the equivalent for the diversity judge) to a `gpt-*`/`claude-*`/`gemini-*` model
name — see `grpo_training/config.sh` and `grpo_training/common/gateway.py`.

## What's NOT in this release

- Checkpoints/datasets (HF repo TBD, see Step 2).
- The plain (non-SI) v6 reward/training path — this release only documents the
  SI variant, though the underlying v6 math (`f9b_impdist_len_v6.py`) is
  present as a dependency.
- v7+/v9+/v10+/v11 reward variants (soundness, new-grounding, crispness,
  cross-check composites).
- The DARLING baseline comparison arm.
- A cross-run comparison report generator (the internal one is tied to our own
  run-tracking vocabulary; `scripts/run_eval.sh` produces single-run output only).

## License

Apache License 2.0 — see `LICENSE` and `NOTICE.md`.
