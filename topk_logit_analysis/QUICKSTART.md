# Quickstart - top-K logit predictability

Operational runbook. See [README.md](README.md) for schema and caveats.

## TL;DR

1. Edit `MODELS=(...)` in `topk_logit_analysis/run_analysis.sh`.
2. Run `sbatch topk_logit_analysis/run_analysis.sh`.
3. Open `$OUT_DIR/report.html` and read per-source unpredictable rate,
   pairwise agreement/Jaccard, target-rank Spearman, then token grids.

## 1. Pick Candidates

Hard precondition: every model must share a vocabulary compatible with the
sampled validation set. The scorer checks the model vocab before loading
weights and fails fast on mismatch.

Good candidates:

- Later checkpoints from the same trainee/reference family.
- Any internally trained Nanotron or HF model that shares the tokenizer.

Not compatible without retokenization or remapping:

- Generic Hub models with a different vocab.
- Off-the-shelf LLM360/K2 if the validation set was sampled with the 250k
  tokenizer.

## 2. Configure

Open `topk_logit_analysis/run_analysis.sh`. The main edit is the `MODELS=()`
array near the top. Format:

```bash
"tag|path-or-hub-id|backend"
```

Backend must be `nanotron`, `hf`, or `auto`. Tags must match
`[A-Za-z0-9._-]+` because they are used in filenames.

Example:

```bash
MODELS=(
  "ref-step01000|${CKPT_BASE}/<run>/1000|nanotron"
  "ref-step10000|${CKPT_BASE}/<run>/10000|nanotron"
  "ref-step32000|${CKPT_BASE}/<run>/32000|nanotron"
)
```

Useful env vars:

```bash
OUT_ROOT=$PWD/topk_logit_analysis
OUT_DIR=$OUT_ROOT/<timestamp>
SOURCE_ROOT=/mnt/weka/.../training/tokenized/train
NUM_DOCS_PER_SOURCE=5
TOP_K_LOGITS=50
MAX_TOKENS_PER_DOC=512
```

## 3. Submit

From the repo root:

```bash
sbatch topk_logit_analysis/run_analysis.sh
```

Or with overrides:

```bash
TOP_K_LOGITS=100 \
NUM_DOCS_PER_SOURCE=10 \
OUT_DIR=$PWD/topk_logit_analysis/2026-06-top100 \
  sbatch topk_logit_analysis/run_analysis.sh
```

Tail the log:

```bash
tail -f topk_logit_analysis-<job_id>.log
```

The job reuses an existing `validation.pt`, skips existing `scored_<tag>.pt`
files, and regenerates `report.html`.

## 4. Read The Report

Read it top-down:

1. Per-source unpredictable rate: higher means more true labels fall outside
   top-K for that source/model.
2. Pairwise unpredictable-mask Jaccard: overlap of the exact not-in-top-K
   token sets at the selected K.
3. Pairwise predictable/unpredictable agreement: direct boolean agreement on
   in-top-K vs outside-top-K.
4. Pairwise target-rank Spearman: whether models order token predictability
   similarly, independent of K.
5. Positional unpredictable profile: whether outside-top-K tokens cluster at
   specific packed-sequence positions.
6. Per-document grids: inspect whether disagreements are on content tokens,
   formatting, boundaries, code identifiers, math answers, etc.

## 5. Try Another K Without Rescoring

Because `score_topk.py` persists exact ranks, rerun only the visualizer:

```bash
python -u topk_logit_analysis/visualize.py \
  --scored "$OUT_DIR"/scored_*.pt \
  --top-k 100 \
  --out "$OUT_DIR"/report_top100.html
```

## Common Failures

| symptom | cause | fix |
|---|---|---|
| `Vocab mismatch` | candidate does not cover validation token ids | remove it or resample with a compatible tokenizer |
| `tag must match` | tag contains spaces, slashes, or shell-hostile chars | rename the `MODELS` tag |
| scoring fails for one model | OOM, missing checkpoint, or HF download issue | check the SLURM log; other successful models still report |
| fewer than two models succeed | nothing useful to compare | fix candidate paths or run one model manually first |
| report token text is `<id:N>` | tokenizer was omitted or unavailable | pass `--tokenizer` to `visualize.py` if decoded text is needed |

## What This Does Not Answer

This is a static screening report. It does not prove that a reference model or
top-K threshold improves training; that still needs an ablation run.
