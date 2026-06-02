#!/bin/bash
#SBATCH --job-name=topk_logit_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Edit partition/qos to match your cluster; `lowprio` is project-specific.
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --wait-all-nodes=1
#SBATCH --output=%x-%j.log
# Intentionally NOT --exclusive: a 1-GPU job doesn't need the whole node and
# `--exclusive` on `lowprio` can balloon queue wait into hours.
#
# Top-K logit predictability comparison across multiple reference checkpoints.
#
# Edit the MODELS array below (one entry per checkpoint), then submit with:
#     sbatch run_analysis.sh
#
# Each MODELS entry uses the format
#     tag|path-or-hub-id|backend
# where backend is "nanotron" or "hf" ("auto" also works for nanotron dirs).
#
# Re-runnability: pass OUT_DIR=<previous-dir> to re-use an existing run's
# validation set and skip already-scored models. Lets you add a candidate
# reference model without re-scoring the others.

set -e -x

# ---------------------------------------------------------------------------
# Configure: where to put outputs and which models to compare
# ---------------------------------------------------------------------------

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/topk_logit_analysis/sample_validation.py" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR/topk_logit_analysis"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/topk_logit_analysis}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$(date +%Y%m%dT%H%M%S)}"
CKPT_BASE="${CKPT_BASE:-/mnt/weka/shrd/k2m/omkar.pangarkar/tokenmix-checkpoints-1p5B}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/weka/shrd/k2m/omkar.pangarkar/training/tokenized/train}"

# ---------------------------------------------------------------------------
# IMPORTANT: same-vocab precondition
#
# Every candidate must use the same 250k LLM360 vocab as the validation set
# (ilikejson-and-python-250k-tokenizer). The script verifies this BEFORE
# loading model weights and aborts with a clear error on mismatch.
#
# This rules out off-the-shelf LLM360/K2 (32k vocab, LLaMA-style) and any
# other HF model that wasn't trained on the same tokenizer. Use:
#   - additional checkpoints from your own current trainee at different steps
#   - any other internally-trained model that shares the 250k vocab
# ---------------------------------------------------------------------------

# Candidates being compared (2026-05 run).
# Baseline = current Reuben 1.5B trainee at step 143051. Colorful-Factorial
# is the other 1.5B final checkpoint. BBQ 4B/8B/32B are candidate reference
# models from the k2m project (expected to share the 250k LLM360 vocab; script
# will verify before loading weights).
# backend=auto picks nanotron if config.yaml exists in the dir, else HF.
MODELS=(
  "trainee-step143051|/mnt/weka/shrd/k2m/seungwook.han/pt-mask-ablation-ckpts/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_cyclic_reuben/143051|auto"
  "colorful-factorial-step143051|/mnt/weka/shrd/k2m/seungwook.han/pt-mask-ablation-ckpts/tokenmix-checkpoints-1p5B/tokenmix_ablation_1p5B_mix_bbq_all_colorful_factorial/143051|auto"
  "bbq-4b-pretrain|/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-4b-pretrain-final|auto"
  "bbq-8b-pretrain|/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-8b-pretrain-final|auto"
  "bbq-32b-pretrain|/mnt/weka/shrd/k2m/suqi.sun/bbq_image/bbq-32b-pretrain-final|auto"
)

NUM_DOCS_PER_SOURCE="${NUM_DOCS_PER_SOURCE:-5}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8192}"
TOP_K_LOGITS="${TOP_K_LOGITS:-50}"
MAX_TOKENS_PER_DOC="${MAX_TOKENS_PER_DOC:-512}"
TOKEN_SIZE="${TOKEN_SIZE:-4}"

# Basic validation of MODELS entries (tag safe, backend known).
TAG_RE='^[A-Za-z0-9._-]+$'
for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG _CKPT BACKEND <<< "$entry"
  if [[ ! "$TAG" =~ $TAG_RE ]]; then
    echo "==> ERROR: tag '$TAG' must match $TAG_RE" >&2
    exit 1
  fi
  if [[ "$BACKEND" != "nanotron" && "$BACKEND" != "hf" && "$BACKEND" != "auto" ]]; then
    echo "==> ERROR: backend '$BACKEND' for tag '$TAG' must be one of nanotron|hf|auto" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton_cache}"
PY_VENDOR_DIR="${PY_VENDOR_DIR:-$HOME/.local/topk_logit_analysis_py}"
if [[ -d "$PY_VENDOR_DIR" ]]; then
  export PYTHONPATH="$PY_VENDOR_DIR:${PYTHONPATH:-}"
fi

mkdir -p "$OUT_DIR"
mkdir -p "$TRITON_CACHE_DIR"
VAL_PATH="$OUT_DIR/validation.pt"
REPORT_PATH="$OUT_DIR/report.html"

# ---------------------------------------------------------------------------
# 1) Sample validation set (CPU only)
# ---------------------------------------------------------------------------

if [[ -f "$VAL_PATH" ]]; then
  echo "==> reusing existing validation set at $VAL_PATH (delete it to re-sample)"
else
  python -u "$SCRIPT_DIR/sample_validation.py" \
    --source-root "$SOURCE_ROOT" \
    --num-docs-per-source "$NUM_DOCS_PER_SOURCE" \
    --sequence-length "$SEQUENCE_LENGTH" \
    --token-size "$TOKEN_SIZE" \
    --out "$VAL_PATH"
fi

# ---------------------------------------------------------------------------
# 2) Score each model on the same validation set
# ---------------------------------------------------------------------------

SCORED_FILES=()
FAILED_MODELS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r TAG CKPT BACKEND <<< "$entry"
  SCORED="$OUT_DIR/scored_${TAG}.pt"

  if [[ -f "$SCORED" ]]; then
    echo "==> $TAG already scored at $SCORED; skipping"
    SCORED_FILES+=("$SCORED")
    continue
  fi

  echo "==> scoring $TAG (backend=$BACKEND) from $CKPT"

  # Don't abort the whole job on a single model failure; we still want a
  # partial report with whichever models succeeded.
  set +e
  if [[ "$BACKEND" == "nanotron" || "$BACKEND" == "auto" ]]; then
    # nanotron needs torch.distributed initialised even at TP=PP=DP=1
    torchrun --nproc_per_node=1 "$SCRIPT_DIR/score_topk.py" \
      --validation "$VAL_PATH" \
      --checkpoint "$CKPT" \
      --backend "$BACKEND" \
      --model-tag "$TAG" \
      --top-k "$TOP_K_LOGITS" \
      --out "$SCORED"
  else
    python -u "$SCRIPT_DIR/score_topk.py" \
      --validation "$VAL_PATH" \
      --checkpoint "$CKPT" \
      --backend "$BACKEND" \
      --model-tag "$TAG" \
      --top-k "$TOP_K_LOGITS" \
      --out "$SCORED"
  fi
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "==> WARNING: scoring $TAG failed with rc=$rc; continuing"
    FAILED_MODELS+=("$TAG")
  elif [[ -f "$SCORED" ]]; then
    SCORED_FILES+=("$SCORED")
  else
    echo "==> WARNING: scoring $TAG exited 0 but produced no output; skipping"
    FAILED_MODELS+=("$TAG")
  fi
done

if [[ ${#SCORED_FILES[@]} -lt 2 ]]; then
  echo "==> ERROR: fewer than 2 models succeeded; nothing to compare. Failed: ${FAILED_MODELS[*]:-(none)}"
  exit 1
fi
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "==> NOTE: report will exclude failed models: ${FAILED_MODELS[*]}"
fi

# ---------------------------------------------------------------------------
# 3) Render report
# ---------------------------------------------------------------------------

python -u "$SCRIPT_DIR/visualize.py" \
  --scored "${SCORED_FILES[@]}" \
  --top-k "$TOP_K_LOGITS" \
  --max-tokens-per-doc "$MAX_TOKENS_PER_DOC" \
  --out "$REPORT_PATH"

echo "Done. Open: $REPORT_PATH"
