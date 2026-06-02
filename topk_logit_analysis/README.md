# Top-K logit predictability analysis

Compare whether candidate reference models put the true next-token label
inside their top-K logits on the same sampled validation sequences. This is a
screening tool for a top-K predictability rule, not a high-loss dropping
analysis.

The scoring pass stores exact per-token target rank. A valid label with
`target_rank <= K` is predictable; a valid label with `target_rank > K` is
unpredictable. Ignored boundary labels keep `target_rank == 0`. Because ranks
are persisted, you can regenerate reports for different K values without
rescoring.

See [QUICKSTART.md](QUICKSTART.md) for the operational runbook.

## What It Produces

`report.html` leads with decision-grade signals for the top-K rule:

- Per-source unpredictable rate: fraction of valid labels outside top-K for
  each source and model.
- Pairwise unpredictable-mask Jaccard: overlap between models' not-in-top-K
  token sets at the selected K.
- Pairwise predictable/unpredictable agreement: direct agreement on the
  boolean top-K decision over valid labels.
- Pairwise target-rank Spearman correlation: rank agreement independent of the
  chosen K.
- Positional unpredictable-token histogram: where outside-top-K tokens appear
  in packed sequences.
- Top-disagreement sequence leaderboard and per-document token grids.

Grid colors:

- Green: predictable, true label is inside top-K.
- Red: unpredictable, true label is outside top-K.
- Gray: ignored by `label_mask`, typically packed-document boundaries.
- Yellow token text: doc-boundary label position.

## Output Schema

Each `scored_<tag>.pt` is a pickle of:

```python
{
    "records": [
        {
            "source":                 str,
            "sample_idx":             int,
            "input_ids":              np.int64 [seq_len + 1],
            "positions":              np.int64 [seq_len + 1],
            "label_ids":              np.int64 [seq_len],
            "label_mask":             np.bool_ [seq_len],
            "per_token_target_rank":  np.int32 [seq_len],   # 1=top logit, 0 where ignored
            "per_token_in_topk":      np.bool_ [seq_len],   # for the scorer's --top-k
            "per_token_top1_id":      np.int64 [seq_len],   # -1 where ignored
            "per_token_top1_prob":    np.float32 [seq_len],
            "per_token_target_logit": np.float32 [seq_len],
            "per_token_target_prob":  np.float32 [seq_len],
            "model_tag":              str,
            "backend":                "nanotron" | "hf",
        },
        ...
    ],
    "model_tag":       str,
    "backend":         str,
    "checkpoint":      str,
    "tokenizer_path":  str,
    "vocab_size":      int,
    "validation_path": str,
    "top_k_logits":    int,
}
```

## Layout

```text
topk_logit_analysis/
  sample_validation.py   # samples fixed packed token sequences
  score_topk.py          # records target ranks and top-1/target diagnostics
  visualize.py           # renders HTML comparison for a chosen K
  run_analysis.sh        # SLURM entry point
  README.md
  QUICKSTART.md
```

## Backends

`backend=nanotron` expects a directory containing `config.yaml` and `model/`.
It uses the same Nanotron loading/forward pattern as the loss-mask analysis
template and assumes TP=PP=DP=1. Qwen2 checkpoints are forwarded with packed
`position_ids`, preserving doc-boundary attention masking. Llama and Starcoder2
checkpoints are forwarded through their `input_mask` interface, which does not
represent packed-document boundaries in this checkout.

`backend=hf` expects a local Hugging Face model directory or Hub id and uses
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. Packed
sequences are split into per-document chunks at `positions == 0`, so HF
models do not attend across packed-document boundaries.

Both backends feed `input_ids[:T]`, score labels `input_ids[1:T+1]`, and use
`positions[1:T+1] != 0` as `label_mask`.

## Same-Vocab Requirement

Every candidate must use a vocabulary compatible with the sampled validation
tokens. `score_topk.py` checks the maximum token id against the model vocab
before loading weights, using Nanotron `config.yaml` or HF `AutoConfig`.

## Knobs

| variable | default | meaning |
|---|---:|---|
| `OUT_ROOT` | `<repo>/topk_logit_analysis` | parent directory for timestamped output runs |
| `OUT_DIR` | `$OUT_ROOT/<timestamp>` | exact output directory |
| `SOURCE_ROOT` | cluster tokenized train path | parent directory with one folder per tokenized source |
| `NUM_DOCS_PER_SOURCE` | `5` | packed sequences sampled per source |
| `SEQUENCE_LENGTH` | `8192` | training sequence length |
| `TOKEN_SIZE` | `4` | bytes per token in `.ds` files |
| `TOP_K_LOGITS` | `50` | K used by scorer logging and report generation |
| `MAX_TOKENS_PER_DOC` | `512` | rendered tokens per document slice |

## Caveats

- This report answers whether reference models consider tokens predictable at
  a fixed K. It does not answer whether a reference will improve training.
- Exact ranks require comparing each target logit against the full vocab. Peak
  memory is still dominated by `[B, T, V]` logits. Use
  `--micro-batch-size 1` for smaller GPUs.
- HF per-doc chunking avoids cross-document attention, but a later frozen
  reference integrated into Nanotron training still needs its own attention
  masking path.
