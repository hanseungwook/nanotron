"""Score reference-model top-K predictability on a validation set.

Loads a single checkpoint (either a Nanotron training checkpoint or a Hugging
Face model directory / hub id), runs forward over every sequence in the
validation pickle produced by `sample_validation.py`, and writes a new pickle
with per-token target rank and small probability diagnostics. The target rank
is the screening signal: a token is predictable for K when its true next-token
label is within the model's top-K logits. Cross-entropy is not computed or used.

The script handles two backends:

  - `nanotron`: checkpoint folder containing `config.yaml` plus `model/`. Uses
    nanotron's `CONFIG_TO_MODEL_CLASS` + `load_weights`, so it works for any
    architecture the trainer already supports (Qwen2 / Llama / Starcoder2).
    Forward runs on the full packed sequence with `_use_doc_masking=True`,
    matching the training-time attention regime.

  - `hf`: a HuggingFace model directory or hub id. Loaded with transformers
    `AutoModelForCausalLM`. To match the trainee's doc-masked attention regime
    as closely as possible without cu_seqlens plumbing, we split each packed
    sequence into per-doc chunks using `positions == 0` boundaries and
    forward each chunk independently with HF's default causal attention.
    Resulting per-token ranks are reassembled into the original layout.

In both cases we feed the model `input_ids[:T]` (length T = seq_len, NOT
T+1), so the model never sees more positions than it was trained on. Labels
are `input_ids[1:T+1]` and `label_mask` is `positions[1:T+1] != 0`.

Output schema (one record per sequence):

    {
        "source":              str,
        "sample_idx":          int,
        "input_ids":           np.int64  [seq_len + 1]  (raw, as sampled)
        "positions":           np.int64  [seq_len + 1]
        "label_ids":           np.int64  [seq_len]
        "label_mask":          np.bool_  [seq_len]
        "per_token_target_rank": np.int32 [seq_len]    # 1=top logit, 0 where masked
        "per_token_in_topk":     np.bool_ [seq_len]    # replay convenience for --top-k
        "per_token_top1_id":     np.int64 [seq_len]    # -1 where masked
        "per_token_top1_prob":   np.float32 [seq_len]  # 0 where masked
        "per_token_target_logit": np.float32 [seq_len] # 0 where masked
        "per_token_target_prob": np.float32 [seq_len]  # 0 where masked
        "model_tag":           str,
        "backend":             "nanotron" | "hf",
    }

Run on a single GPU with `torchrun --nproc_per_node=1` for the nanotron
backend (it needs torch.distributed init even at TP=PP=DP=1). The HF backend
can run under plain `python`.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--validation",
        type=Path,
        required=True,
        help="Pickle produced by sample_validation.py.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a nanotron checkpoint folder OR a HuggingFace model dir / hub id.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "nanotron", "hf"],
        default="auto",
        help="Loader backend. 'auto' picks 'nanotron' if config.yaml exists, else 'hf'.",
    )
    p.add_argument(
        "--model-tag",
        type=str,
        required=True,
        help="Short identifier (e.g. 'qwen2-1p5b-step20k', 'k2-base'). Goes into the output file.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output pickle path.",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="How many sequences to forward at once (nanotron path only). HF path "
        "always forwards one sequence at a time because it chunks per-doc.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("TOP_K_LOGITS", "50")),
        help="K for logging and the persisted per_token_in_topk convenience mask. "
        "Exact ranks are saved, so other K values can be replayed in visualize.py.",
    )
    return p.parse_args()


def resolve_backend(checkpoint: str, backend: str) -> str:
    if backend != "auto":
        return backend
    ckpt = Path(checkpoint)
    if ckpt.exists() and (ckpt / "config.yaml").exists():
        return "nanotron"
    return "hf"


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


# --------------------------------------------------------------------------- #
# Vocab compatibility
# --------------------------------------------------------------------------- #


def assert_vocab_compatible(vocab_size: int, records: List[dict]) -> None:
    """Catch a wrong-tokenizer checkpoint early instead of after a CUDA assert."""
    max_id = max(int(r["input_ids"].max()) for r in records)
    if max_id >= vocab_size:
        raise SystemExit(
            f"Vocab mismatch: validation set has token id {max_id} but model "
            f"vocab size is {vocab_size}. Are all checkpoints using the same "
            f"tokenizer as sample_validation.py? Re-sample or pick a different "
            f"checkpoint."
        )


# --------------------------------------------------------------------------- #
# Nanotron loader
# --------------------------------------------------------------------------- #


def load_nanotron(checkpoint: Path, dtype: torch.dtype):
    """Build a nanotron model on the local GPU at TP=PP=DP=1 and load weights."""
    from nanotron.config import ParallelismArgs, get_config_from_file
    from nanotron.models import build_model
    from nanotron.parallel import ParallelContext
    from nanotron.parallel.parameters import sanity_check
    from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
    from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
    from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state
    from nanotron.serialize import load_weights
    from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

    config = get_config_from_file((checkpoint / "config.yaml").as_posix())
    model_config = config.model.model_config
    tokenizer_path = config.tokenizer.tokenizer_name_or_path

    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    parallel_context = ParallelContext(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )

    if parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
        random_states = RandomStates(
            {"tp_synced": get_synced_random_state(
                random_state=get_current_random_state(),
                pg=parallel_context.tp_pg,
            )}
        )
    else:
        random_states = RandomStates({})

    model_config_cls = model_config.__class__.__name__
    assert model_config_cls in CONFIG_TO_MODEL_CLASS, (
        f"Unsupported model config {model_config_cls}. "
        f"Known: {sorted(CONFIG_TO_MODEL_CLASS.keys())}"
    )

    model = build_model(
        model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        dtype=dtype,
        parallel_context=parallel_context,
    )
    mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)
    sanity_check(root_module=model)
    load_weights(model=model, parallel_context=parallel_context, root_folder=checkpoint)
    model.eval()

    vocab_size = int(model_config.vocab_size)
    max_pos = int(getattr(model_config, "max_position_embeddings", 2048))
    return model, parallel_context, tokenizer_path, vocab_size, max_pos


def forward_nanotron(
    model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Return logits [B, T, V]. TP=1 so sharded_logits are already full logits."""
    sharded_logits = model.model(input_ids=input_ids, position_ids=positions)
    if sharded_logits.dim() == 2:
        sharded_logits = sharded_logits.view(input_ids.shape[0], input_ids.shape[1], -1)
    return sharded_logits


# --------------------------------------------------------------------------- #
# HuggingFace loader
# --------------------------------------------------------------------------- #


def install_transformers_remote_code_compat() -> None:
    """Patch minor Transformers API moves expected by local trust_remote_code modules."""
    import transformers.utils.generic as generic
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder

    check_model_inputs = getattr(generic, "check_model_inputs", None)
    if check_model_inputs is not None:
        try:
            check_model_inputs()
        except TypeError:
            original_check_model_inputs = check_model_inputs

            def check_model_inputs_compat(func=None):
                if func is None:
                    return original_check_model_inputs
                return original_check_model_inputs(func)

            generic.check_model_inputs = check_model_inputs_compat

    if "default" not in ROPE_INIT_FUNCTIONS:

        def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
            import torch

            config.standardize_rope_params()
            rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
            base = rope_parameters["rope_theta"]
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


def normalize_hf_config(config):
    if getattr(config, "pad_token_id", None) is None:
        pad_token_id = getattr(config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "bos_token_id", 0)
        config.pad_token_id = pad_token_id
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = None
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_scaling is None or rope_type == "default":
        rope_config = {
            "rope_type": "linear",
            "factor": 1.0,
            "rope_theta": getattr(config, "rope_theta", 10000.0),
        }
        config.rope_scaling = rope_config
        config.rope_parameters = dict(rope_config)
    return config


def hf_vocab_size_early(checkpoint: str) -> int:
    """Read vocab_size from the HF config without downloading model weights."""
    install_transformers_remote_code_compat()
    from transformers import AutoConfig

    return int(normalize_hf_config(AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)).vocab_size)


def nanotron_vocab_size_early(checkpoint: Path) -> int:
    """Read vocab_size from nanotron's config.yaml without loading weights."""
    from nanotron.config import get_config_from_file

    cfg = get_config_from_file((checkpoint / "config.yaml").as_posix())
    return int(cfg.model.model_config.vocab_size)


def load_hf(checkpoint: str, dtype: torch.dtype):
    install_transformers_remote_code_compat()
    from transformers import AutoConfig, AutoModelForCausalLM

    config = normalize_hf_config(AutoConfig.from_pretrained(checkpoint, trust_remote_code=True))
    device_map = os.environ.get("HF_DEVICE_MAP")
    if device_map is None and torch.cuda.device_count() > 1:
        device_map = "auto"

    load_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
        max_memory = os.environ.get("HF_MAX_MEMORY")
        if max_memory:
            load_kwargs["max_memory"] = {i: max_memory for i in range(torch.cuda.device_count())}

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
    model = model.eval()
    if not device_map:
        model = model.to("cuda")
    vocab_size = int(model.config.vocab_size)
    max_pos = int(getattr(model.config, "max_position_embeddings", 2048))
    return model, vocab_size, max_pos


# --------------------------------------------------------------------------- #
# Top-K predictability computation
# --------------------------------------------------------------------------- #


def _rank_stats_from_logits(
    flat_logits_fp32: torch.Tensor,  # [N, V] float
    flat_labels: torch.Tensor,        # [N] long
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute target rank and cheap diagnostics per row from fp32 logits.

    Rank uses the "best rank under ties" convention:
    ``1 + count(logit > target_logit)``. Thus rank=1 means the true label is
    one of the maximum-logit tokens. This supports replaying arbitrary K values
    without sorting or rescoring.
    """
    target_logit = flat_logits_fp32.gather(1, flat_labels.unsqueeze(-1)).squeeze(-1)
    target_rank = (flat_logits_fp32 > target_logit.unsqueeze(-1)).sum(dim=-1).to(torch.int32) + 1
    top1_logit, top1_id = flat_logits_fp32.max(dim=-1)

    # Only a 1D softmax denominator is retained, so this avoids materializing a
    # full [N, V] probability tensor while still giving useful confidence fields.
    log_denom = torch.logsumexp(flat_logits_fp32, dim=-1)
    top1_prob = (top1_logit - log_denom).exp()
    target_prob = (target_logit - log_denom).exp()
    return target_rank, top1_id, top1_prob, target_logit, target_prob


def compute_rank_stats(
    logits: torch.Tensor,             # [B, T, V] (any dtype)
    label_ids: torch.Tensor,          # [B, T] long
    label_mask: torch.Tensor,         # [B, T] bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-token rank/probability stats as numpy arrays of shape [B, T].

    Masked positions use rank=0 and top1_id=-1 so valid ranks remain positive.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V).float()
    flat_labels = label_ids.reshape(B * T)
    rank, top1_id, top1_prob, target_logit, target_prob = _rank_stats_from_logits(
        flat_logits, flat_labels
    )

    mask = label_mask.view(B * T)
    rank = torch.where(mask, rank, torch.zeros_like(rank))
    top1_id = torch.where(mask, top1_id, torch.full_like(top1_id, -1))
    mask_float = mask.float()
    top1_prob = top1_prob * mask_float
    target_logit = target_logit * mask_float
    target_prob = target_prob * mask_float
    return (
        rank.view(B, T).detach().cpu().numpy().astype(np.int32),
        top1_id.view(B, T).detach().cpu().numpy().astype(np.int64),
        top1_prob.view(B, T).detach().to(torch.float32).cpu().numpy(),
        target_logit.view(B, T).detach().to(torch.float32).cpu().numpy(),
        target_prob.view(B, T).detach().to(torch.float32).cpu().numpy(),
    )


def score_hf_per_doc(
    model,
    model_input: torch.Tensor,
    model_positions: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    max_position_embeddings: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score an HF model on a packed sequence by splitting it into per-doc chunks.

    For each contiguous span [s, e) in model_input where model_positions == 0
    marks doc starts, we forward the doc independently with HF's default
    causal attention (no cross-doc context), then place the resulting
    per-token target-rank stats back into the packed-position layout.
    Tokens at packed positions that straddle a doc boundary are zeroed by
    `label_mask` afterwards (the label there is the next doc's first token).

    If a doc is longer than `max_position_embeddings`, it is further split
    into chunks of that length to avoid RoPE / abs-pos-emb OOB asserts. Each
    sub-chunk is forwarded independently (so the model sees no context from
    the prior chunk of the same doc).

    Returns (target_rank, top1_id, top1_prob, target_logit, target_prob), each [B, T] numpy.
    """
    B, T = model_input.shape
    device = model_input.device
    out_rank = torch.zeros((B, T), dtype=torch.int32, device=device)
    out_top1_id = torch.full((B, T), -1, dtype=torch.long, device=device)
    out_top1_prob = torch.zeros((B, T), dtype=torch.float32, device=device)
    out_target_logit = torch.zeros((B, T), dtype=torch.float32, device=device)
    out_target_prob = torch.zeros((B, T), dtype=torch.float32, device=device)

    cap = max(1, int(max_position_embeddings))
    for b in range(B):
        positions = model_positions[b].detach().cpu().numpy()
        starts = list(np.flatnonzero(positions == 0).tolist())
        if not starts or starts[0] != 0:
            starts = [0] + starts
        boundaries = starts + [T]

        for s, e in zip(boundaries[:-1], boundaries[1:]):
            doc_len = e - s
            if doc_len < 1:
                continue
            # Walk the doc in `cap`-sized chunks so we never feed more
            # positions than the HF model was trained for.
            chunk_start = s
            while chunk_start < e:
                chunk_end = min(chunk_start + cap, e)
                doc_ids = model_input[b, chunk_start:chunk_end].unsqueeze(0)
                with torch.no_grad():
                    logits = model(input_ids=doc_ids, use_cache=False).logits
                flat_logits = logits[0].float()
                flat_labels = label_ids[b, chunk_start:chunk_end].to(flat_logits.device)
                rank, top1_id, top1_prob, target_logit, target_prob = _rank_stats_from_logits(
                    flat_logits, flat_labels
                )
                out_rank[b, chunk_start:chunk_end] = rank.to(device=device, dtype=torch.int32)
                out_top1_id[b, chunk_start:chunk_end] = top1_id.to(device=device, dtype=torch.long)
                out_top1_prob[b, chunk_start:chunk_end] = top1_prob.to(device=device, dtype=torch.float32)
                out_target_logit[b, chunk_start:chunk_end] = target_logit.to(device=device, dtype=torch.float32)
                out_target_prob[b, chunk_start:chunk_end] = target_prob.to(device=device, dtype=torch.float32)
                chunk_start = chunk_end

    mask = label_mask
    out_rank = torch.where(mask, out_rank, torch.zeros_like(out_rank))
    out_top1_id = torch.where(mask, out_top1_id, torch.full_like(out_top1_id, -1))
    mask_float = mask.float()
    out_top1_prob = out_top1_prob * mask_float
    out_target_logit = out_target_logit * mask_float
    out_target_prob = out_target_prob * mask_float
    return (
        out_rank.detach().cpu().numpy().astype(np.int32),
        out_top1_id.detach().cpu().numpy().astype(np.int64),
        out_top1_prob.detach().cpu().numpy(),
        out_target_logit.detach().cpu().numpy(),
        out_target_prob.detach().cpu().numpy(),
    )


def warn_if_nonfinite(
    name: str,
    arr: np.ndarray,
    label_mask_np: np.ndarray,
    record_label: str,
) -> None:
    """Print a loud warning when a float metric contains NaN or Inf on valid positions."""
    valid = label_mask_np.astype(bool)
    sub = arr[valid]
    n_nan = int(np.isnan(sub).sum())
    n_inf = int(np.isinf(sub).sum())
    if n_nan or n_inf:
        print(
            f"  [WARN] {name}: {n_nan} NaN, {n_inf} Inf in {record_label} "
            f"(out of {sub.size} valid tokens)",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def iter_micro_batches(records, micro_batch_size):
    for i in range(0, len(records), micro_batch_size):
        yield records[i : i + micro_batch_size]


def main():
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    backend = resolve_backend(args.checkpoint, args.backend)
    dtype = torch_dtype(args.dtype)

    with open(args.validation, "rb") as f:
        val = pickle.load(f)
    records = val["records"]
    if not records:
        raise SystemExit(
            f"Validation set at {args.validation} is empty; "
            f"re-check sample_validation.py output (likely a missing --source-root)."
        )
    print(
        f"Loaded {len(records)} validation sequences from {args.validation}",
        flush=True,
    )

    # Cheap vocab compat check BEFORE the (expensive) model load. Reads
    # `config.yaml` / HF AutoConfig without touching weights.
    if backend == "nanotron":
        early_vocab = nanotron_vocab_size_early(Path(args.checkpoint))
    else:
        early_vocab = hf_vocab_size_early(args.checkpoint)
    assert_vocab_compatible(early_vocab, records)

    if backend == "nanotron":
        model, _parallel_context, tokenizer_path, vocab_size, max_pos = load_nanotron(
            Path(args.checkpoint), dtype
        )
        print(
            f"[nanotron] loaded {args.checkpoint}, "
            f"tokenizer={tokenizer_path}, vocab_size={vocab_size}, max_pos={max_pos}",
            flush=True,
        )
    elif backend == "hf":
        model, vocab_size, max_pos = load_hf(args.checkpoint, dtype)
        tokenizer_path = args.checkpoint
        print(
            f"[hf] loaded {args.checkpoint}, vocab_size={vocab_size}, max_pos={max_pos}",
            flush=True,
        )
    else:
        raise ValueError(backend)

    # HF backend always operates micro_batch_size=1 because per-doc chunks
    # have variable length and can't be stacked cleanly.
    micro_batch_size = args.micro_batch_size if backend == "nanotron" else 1

    device = torch.device("cuda")
    out_records = []
    for batch_idx, chunk in enumerate(iter_micro_batches(records, micro_batch_size)):
        raw_input = torch.from_numpy(
            np.stack([r["input_ids"] for r in chunk], axis=0)
        ).to(device=device, dtype=torch.long)
        raw_positions = torch.from_numpy(
            np.stack([r["positions"] for r in chunk], axis=0)
        ).to(device=device, dtype=torch.long)

        # Slice so the model never sees seq_len+1 positions. Labels are the
        # shifted version of the raw input.
        model_input = raw_input[:, :-1].contiguous()
        model_positions = raw_positions[:, :-1].contiguous()
        label_ids = raw_input[:, 1:].contiguous()
        label_mask = (raw_positions[:, 1:] != 0).contiguous()

        if backend == "nanotron":
            with torch.no_grad():
                logits = forward_nanotron(model, model_input, model_positions)
            (
                per_token_target_rank,
                per_token_top1_id,
                per_token_top1_prob,
                per_token_target_logit,
                per_token_target_prob,
            ) = compute_rank_stats(
                logits, label_ids, label_mask
            )
        else:
            (
                per_token_target_rank,
                per_token_top1_id,
                per_token_top1_prob,
                per_token_target_logit,
                per_token_target_prob,
            ) = score_hf_per_doc(
                model, model_input, model_positions, label_ids, label_mask, max_pos
            )

        label_ids_np = label_ids.cpu().numpy().astype(np.int64)
        label_mask_np = label_mask.cpu().numpy().astype(np.bool_)

        for j, rec in enumerate(chunk):
            record_label = f"{rec['source']}/sample-{rec['sample_idx']}"
            warn_if_nonfinite("top1_prob", per_token_top1_prob[j], label_mask_np[j], record_label)
            warn_if_nonfinite("target_logit", per_token_target_logit[j], label_mask_np[j], record_label)
            warn_if_nonfinite("target_prob", per_token_target_prob[j], label_mask_np[j], record_label)
            in_topk = (per_token_target_rank[j] > 0) & (per_token_target_rank[j] <= args.top_k)
            out_records.append(
                {
                    "source": rec["source"],
                    "sample_idx": rec["sample_idx"],
                    "input_ids": rec["input_ids"],
                    "positions": rec["positions"],
                    "label_ids": label_ids_np[j],
                    "label_mask": label_mask_np[j],
                    "per_token_target_rank": per_token_target_rank[j],
                    "per_token_in_topk": in_topk.astype(np.bool_),
                    "per_token_top1_id": per_token_top1_id[j],
                    "per_token_top1_prob": per_token_top1_prob[j],
                    "per_token_target_logit": per_token_target_logit[j],
                    "per_token_target_prob": per_token_target_prob[j],
                    "model_tag": args.model_tag,
                    "backend": backend,
                }
            )
            valid = int(label_mask_np[j].sum())
            unpredictable = int((label_mask_np[j] & ~in_topk).sum())
            unpredictable_rate = unpredictable / max(valid, 1)
            median_rank = (
                float(np.median(per_token_target_rank[j][label_mask_np[j]])) if valid > 0 else float("nan")
            )
            print(
                f"  [{batch_idx:>3}.{j}] {rec['source']:<22} idx={rec['sample_idx']:<7} "
                f"top{args.top_k}_unpredictable={100 * unpredictable_rate:.2f}% "
                f"median_rank={median_rank:.1f} valid_tokens={valid}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(
            {
                "records": out_records,
                "model_tag": args.model_tag,
                "backend": backend,
                "checkpoint": str(args.checkpoint),
                "tokenizer_path": tokenizer_path,
                "vocab_size": vocab_size,
                "validation_path": str(args.validation),
                "top_k_logits": args.top_k,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\nWrote {len(out_records)} scored sequences to {args.out}", flush=True)


if __name__ == "__main__":
    main()
