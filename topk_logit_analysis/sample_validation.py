"""Sample a small diverse validation set from the pretokenized training mix.

Each tokenized source folder contains datatrove `.ds` shards. We pull a few
fixed sequences per source so the same exact token IDs are scored by every
model in the comparison.

Output: a single torch-pickle file with a list of dicts, one per sampled
sequence:
    {
        "source":      str   - folder basename (e.g. "math-qwen")
        "sample_idx":  int   - index inside the DatatroveFolderDataset
        "input_ids":   np.int64 [seq_len + 1]
        "positions":   np.int64 [seq_len + 1]  (0 marks doc start)
    }

The sequence length is kept whatever the training config uses (default 8192),
so each sequence typically contains multiple packed documents. Document
boundaries are recoverable from `positions == 0`.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset

DEFAULT_SOURCE_ROOT = Path("/mnt/weka/shrd/k2m/omkar.pangarkar/training/tokenized/train")

DEFAULT_SOURCES = [
    "web-high",
    "web-high-medium",
    "opencoder",
    "stack-edu",
    "txt360-qa",
    "arabic",
    "math-qwen",
    "math-oss",
    "math-rewrite",
    "math-dialogue",
    "agentic-math-dialogue",
    "nemotron-sft",
    "general",
    "planning",
    "ai",
    "reasoning",
    "games",
    "other",
    "hq-rewrite",
]


def positive_int(value: str) -> int:
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {iv}")
    return iv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Parent directory holding one folder per tokenized source.",
    )
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Subset of source folder names to sample. Default: all 19 mix sources.",
    )
    p.add_argument(
        "--num-docs-per-source",
        type=positive_int,
        default=5,
        help="How many sequences to pull from each source. Default 5 keeps the "
        "report compact; bump higher when scoring is cheap.",
    )
    p.add_argument(
        "--sequence-length",
        type=positive_int,
        default=8192,
        help="Sequence length used during training; must match training config.",
    )
    p.add_argument(
        "--token-size",
        type=positive_int,
        default=4,
        help="Bytes per token in .ds files (4 for vocab >= 65535, else 2).",
    )
    p.add_argument(
        "--eos-token-id",
        type=int,
        default=1,
        help="EOS token id. Used by datatrove when return_positions=False; we set return_positions=True so this is informational only.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for picking which sequences to sample within each source.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pt file path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.sources if args.sources is not None else DEFAULT_SOURCES
    rng = np.random.default_rng(args.seed)

    out_records = []
    for source in sources:
        folder = args.source_root / source
        if not folder.exists():
            print(f"[WARN] skipping {source}: folder {folder} does not exist", flush=True)
            continue

        ds = DatatroveFolderDataset(
            data_folder=str(folder),
            filename_pattern="*.ds",
            seq_len=args.sequence_length,
            recursive=False,
            token_size=args.token_size,
            shuffle=False,
            return_positions=True,
        )

        n = len(ds)
        if n == 0:
            print(f"[WARN] skipping {source}: empty dataset", flush=True)
            continue

        k = min(args.num_docs_per_source, n)
        if k < args.num_docs_per_source:
            print(
                f"[WARN] {source}: requested {args.num_docs_per_source} sequences "
                f"but only {n} available; sampling {k}",
                flush=True,
            )
        sample_ids = rng.choice(n, size=k, replace=False)

        for sid in sample_ids:
            sample = ds[int(sid)]
            input_ids = np.asarray(sample["input_ids"], dtype=np.int64)
            positions = np.asarray(sample["positions"], dtype=np.int64)
            out_records.append(
                {
                    "source": source,
                    "sample_idx": int(sid),
                    "input_ids": input_ids,
                    "positions": positions,
                }
            )

        print(
            f"[OK] {source}: dataset has {n} sequences, sampled {k}",
            flush=True,
        )

    if not out_records:
        raise SystemExit(
            f"No source folders resolved under {args.source_root}. "
            f"Re-check --source-root or --sources (looked for: {', '.join(sources)})."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(
            {
                "records": out_records,
                "sequence_length": args.sequence_length,
                "token_size": args.token_size,
                "seed": args.seed,
                "sources": sources,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    total_tokens = sum(r["input_ids"].size - 1 for r in out_records)
    print(
        f"\nWrote {len(out_records)} sequences ({total_tokens:,} label tokens) to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
