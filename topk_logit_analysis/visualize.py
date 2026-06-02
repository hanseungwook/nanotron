"""Render an HTML comparison of reference-model top-K predictability.

Inputs: one or more pickles from `score_topk.py`. All pickles must reference
the same validation set (same sequence order, token ids, positions, labels,
and label masks).

For a chosen K, a valid label token is:
  - predictable when target_rank <= K
  - unpredictable when target_rank > K
  - ignored when label_mask is false

The scored files persist exact target ranks, so changing --top-k does not
require rescoring.
"""

import argparse
import html
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scored",
        type=Path,
        nargs="+",
        required=True,
        help="One or more pickles produced by score_topk.py.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional HuggingFace tokenizer name/path. If omitted or unavailable, token ids are shown.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="K for the predictability rule. Valid tokens with target_rank > K are unpredictable.",
    )
    p.add_argument(
        "--max-tokens-per-doc",
        type=int,
        default=512,
        help="Trim each rendered document to at most this many tokens.",
    )
    p.add_argument(
        "--num-disagreement-docs",
        type=int,
        default=10,
        help="How many top-disagreement sequences to surface in the leaderboard.",
    )
    p.add_argument(
        "--max-sequences-per-source",
        type=int,
        default=None,
        help="If set, keep only the first N sampled packed sequences per source for the report.",
    )
    p.add_argument(
        "--position-bins",
        type=int,
        default=32,
        help="Number of bins for the positional unpredictable-token histogram.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output HTML file.",
    )
    return p.parse_args()


def unpredictable_mask(target_rank: np.ndarray, label_mask: np.ndarray, top_k: int) -> np.ndarray:
    valid = label_mask.astype(bool)
    rank = target_rank.astype(np.int64)
    return valid & (rank > int(top_k))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def agreement(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    valid = valid.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    return float((a[valid] == b[valid]).sum() / valid.sum())


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    xs = x[finite]
    ys = y[finite]
    rx = np.argsort(np.argsort(xs)).astype(np.float64)
    ry = np.argsort(np.argsort(ys)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_scored_files(paths: List[Path]) -> Tuple[List[Dict], List[Tuple[str, int]]]:
    bundles = []
    for path in paths:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        if "records" not in bundle:
            raise SystemExit(f"{path} does not look like a score_topk.py output: missing records")
        if bundle["records"] and "per_token_target_rank" not in bundle["records"][0]:
            raise SystemExit(f"{path} is missing per_token_target_rank; use score_topk.py outputs")
        bundles.append(bundle)

    if not bundles:
        raise SystemExit("No --scored files provided.")

    if len(bundles) == 1:
        print(
            f"[WARN] only one --scored file ({bundles[0].get('model_tag', '<unknown>')}); "
            "pairwise tables will contain only self-comparison.",
            flush=True,
        )

    seq_keys = None
    for bundle in bundles:
        keys = [(r["source"], r["sample_idx"]) for r in bundle["records"]]
        if seq_keys is None:
            seq_keys = keys
        elif keys != seq_keys:
            raise SystemExit(
                f"Scored files disagree on validation sequence order: "
                f"{bundle.get('model_tag', '<unknown>')} differs from {bundles[0].get('model_tag', '<unknown>')}."
            )

    base = bundles[0]["records"]
    for bundle in bundles[1:]:
        for i, key in enumerate(seq_keys):
            for field in ("input_ids", "positions", "label_ids", "label_mask"):
                if not np.array_equal(base[i][field], bundle["records"][i][field]):
                    raise SystemExit(
                        f"{field} disagrees for {key} between {bundles[0]['model_tag']} "
                        f"and {bundle['model_tag']}. Re-score against the same validation.pt."
                    )
    return bundles, seq_keys


def limit_sequences_per_source(
    bundles: List[Dict],
    seq_keys: List[Tuple[str, int]],
    max_per_source: int | None,
) -> Tuple[List[Dict], List[Tuple[str, int]]]:
    if max_per_source is None:
        return bundles, seq_keys
    if max_per_source < 1:
        raise SystemExit("--max-sequences-per-source must be >= 1")

    counts: Dict[str, int] = defaultdict(int)
    keep_indices = []
    for i, (source, _) in enumerate(seq_keys):
        if counts[source] < max_per_source:
            keep_indices.append(i)
            counts[source] += 1

    filtered = []
    for bundle in bundles:
        b = dict(bundle)
        b["records"] = [bundle["records"][i] for i in keep_indices]
        filtered.append(b)
    return filtered, [seq_keys[i] for i in keep_indices]


def build_token_renderer(tokenizer_name: str | None, all_token_ids):
    if not tokenizer_name:
        return lambda token_id: f"<id:{int(token_id)}>"

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        print(
            f"[WARN] could not load tokenizer {tokenizer_name!r}: {exc}. Rendering token ids instead.",
            flush=True,
        )
        return lambda token_id: f"<id:{int(token_id)}>"

    replacement = "\ufffd"

    def visualize_text(s: str) -> str:
        return (
            s.replace(" ", "[space]")
            .replace("\t", "[tab]")
            .replace("\n", "[nl]")
            .replace("\r", "[cr]")
        )

    cache: Dict[int, str] = {}
    for tid in sorted({int(tid) for tid in all_token_ids}):
        decoded = tok.decode([tid], clean_up_tokenization_spaces=False)
        if not decoded or replacement in decoded:
            raw = tok.convert_ids_to_tokens(tid)
            cache[tid] = visualize_text(str(raw)) if raw is not None else visualize_text(decoded)
        else:
            cache[tid] = visualize_text(decoded)

    return lambda token_id: cache.get(int(token_id), f"<id:{int(token_id)}>")


def split_into_docs(positions_label: np.ndarray, max_doc_len: int) -> List[Tuple[int, int]]:
    starts = list(np.flatnonzero(positions_label == 0).tolist())
    if not starts or starts[0] != 0:
        starts = [0] + starts
    boundaries = starts + [int(positions_label.size)]

    docs: List[Tuple[int, int]] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s <= 0:
            continue
        docs.append((s, min(e, s + max_doc_len)))
    return docs


def validate_ranks(records_per_seq: List[List[Dict]]) -> None:
    for recs in records_per_seq:
        for rec in recs:
            valid = rec["label_mask"].astype(bool)
            ranks = rec["per_token_target_rank"]
            bad = valid & (ranks <= 0)
            if bad.any():
                raise SystemExit(
                    f"{rec['model_tag']} has non-positive target ranks on valid labels "
                    f"for {rec['source']}/sample-{rec['sample_idx']}."
                )


def compute_summaries(
    bundles: List[Dict],
    seq_keys: List[Tuple[str, int]],
    masks: List[List[np.ndarray]],
    records_per_seq: List[List[Dict]],
    position_bins: int,
):
    tags = [str(b["model_tag"]) for b in bundles]
    n_models = len(tags)

    sources = sorted({source for source, _ in seq_keys})
    src_unpred = {src: np.zeros(n_models, dtype=np.int64) for src in sources}
    src_valid = {src: 0 for src in sources}
    for (source, _), recs, seq_masks in zip(seq_keys, records_per_seq, masks):
        valid_count = int(recs[0]["label_mask"].sum())
        src_valid[source] += valid_count
        for m, mask in enumerate(seq_masks):
            src_unpred[source][m] += int(mask.sum())
    per_source_rate = {
        src: src_unpred[src] / max(src_valid[src], 1)
        for src in sources
    }

    pairwise_jaccard = np.full((n_models, n_models), np.nan)
    pairwise_agreement = np.full((n_models, n_models), np.nan)
    for i in range(n_models):
        for j in range(n_models):
            j_vals = []
            a_vals = []
            for seq_idx, seq_masks in enumerate(masks):
                valid = records_per_seq[seq_idx][0]["label_mask"].astype(bool)
                j_vals.append(jaccard(seq_masks[i], seq_masks[j]))
                a_vals.append(agreement(seq_masks[i], seq_masks[j], valid))
            pairwise_jaccard[i, j] = float(np.nanmean(j_vals)) if j_vals else float("nan")
            pairwise_agreement[i, j] = float(np.nanmean(a_vals)) if a_vals else float("nan")

    spearman = np.full((n_models, n_models), np.nan)
    pooled_ranks = []
    for m in range(n_models):
        parts = []
        for recs in records_per_seq:
            valid = recs[m]["label_mask"].astype(bool)
            parts.append(recs[m]["per_token_target_rank"][valid].astype(np.float64))
        pooled_ranks.append(np.concatenate(parts) if parts else np.array([]))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                spearman[i, j] = 1.0
            elif pooled_ranks[i].size > 0 and pooled_ranks[i].size == pooled_ranks[j].size:
                spearman[i, j] = spearmanr_safe(pooled_ranks[i], pooled_ranks[j])

    seq_len = records_per_seq[0][0]["label_mask"].size if records_per_seq else 0
    bins = max(1, min(int(position_bins), max(seq_len, 1)))
    hist = np.zeros((n_models, bins), dtype=np.float64)
    totals = np.zeros(n_models, dtype=np.float64)
    for seq_masks in masks:
        for m, mask in enumerate(seq_masks):
            for bin_idx in range(bins):
                lo = bin_idx * seq_len // bins
                hi = (bin_idx + 1) * seq_len // bins
                hist[m, bin_idx] += int(mask[lo:hi].sum())
            totals[m] += int(mask.sum())
    hist_normalized = hist / np.maximum(totals[:, None], 1)

    seq_disagreement = []
    for seq_idx, seq_masks in enumerate(masks):
        pair_values = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                valid = records_per_seq[seq_idx][0]["label_mask"].astype(bool)
                v = agreement(seq_masks[i], seq_masks[j], valid)
                if not np.isnan(v):
                    pair_values.append(v)
        seq_disagreement.append(1.0 - min(pair_values) if pair_values else float("nan"))

    return {
        "tags": tags,
        "per_source_rate": per_source_rate,
        "pairwise_jaccard": pairwise_jaccard,
        "pairwise_agreement": pairwise_agreement,
        "spearman": spearman,
        "hist": hist_normalized,
        "seq_disagreement": seq_disagreement,
    }


CSS = """
:root {
  --fg: #1f2933;
  --muted: #697586;
  --border: #d9e2ec;
  --header: #f8fafc;
  --predictable: #edf7ed;
  --unpredictable: #ff8a80;
  --ignored: #cbd5e1;
  --boundary: #fff4bf;
  --link: #1d4ed8;
}
body { font-family: ui-sans-serif, system-ui, sans-serif; color: var(--fg); margin: 0; padding: 0 24px 60px; }
h1 { margin: 24px 0 6px; }
h2 { margin-top: 34px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
h3 { margin-top: 22px; font-size: 14px; }
.lead { color: var(--muted); font-size: 13px; max-width: 86ch; margin: 6px 0 16px; }
.legend { color: var(--muted); font-size: 12px; margin: 10px 0 22px; }
.legend .sw { display: inline-block; width: 14px; height: 12px; border: 1px solid #aab7c4; border-radius: 2px; vertical-align: middle; margin: 0 4px 0 12px; }
.tbl { border-collapse: collapse; font-size: 12px; margin: 8px 0 20px; }
.tbl th, .tbl td { border: 1px solid var(--border); padding: 5px 8px; text-align: right; }
.tbl th { background: var(--header); font-weight: 600; }
.tbl th:first-child, .tbl td:first-child { text-align: left; }
.heatmap td, .heatmap th { min-width: 96px; height: 34px; text-align: center; }
.heatmap th:first-child { min-width: 150px; text-align: left; }
.heat-cell { font-variant-numeric: tabular-nums; font-weight: 600; }
.source-index { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 20px; max-width: 1100px; }
.source-index a { color: var(--link); text-decoration: none; background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 3px; padding: 3px 7px; font-size: 12px; }
.source-switcher { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px 0; background: rgba(255,255,255,0.96); border-bottom: 1px solid var(--border); }
.source-button { border: 1px solid #bcccdc; background: #f8fafc; color: var(--fg); border-radius: 3px; padding: 4px 8px; font-size: 12px; cursor: pointer; }
.source-button.active { border-color: #2563eb; background: #dbeafe; color: #1d4ed8; font-weight: 600; }
.source-block[hidden] { display: none; }
.position-chart { width: 100%; max-width: 1180px; overflow-x: auto; margin: 8px 0 20px; }
.position-chart svg { min-width: 920px; background: #fff; }
.position-chart .axis { stroke: #9aa5b1; stroke-width: 1; }
.position-chart .grid-line { stroke: #e4e7eb; stroke-width: 1; }
.position-chart .series-line { fill: none; stroke-width: 2.2; }
.position-chart text { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px; fill: var(--muted); }
.leaderboard a { color: var(--link); text-decoration: none; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.doc { margin: 20px 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.doc-header { background: var(--header); padding: 6px 12px; font-size: 12px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; display: flex; gap: 8px; flex-wrap: wrap; }
.doc-header .pill { padding: 1px 6px; border-radius: 3px; background: #e4e7eb; }
.grid-scroll { overflow-x: auto; max-width: 100%; padding: 8px 12px; }
.grid { display: grid; grid-template-rows: auto; grid-auto-flow: column; grid-auto-columns: max-content; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; align-items: center; }
.text-cell { padding: 1px 3px; min-width: 12px; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.text-cell.boundary-mark { background: var(--boundary); color: #7c4a03; }
.rank-cell { height: 10px; min-width: 10px; margin: 1px 0; border-radius: 2px; }
.rank-predictable { background: var(--predictable); border: 1px solid #c6e7c6; }
.rank-unpredictable { background: var(--unpredictable); }
.rank-ignored { background: var(--ignored); }
.row-label { padding: 1px 8px 1px 0; text-align: right; color: var(--muted); font-size: 11px; min-width: 150px; position: sticky; left: 0; background: white; z-index: 1; }
.row-label.header { font-weight: 600; color: var(--fg); }
"""


SCRIPT = """
<script>
(function () {
  function setSource(source) {
    document.querySelectorAll('.source-block').forEach(function (block) {
      block.hidden = source !== '__all__' && block.dataset.source !== source;
    });
    document.querySelectorAll('.source-button').forEach(function (button) {
      button.classList.toggle('active', button.dataset.source === source);
    });
  }
  function bind(selector, shouldScroll) {
    document.querySelectorAll(selector).forEach(function (control) {
      control.addEventListener('click', function (event) {
        event.preventDefault();
        setSource(control.dataset.source);
        if (shouldScroll) {
          var section = document.getElementById('per-doc-section');
          if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }
  document.addEventListener('DOMContentLoaded', function () {
    bind('.source-button', false);
    bind('.source-index a[data-source]', true);
  });
})();
</script>
"""


def heat_style(v: float, lo: float = 0.0, hi: float = 1.0, invert: bool = False) -> str:
    if np.isnan(v):
        return ""
    z = (float(v) - lo) / max(hi - lo, 1e-9)
    z = max(0.0, min(1.0, z))
    if invert:
        z = 1.0 - z
    alpha = 0.10 + 0.80 * z
    color = "255,255,255" if z >= 0.72 else "31,41,55"
    return f"background-color: rgba(220, 38, 38, {alpha:.2f}); color: rgb({color});"


def corr_style(v: float) -> str:
    if np.isnan(v):
        return ""
    v = max(-1.0, min(1.0, float(v)))
    if v >= 0:
        alpha = 0.10 + 0.80 * v
        color = "255,255,255" if v >= 0.72 else "31,41,55"
        return f"background-color: rgba(37, 99, 235, {alpha:.2f}); color: rgb({color});"
    alpha = 0.10 + 0.80 * abs(v)
    color = "255,255,255" if v <= -0.72 else "31,41,55"
    return f"background-color: rgba(220, 38, 38, {alpha:.2f}); color: rgb({color});"


def render_legend(top_k: int) -> str:
    return (
        '<div class="legend">'
        f"Rule: valid label tokens with target rank <= <b>{top_k}</b> are predictable; rank > {top_k} are unpredictable. "
        '<span class="sw" style="background:#edf7ed;"></span>predictable/in-top-K '
        '<span class="sw" style="background:#ff8a80;"></span>unpredictable/not-in-top-K '
        '<span class="sw" style="background:#cbd5e1;"></span>ignored boundary '
        '<span class="sw" style="background:#fff4bf;"></span>doc-boundary token'
        '</div>'
    )


def render_source_index(seq_keys: List[Tuple[str, int]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    ordered_sources = []
    for source, _ in seq_keys:
        counts[source] += 1
        if source not in ordered_sources:
            ordered_sources.append(source)
    links = "".join(
        f'<a href="#per-doc-section" data-source="{html.escape(source)}">'
        f'{html.escape(source)} ({counts[source]})</a>'
        for source in ordered_sources
    )
    return f'<h2>Source index</h2><div class="source-index">{links}</div>'


def render_source_switcher(seq_keys: List[Tuple[str, int]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    ordered_sources = []
    for source, _ in seq_keys:
        counts[source] += 1
        if source not in ordered_sources:
            ordered_sources.append(source)
    if not ordered_sources:
        return ""
    buttons = []
    for i, source in enumerate(ordered_sources):
        active = " active" if i == 0 else ""
        buttons.append(
            f'<button type="button" class="source-button{active}" '
            f'data-source="{html.escape(source)}">{html.escape(source)} ({counts[source]})</button>'
        )
    buttons.append('<button type="button" class="source-button" data-source="__all__">All</button>')
    return f'<div class="source-switcher">{"".join(buttons)}</div>'


def render_per_source_rate(per_source_rate: Dict[str, np.ndarray], tags: List[str], top_k: int) -> str:
    head = "".join(f"<th>{html.escape(t)}</th>" for t in tags)
    rows = []
    all_vals = np.concatenate(list(per_source_rate.values())) if per_source_rate else np.array([0.0])
    hi = max(float(np.nanmax(all_vals)), 1e-9)
    for src in sorted(per_source_rate.keys()):
        cells = []
        for v in per_source_rate[src]:
            cells.append(
                f'<td class="heat-cell" style="{heat_style(float(v), 0.0, hi)}" '
                f'title="{100 * v:.3f}%">{100 * v:.2f}%</td>'
            )
        rows.append(f"<tr><th>{html.escape(src)}</th>{''.join(cells)}</tr>")
    return (
        '<h2>Per-source unpredictable rate</h2>'
        f'<p class="lead">Fraction of valid tokens whose true label is outside each model\'s top-{top_k} logits. '
        'Higher values mean the reference model considers more tokens unpredictable for that source.</p>'
        f'<table class="tbl heatmap"><tr><th>source</th>{head}</tr>{"".join(rows)}</table>'
    )


def render_pairwise_table(matrix: np.ndarray, tags: List[str], title: str, description: str, style_fn) -> str:
    head = "".join(f"<th>{html.escape(t)}</th>" for t in tags)
    rows = []
    for i, tag in enumerate(tags):
        cells = []
        for j in range(len(tags)):
            v = matrix[i, j]
            if np.isnan(v):
                cells.append("<td>-</td>")
            else:
                cells.append(
                    f'<td class="heat-cell" style="{style_fn(float(v))}" title="{v:.4f}">{v:.2f}</td>'
                )
        rows.append(f"<tr><th>{html.escape(tag)}</th>{''.join(cells)}</tr>")
    return (
        f'<h2>{html.escape(title)}</h2>'
        f'<p class="lead">{description}</p>'
        f'<table class="tbl heatmap"><tr><th></th>{head}</tr>{"".join(rows)}</table>'
    )


def render_histogram(hist: np.ndarray, tags: List[str]) -> str:
    bins = hist.shape[1]
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c"]
    width, height = 1120, 340
    left, right, top, bottom = 70, 250, 24, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    uniform = 1.0 / max(bins, 1)
    max_y = float(np.nanmax(hist)) if hist.size else uniform
    max_y = max(max_y, uniform, 1e-9) * 1.12

    def x_for_bin(bin_idx: int) -> float:
        if bins <= 1:
            return left + plot_w / 2
        return left + plot_w * bin_idx / (bins - 1)

    def y_for_value(value: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(value, max_y)) / max_y)

    y_ticks = [0.0, max_y / 2.0, max_y]
    y_grid = "".join(
        f'<line class="grid-line" x1="{left}" y1="{y_for_value(v):.1f}" '
        f'x2="{width - right}" y2="{y_for_value(v):.1f}"></line>'
        f'<text x="{left - 8}" y="{y_for_value(v) + 4:.1f}" text-anchor="end">{100 * v:.1f}%</text>'
        for v in y_ticks
    )
    x_ticks = [(0, "0%"), (max(0, bins // 2), "50%"), (max(0, bins - 1), "100%")]
    x_grid = "".join(
        f'<line class="grid-line" x1="{x_for_bin(i):.1f}" y1="{top}" '
        f'x2="{x_for_bin(i):.1f}" y2="{top + plot_h}"></line>'
        f'<text x="{x_for_bin(i):.1f}" y="{top + plot_h + 22}" text-anchor="middle">{label}</text>'
        for i, label in x_ticks
    )
    series = []
    for m, tag in enumerate(tags):
        color = colors[m % len(colors)]
        points = " ".join(
            f"{x_for_bin(b):.1f},{y_for_value(float(hist[m, b])):.1f}"
            for b in range(bins)
        )
        series.append(
            f'<polyline class="series-line" points="{points}" stroke="{color}">'
            f'<title>{html.escape(tag)}</title></polyline>'
        )
    legend = []
    legend_x = width - right + 28
    for m, tag in enumerate(tags):
        y = top + 18 + m * 24
        color = colors[m % len(colors)]
        legend.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" '
            f'stroke="{color}" stroke-width="2.5"></line>'
            f'<text x="{legend_x + 32}" y="{y + 4}">{html.escape(tag)}</text>'
        )

    svg = (
        f'<div class="position-chart"><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Fraction of unpredictable tokens by position bin">'
        f"{y_grid}{x_grid}"
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"></line>'
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"></line>'
        f'{"".join(series)}'
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 14}" text-anchor="middle">position in packed sequence</text>'
        f'<text x="16" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {top + plot_h / 2:.1f})">fraction of unpredictable tokens</text>'
        f'{"".join(legend)}'
        '</svg></div>'
    )
    return (
        '<h2>Positional unpredictable profile</h2>'
        f'<p class="lead">Where each model concentrates tokens outside top-K. Each point is one of {bins} position bins, normalized by that model\'s unpredictable-token count.</p>'
        + svg
    )


def render_leaderboard(seq_disagreement: List[float], seq_keys: List[Tuple[str, int]], num: int) -> str:
    pairs = [
        (seq_disagreement[i], seq_keys[i][0], seq_keys[i][1], i)
        for i in range(len(seq_keys))
        if not np.isnan(seq_disagreement[i])
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for v, src, sid, idx in pairs[:num]:
        rows.append(
            f'<tr><td><a href="#seq-{idx}">{html.escape(src)} sample {sid}</a></td>'
            f'<td>{v:.2f}</td></tr>'
        )
    return (
        f'<h2>Top {num} most-disagreeing sequences</h2>'
        '<p class="lead">Sequences where the least-agreeing model pair differs most on predictable vs unpredictable labels.</p>'
        f'<table class="tbl leaderboard"><tr><th>sequence</th><th>disagreement</th></tr>{"".join(rows)}</table>'
    )


def render_doc_grid(
    label_ids: np.ndarray,
    label_mask: np.ndarray,
    target_ranks: List[np.ndarray],
    top1_ids: List[np.ndarray],
    masks: List[np.ndarray],
    positions_label: np.ndarray,
    decoder,
    tags: List[str],
    top_k: int,
    s: int,
    e: int,
) -> str:
    length = e - s
    text_cells = []
    for col in range(length):
        idx = s + col
        token_text = html.escape(decoder(int(label_ids[idx])))
        cls = "text-cell"
        if not bool(label_mask[idx]) and int(positions_label[idx]) == 0:
            cls += " boundary-mark"
        text_cells.append(
            f'<span class="{cls}" style="grid-row:1;grid-column:{col + 2};">{token_text}</span>'
        )
    header_label = '<span class="row-label header" style="grid-row:1;grid-column:1;">token</span>'

    model_cells = []
    for m, tag in enumerate(tags):
        model_cells.append(
            f'<span class="row-label" style="grid-row:{m + 2};grid-column:1;">{html.escape(tag)}</span>'
        )
        for col in range(length):
            idx = s + col
            if not bool(label_mask[idx]):
                cls = "rank-cell rank-ignored"
                hover = "ignored boundary label"
            elif bool(masks[m][idx]):
                cls = "rank-cell rank-unpredictable"
                hover = (
                    f"id={int(label_ids[idx])} rank={int(target_ranks[m][idx])} "
                    f"top1={int(top1_ids[m][idx])} outside top-{top_k}"
                )
            else:
                cls = "rank-cell rank-predictable"
                hover = (
                    f"id={int(label_ids[idx])} rank={int(target_ranks[m][idx])} "
                    f"top1={int(top1_ids[m][idx])} within top-{top_k}"
                )
            model_cells.append(
                f'<span class="{cls}" style="grid-row:{m + 2};grid-column:{col + 2};" '
                f'title="{html.escape(hover)}"></span>'
            )
    return f'<div class="grid-scroll"><div class="grid">{header_label}{"".join(text_cells)}{"".join(model_cells)}</div></div>'


def main():
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.max_tokens_per_doc < 1:
        raise SystemExit("--max-tokens-per-doc must be >= 1")

    bundles, seq_keys = load_scored_files(args.scored)
    bundles, seq_keys = limit_sequences_per_source(bundles, seq_keys, args.max_sequences_per_source)
    records_per_seq = [
        [bundle["records"][seq_idx] for bundle in bundles]
        for seq_idx in range(len(seq_keys))
    ]
    validate_ranks(records_per_seq)

    masks: List[List[np.ndarray]] = []
    for recs in records_per_seq:
        masks.append([
            unpredictable_mask(rec["per_token_target_rank"], rec["label_mask"], args.top_k)
            for rec in recs
        ])

    tokenizer_name = args.tokenizer or bundles[0].get("tokenizer_path")
    visible_ids = set()
    for recs in records_per_seq:
        visible_ids.update(int(x) for x in recs[0]["label_ids"][: args.max_tokens_per_doc])
        visible_ids.update(int(x) for x in recs[0]["label_ids"])
    decoder = build_token_renderer(tokenizer_name, visible_ids)

    summaries = compute_summaries(
        bundles=bundles,
        seq_keys=seq_keys,
        masks=masks,
        records_per_seq=records_per_seq,
        position_bins=args.position_bins,
    )
    tags = summaries["tags"]

    out = [
        "<html><head><meta charset='utf-8'>",
        f"<title>Top-{args.top_k} logit predictability comparison</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Reference top-{args.top_k} predictability: model comparison</h1>",
        render_legend(args.top_k),
        render_source_index(seq_keys),
        render_per_source_rate(summaries["per_source_rate"], tags, args.top_k),
        render_pairwise_table(
            summaries["pairwise_jaccard"],
            tags,
            "Pairwise unpredictable-mask Jaccard",
            "Average overlap between models' not-in-top-K token sets. 1.0 means identical unpredictable-token sets; lower means different tokens are considered unpredictable.",
            corr_style,
        ),
        render_pairwise_table(
            summaries["pairwise_agreement"],
            tags,
            "Pairwise predictable/unpredictable agreement",
            "Fraction of valid labels where two models make the same in-top-K vs not-in-top-K decision.",
            corr_style,
        ),
        render_pairwise_table(
            summaries["spearman"],
            tags,
            "Pairwise target-rank Spearman correlation",
            "Rank correlation over exact target ranks for all valid labels. This is independent of the chosen K and helps judge whether models order token predictability similarly.",
            corr_style,
        ),
        render_histogram(summaries["hist"], tags),
        render_leaderboard(summaries["seq_disagreement"], seq_keys, args.num_disagreement_docs),
        '<h2 id="per-doc-section">Per-document token grids</h2>',
        '<p class="lead">Rows are models and columns are next-token labels. Green means the true label is in top-K, red means outside top-K, and gray means ignored by label_mask.</p>',
        render_source_switcher(seq_keys),
    ]

    last_source = None
    first_source = seq_keys[0][0] if seq_keys else None
    source_open = False
    for seq_idx, (source, sample_idx) in enumerate(seq_keys):
        if source != last_source:
            if source_open:
                out.append("</section>")
            hidden = "" if source == first_source else " hidden"
            out.append(
                f'<section class="source-block" data-source="{html.escape(source)}"{hidden}>'
            )
            out.append(f'<h3 id="src-{html.escape(source)}">Source: {html.escape(source)}</h3>')
            last_source = source
            source_open = True

        anchor = f"seq-{seq_idx}"
        first_rec = records_per_seq[seq_idx][0]
        positions_label = first_rec["positions"][1:]
        docs = split_into_docs(positions_label, args.max_tokens_per_doc)
        out.append(
            f'<div class="doc-header" id="{anchor}">'
            f"<b>{html.escape(source)}</b>"
            f'<span class="pill">sample {sample_idx}</span>'
            f'<span class="pill">{len(docs)} docs</span>'
            f'<span class="pill">#{anchor}</span>'
            "</div>"
        )

        for doc_n, (s, e) in enumerate(docs):
            meta = [
                f'<span class="pill">doc {doc_n + 1}/{len(docs)}</span>',
                f'<span class="pill">label pos [{s}, {e})</span>',
            ]
            for tag, rec, mask in zip(tags, records_per_seq[seq_idx], masks[seq_idx]):
                valid = rec["label_mask"][s:e].astype(bool)
                unpred = int(mask[s:e].sum())
                valid_n = int(valid.sum())
                median_rank = (
                    float(np.median(rec["per_token_target_rank"][s:e][valid]))
                    if valid_n > 0
                    else float("nan")
                )
                meta.append(
                    f'<span class="pill">{html.escape(tag)}: {unpred}/{valid_n} outside top-{args.top_k}, median rank {median_rank:.1f}</span>'
                )

            out.append(
                '<div class="doc">'
                f'<div class="doc-header">{"".join(meta)}</div>'
                + render_doc_grid(
                    label_ids=first_rec["label_ids"],
                    label_mask=first_rec["label_mask"],
                    target_ranks=[rec["per_token_target_rank"] for rec in records_per_seq[seq_idx]],
                    top1_ids=[rec.get("per_token_top1_id", np.full_like(rec["label_ids"], -1)) for rec in records_per_seq[seq_idx]],
                    masks=masks[seq_idx],
                    positions_label=positions_label,
                    decoder=decoder,
                    tags=tags,
                    top_k=args.top_k,
                    s=s,
                    e=e,
                )
                + "</div>"
            )

    if source_open:
        out.append("</section>")

    out.append(SCRIPT)
    out.append("</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(out), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
