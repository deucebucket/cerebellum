"""
recall_bench_compiled.py -- Baseline-controlled recall benchmark for dead-block GGUF.

Loads model-A (baseline) and model-B (dead-block) sequentially via llama-cpp-python,
runs the same N prompts greedy against both, and reports:
  - Overall recall: A vs B
  - Post-cutoff slice recall: A vs B  (primary metric — symbols the base model cannot know)

Post-cutoff slice definition:
  1. Symbols whose module is in a hardcoded list of Python 3.13/3.14-era additions
     (annotationlib, compression.zstd, dbm.sqlite3, ... — see POST_CUTOFF_MODULES below).
  2. Any symbol where model-A scores 0 (baseline ignorant — dead block is the only hope).

Usage:
    python recall_bench_compiled.py \\
        --model-a qwen2.5-3b-brainloop.gguf \\
        --model-b cerebellum-deadblock-python.gguf \\
        [--n 200] [--seed 42] [--out results.json]

    # Dry-run (argument parsing only, no model loading):
    python recall_bench_compiled.py --dry-run \\
        --model-a x.gguf --model-b y.gguf
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
import torch


# ---------------------------------------------------------------------------
# Post-cutoff module list
# Modules added or substantially changed in Python 3.13 / 3.14
# (annotationlib is entirely new in 3.14; compression.zstd and dbm.sqlite3
#  in 3.13; others are candidates if they appear in the corpus).
# ---------------------------------------------------------------------------
POST_CUTOFF_MODULES = frozenset({
    "annotationlib",
    "compression",
    "compression.zstd",
    "dbm.sqlite3",
    # Additional 3.13+ additions
    "pathlib.Path.full_match",   # method added 3.13
    "warnings.deprecated",       # added 3.13
})

STATUS_FILE = "DEADBLOCK_STATUS.md"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def token_overlap(ref: str, hyp: str) -> float:
    """Fraction of whitespace-split tokens in ref that appear in hyp (case-insensitive)."""
    ref_tokens = set(re.split(r"\s+", ref.strip().lower()))
    hyp_tokens = set(re.split(r"\s+", hyp.strip().lower()))
    ref_tokens.discard("")
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def content_overlap(doc: str, completion: str) -> float:
    """
    Score a completion against all content lines in the doc (all lines after the first).
    Returns the max token_overlap across all content lines, or 0 if no content lines.
    This preserves / improves the existing per-symbol scoring logic.
    """
    lines = [ln.strip() for ln in doc.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return token_overlap(lines[0] if lines else "", completion)
    content_lines = lines[1:]
    return max(token_overlap(ln, completion) for ln in content_lines)


def get_symbol_module(symbol: str) -> str:
    """Return the top-level module name from 'module.submodule.symbol'."""
    return symbol.split(".")[0] if "." in symbol else symbol


def is_post_cutoff(symbol: str) -> bool:
    """Return True if the symbol's module (or dotted prefix) is in POST_CUTOFF_MODULES."""
    parts = symbol.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if prefix in POST_CUTOFF_MODULES:
            return True
    return False


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_docs_get_symbol(docs: list, idx: int):
    """Return (symbol, doc_string) for the given index."""
    doc = docs[idx]
    lines = [ln.strip() for ln in doc.strip().split("\n") if ln.strip()]
    symbol = lines[0] if lines else f"doc_{idx}"
    return symbol, doc


def run_model(llm, docs: list, indices: list, max_tokens: int, overlap_threshold: float,
              label: str) -> list:
    """
    Run inference for all indices and return a list of result dicts.
    Each dict: {idx, symbol, doc, completion, overlap, hit, post_cutoff}
    """
    results = []
    total = len(indices)
    hits = 0

    for rank, idx in enumerate(indices):
        symbol, doc = load_docs_get_symbol(docs, idx)
        prompt = f"Question: How do I use {symbol} in Python?\nAnswer: "

        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            echo=False,
        )
        completion = output["choices"][0]["text"]
        overlap = content_overlap(doc, completion)
        is_hit = overlap >= overlap_threshold
        if is_hit:
            hits += 1

        results.append({
            "idx": idx,
            "symbol": symbol,
            "doc": doc,
            "completion": completion.strip(),
            "overlap": round(overlap, 4),
            "hit": is_hit,
            "post_cutoff": is_post_cutoff(symbol),
        })

        if (rank + 1) % 20 == 0 or (rank + 1) == total:
            print(f"  [{label}] {rank+1}/{total}  running recall={hits/(rank+1):.3f}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_stats(results: list, baseline_results: list | None = None):
    """
    Returns a dict with:
      overall_hits, overall_total, overall_recall
      pc_hits, pc_total, pc_recall           (post-cutoff from module list)
      baseline0_hits, baseline0_total, baseline0_recall  (symbols where baseline=0)
      combined_pc_hits, combined_pc_total, combined_pc_recall  (union of both slices)
    """
    total = len(results)
    hits = sum(1 for r in results if r["hit"])

    # Post-cutoff: module-based
    pc = [r for r in results if r["post_cutoff"]]
    pc_hits = sum(1 for r in pc if r["hit"])

    # Post-cutoff: baseline-0 slice (symbols where baseline scored 0)
    b0_indices = set()
    if baseline_results is not None:
        b0_indices = {r["idx"] for r in baseline_results if r["overlap"] == 0.0}
    b0 = [r for r in results if r["idx"] in b0_indices]
    b0_hits = sum(1 for r in b0 if r["hit"])

    # Combined post-cutoff: union
    combined_pc_indices = {r["idx"] for r in pc} | b0_indices
    combined_pc = [r for r in results if r["idx"] in combined_pc_indices]
    combined_hits = sum(1 for r in combined_pc if r["hit"])

    return {
        "overall_hits": hits,
        "overall_total": total,
        "overall_recall": hits / total if total else 0.0,
        "pc_hits": pc_hits,
        "pc_total": len(pc),
        "pc_recall": pc_hits / len(pc) if pc else 0.0,
        "baseline0_hits": b0_hits,
        "baseline0_total": len(b0),
        "baseline0_recall": b0_hits / len(b0) if b0 else 0.0,
        "combined_pc_hits": combined_hits,
        "combined_pc_total": len(combined_pc),
        "combined_pc_recall": combined_hits / len(combined_pc) if combined_pc else 0.0,
    }


def format_markdown_table(stats_a: dict, stats_b: dict,
                           model_a: str, model_b: str) -> str:
    """Render a markdown summary table comparing A vs B."""
    name_a = os.path.basename(model_a)
    name_b = os.path.basename(model_b)

    def pct(x):
        return f"{x*100:.1f}%"

    def delta(a, b):
        d = b - a
        return f"+{d*100:.1f}pp" if d >= 0 else f"{d*100:.1f}pp"

    rows = [
        ("Overall recall",
         f"{stats_a['overall_hits']}/{stats_a['overall_total']} ({pct(stats_a['overall_recall'])})",
         f"{stats_b['overall_hits']}/{stats_b['overall_total']} ({pct(stats_b['overall_recall'])})",
         delta(stats_a['overall_recall'], stats_b['overall_recall'])),
        ("Post-cutoff (module list)",
         f"{stats_a['pc_hits']}/{stats_a['pc_total']} ({pct(stats_a['pc_recall'])})",
         f"{stats_b['pc_hits']}/{stats_b['pc_total']} ({pct(stats_b['pc_recall'])})",
         delta(stats_a['pc_recall'], stats_b['pc_recall'])),
        ("Baseline-0 slice",
         f"{stats_a['baseline0_hits']}/{stats_a['baseline0_total']} ({pct(stats_a['baseline0_recall'])})",
         f"{stats_b['baseline0_hits']}/{stats_b['baseline0_total']} ({pct(stats_b['baseline0_recall'])})",
         delta(stats_a['baseline0_recall'], stats_b['baseline0_recall'])),
        ("Combined post-cutoff (primary) *",
         f"{stats_a['combined_pc_hits']}/{stats_a['combined_pc_total']} ({pct(stats_a['combined_pc_recall'])})",
         f"{stats_b['combined_pc_hits']}/{stats_b['combined_pc_total']} ({pct(stats_b['combined_pc_recall'])})",
         delta(stats_a['combined_pc_recall'], stats_b['combined_pc_recall'])),
    ]

    lines = [
        f"| Metric | {name_a} (A) | {name_b} (B) | Delta B-A |",
        "|--------|------------|------------|-----------|",
    ]
    for row in rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines.append("")
    lines.append("\\* Combined post-cutoff = module-list slice UNION baseline-0 slice.")
    return "\n".join(lines)


def append_status(markdown_table: str, model_a: str, model_b: str,
                  n: int, seed: int) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n---\n\n"
        f"## Recall Bench Results [{ts}]\n\n"
        f"**model-A**: `{model_a}`  \n"
        f"**model-B**: `{model_b}`  \n"
        f"**n**: {n}, **seed**: {seed}\n\n"
        f"{markdown_table}\n"
    )
    print(entry)
    with open(STATUS_FILE, "a") as fh:
        fh.write(entry)
    print(f"[+] Results appended to {STATUS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline-controlled recall benchmark: model-A vs model-B."
    )
    parser.add_argument("--model-a", required=True,
                        help="Baseline GGUF path (model A)")
    parser.add_argument("--model-b", required=True,
                        help="Dead-block GGUF path (model B)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of symbols to evaluate (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for symbol selection (default: 42)")
    parser.add_argument("--out", default="results.json",
                        help="Output JSON path (default: results.json)")
    parser.add_argument("--docs", default="python_13k_docs.pt",
                        help="Path to python_13k_docs.pt")
    parser.add_argument("--max-tokens", type=int, default=60,
                        help="Max new tokens per completion (default: 60)")
    parser.add_argument("--overlap-threshold", type=float, default=0.5,
                        help="Min token overlap to count as hit (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse args and exit without loading models.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Argument parsing successful.")
        print(f"  model-a:           {args.model_a}")
        print(f"  model-b:           {args.model_b}")
        print(f"  n:                 {args.n}")
        print(f"  seed:              {args.seed}")
        print(f"  out:               {args.out}")
        print(f"  docs:              {args.docs}")
        print(f"  max-tokens:        {args.max_tokens}")
        print(f"  overlap-threshold: {args.overlap_threshold}")
        print(f"  post-cutoff modules: {sorted(POST_CUTOFF_MODULES)}")
        return

    # Import here so the script can be syntax-checked without llama-cpp-python
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        sys.exit(1)

    # Load docs and sample indices
    print(f"[*] Loading docs from {args.docs}...")
    docs = torch.load(args.docs, map_location="cpu", weights_only=False)
    print(f"[*] Loaded {len(docs)} docs.")

    random.seed(args.seed)
    indices = random.sample(range(len(docs)), min(args.n, len(docs)))
    print(f"[*] Sampled {len(indices)} symbols (seed={args.seed}).")

    # Identify post-cutoff symbols in the sample (for info)
    pc_sample = [i for i in indices
                 if is_post_cutoff(load_docs_get_symbol(docs, i)[0])]
    print(f"[*] Post-cutoff (module-list) symbols in sample: {len(pc_sample)}")

    # ---- Model A (baseline) ----
    print(f"\n[*] Loading model A: {args.model_a}")
    llm_a = Llama(
        model_path=args.model_a,
        n_gpu_layers=-1,
        n_ctx=512,
        verbose=False,
    )
    print(f"[*] Running model A...")
    results_a = run_model(llm_a, docs, indices, args.max_tokens,
                          args.overlap_threshold, label="A")
    del llm_a  # free before loading B

    # ---- Model B (dead-block) ----
    print(f"\n[*] Loading model B: {args.model_b}")
    llm_b = Llama(
        model_path=args.model_b,
        n_gpu_layers=-1,
        n_ctx=512,
        verbose=False,
    )
    print(f"[*] Running model B...")
    results_b = run_model(llm_b, docs, indices, args.max_tokens,
                          args.overlap_threshold, label="B")
    del llm_b

    # ---- Compute stats ----
    stats_a = compute_stats(results_a, baseline_results=results_a)
    stats_b = compute_stats(results_b, baseline_results=results_a)

    # ---- Markdown summary ----
    table = format_markdown_table(stats_a, stats_b, args.model_a, args.model_b)
    print("\n" + table)

    # ---- Save JSON ----
    output = {
        "meta": {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "n": args.n,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "overlap_threshold": args.overlap_threshold,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "stats_a": stats_a,
        "stats_b": stats_b,
        "results_a": results_a,
        "results_b": results_b,
    }
    with open(args.out, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[+] Results saved to {args.out}")

    # ---- Append to status file ----
    append_status(table, args.model_a, args.model_b, args.n, args.seed)


if __name__ == "__main__":
    main()
