#!/usr/bin/env python3
"""Build the cerebellum multi-domain calibration corpus.

Output: /var/home/deucebucket/games/cerebellum-calibration/
  cerebellum_calibration_wiki.txt        (~100k tokens)
  cerebellum_calibration_code.txt        (~100k tokens)
  cerebellum_calibration_math.txt        (~100k tokens)
  cerebellum_calibration_dialogue.txt    (~100k tokens)
  cerebellum_calibration_combined.txt    (all 4 with domain markers)
  README.md                              (sources, slice indices, sha256s)

Token target per domain: ~100k. Approximated as ~400KB of text (assuming
~4 chars/token for English/code mix). Exact token count matters less than
having a stable, reproducible per-domain signal.

Sources (HuggingFace datasets, pulled via huggingface_hub + pyarrow):
  wiki:     Salesforce/wikitext, config wikitext-103-v1, split validation
  code:     bigcode/the-stack-smol, python+javascript files
  math:     openai/gsm8k, config main, split train (full problems w/ reasoning)
  dialogue: HuggingFaceH4/ultrachat_200k, split train_sft, multi-turn convos

Reproducibility: README captures HF revisions + slice indices + sha256s.
Anyone re-running must reproduce byte-for-byte.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, dataset_info

OUTPUT_DIR = Path("/var/home/deucebucket/games/cerebellum-calibration")
TARGET_BYTES = 400_000   # ~100k tokens at 4 chars/token


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_to_target(text_iter, target_bytes: int):
    """Concatenate strings from iter until total bytes >= target. Returns (text, n_items)."""
    parts = []
    total = 0
    n = 0
    for text in text_iter:
        if not text or not text.strip():
            continue
        parts.append(text)
        total += len(text.encode("utf-8"))
        n += 1
        if total >= target_bytes:
            break
    return "\n\n".join(parts), n


def build_wiki():
    print("[wiki] downloading wikitext-103-v1 validation parquet...")
    info = dataset_info("Salesforce/wikitext")
    rev = info.sha
    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename="wikitext-103-v1/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=rev,
    )
    table = pq.read_table(path)
    texts = table.column("text").to_pylist()
    # wikitext has many empty / heading lines; filter to substantial paragraphs
    non_trivial = [t for t in texts if t and len(t.strip()) > 200]
    text, n = slice_to_target(iter(non_trivial), TARGET_BYTES)
    return text, {
        "source": "Salesforce/wikitext",
        "revision": rev,
        "config": "wikitext-103-v1",
        "split": "validation",
        "filter": "len(t.strip()) > 200",
        "items": n,
        "bytes": len(text.encode("utf-8")),
    }


def build_code():
    print("[code] downloading code_search_net python + javascript...")
    # code_search_net is public and parquet-formatted. Schema has 'whole_func_string'
    # (the function source) plus a docstring split out — we use the full func_string
    # which keeps docstrings inline so the model sees realistic code shape.
    info = dataset_info("code_search_net")
    rev = info.sha
    parts = []
    items_total = 0
    for lang, want_bytes in [("python", int(TARGET_BYTES * 0.7)),
                              ("javascript", int(TARGET_BYTES * 0.3))]:
        path = hf_hub_download(
            repo_id="code_search_net",
            filename=f"{lang}/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=rev,
        )
        table = pq.read_table(path)
        # code_search_net has 'whole_func_string' (full function with docstring)
        # or 'func_code_string' depending on the schema version
        cols = table.column_names
        col_name = "whole_func_string" if "whole_func_string" in cols else "func_code_string"
        contents = table.column(col_name).to_pylist()
        # Filter to function-sized chunks (substantive but not file-scale)
        sized = [c for c in contents if c and 500 < len(c) < 6000]
        text, n = slice_to_target(iter(sized), want_bytes)
        parts.append(f"# === {lang} ===\n{text}")
        items_total += n
    full_text = "\n\n".join(parts)
    return full_text, {
        "source": "code_search_net",
        "revision": rev,
        "languages": ["python (70%)", "javascript (30%)"],
        "split": "train",
        "field": "whole_func_string (full function source with docstring)",
        "filter": "500 < len(content) < 6000",
        "items": items_total,
        "bytes": len(full_text.encode("utf-8")),
    }


def build_math():
    print("[math] downloading gsm8k main train parquet...")
    info = dataset_info("openai/gsm8k")
    rev = info.sha
    path = hf_hub_download(
        repo_id="openai/gsm8k",
        filename="main/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=rev,
    )
    table = pq.read_table(path)
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    # Format as "Q: ...\nA: ..." (full chain-of-thought reasoning included in answer)
    formatted = [f"Q: {q.strip()}\nA: {a.strip()}" for q, a in zip(questions, answers)]
    text, n = slice_to_target(iter(formatted), TARGET_BYTES)
    return text, {
        "source": "openai/gsm8k",
        "revision": rev,
        "config": "main",
        "split": "train",
        "format": "Q: <question>\\nA: <full reasoning + answer>",
        "items": n,
        "bytes": len(text.encode("utf-8")),
    }


def build_dialogue():
    print("[dialogue] downloading ultrachat_200k train_sft parquet...")
    info = dataset_info("HuggingFaceH4/ultrachat_200k")
    rev = info.sha
    # ultrachat_200k filenames have hash suffixes
    path = hf_hub_download(
        repo_id="HuggingFaceH4/ultrachat_200k",
        filename="data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        repo_type="dataset",
        revision=rev,
    )
    table = pq.read_table(path)
    convos = table.column("messages").to_pylist()
    # Format multi-turn conversations as "User: ...\nAssistant: ..." blocks
    formatted = []
    for messages in convos:
        if not messages:
            continue
        turns = []
        for m in messages:
            role = m.get("role", "?").capitalize()
            content = m.get("content", "").strip()
            if content:
                turns.append(f"{role}: {content}")
        if turns:
            formatted.append("\n\n".join(turns))
    text, n = slice_to_target(iter(formatted), TARGET_BYTES)
    return text, {
        "source": "HuggingFaceH4/ultrachat_200k",
        "revision": rev,
        "split": "train_sft",
        "file": "data/train_sft-00000-of-00003.parquet",
        "format": "User: ...\\n\\nAssistant: ... (multi-turn)",
        "items": n,
        "bytes": len(text.encode("utf-8")),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUTPUT_DIR / "_corpus_meta.json"
    domains = {}
    if meta_path.exists():
        try:
            domains = json.loads(meta_path.read_text())
            print(f"[resume] loaded existing meta for {list(domains.keys())}")
        except Exception:
            domains = {}

    builders = [
        ("wiki", build_wiki),
        ("code", build_code),
        ("math", build_math),
        ("dialogue", build_dialogue),
    ]

    for name, fn in builders:
        out_path = OUTPUT_DIR / f"cerebellum_calibration_{name}.txt"
        # Resume: skip domains whose output file exists, is non-empty, and has cached metadata
        if name in domains and out_path.exists() and out_path.stat().st_size > 1000:
            cached_sha = domains[name].get("sha256")
            actual_sha = sha256_file(out_path)
            if cached_sha == actual_sha:
                print(f"  [{name}] SKIP — already built ({out_path.stat().st_size:,} bytes, sha256 match)")
                continue
            else:
                print(f"  [{name}] sha mismatch, rebuilding")
        text, meta = fn()
        out_path.write_text(text, encoding="utf-8")
        meta["sha256"] = sha256_file(out_path)
        domains[name] = meta
        # Save meta after each successful domain so a crash doesn't lose progress
        meta_path.write_text(json.dumps(domains, indent=2), encoding="utf-8")
        print(f"  [{name}] {meta['items']} items, {meta['bytes']:,} bytes, sha256={meta['sha256'][:16]}...")

    # Concatenated combined file with domain markers
    combined_parts = []
    for name in ("wiki", "code", "math", "dialogue"):
        sep = f"\n\n===== DOMAIN: {name} =====\n\n"
        text = (OUTPUT_DIR / f"cerebellum_calibration_{name}.txt").read_text(encoding="utf-8")
        combined_parts.append(sep + text)
    combined = "".join(combined_parts)
    combined_path = OUTPUT_DIR / "cerebellum_calibration_combined.txt"
    combined_path.write_text(combined, encoding="utf-8")
    combined_sha = sha256_file(combined_path)
    print(f"  [combined] {len(combined.encode('utf-8')):,} bytes, sha256={combined_sha[:16]}...")

    # README
    readme = ["# Cerebellum Multi-Domain Calibration Corpus", ""]
    readme.append("Reproducibility data for the multi-domain ablation methodology.")
    readme.append("Built by `cerebellum-dev/build_calibration_corpus.py`.")
    readme.append("")
    readme.append("## Files")
    readme.append("")
    readme.append("| File | Items | Bytes | SHA256 |")
    readme.append("|---|---|---|---|")
    for name in ("wiki", "code", "math", "dialogue"):
        m = domains[name]
        readme.append(f"| cerebellum_calibration_{name}.txt | {m['items']} | {m['bytes']:,} | `{m['sha256']}` |")
    readme.append(f"| cerebellum_calibration_combined.txt | (all 4) | {len(combined.encode('utf-8')):,} | `{combined_sha}` |")
    readme.append("")
    readme.append("## Sources")
    readme.append("")
    for name, meta in domains.items():
        readme.append(f"### {name}")
        readme.append("")
        readme.append("```json")
        readme.append(json.dumps(meta, indent=2))
        readme.append("```")
        readme.append("")
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print()
    print(f"=== Corpus built at {OUTPUT_DIR} ===")
    print(f"Total: {sum(d['bytes'] for d in domains.values()):,} bytes across 4 domains")
    print(f"Combined: {len(combined.encode('utf-8')):,} bytes")
    print(f"README: {OUTPUT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
