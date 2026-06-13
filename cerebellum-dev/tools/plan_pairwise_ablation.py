#!/usr/bin/env python3
"""Plan top-K pairwise ablations from Cerebellum analyzer output."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    tensor: str
    label: str
    class_name: str
    weighted_rel_pct: float
    max_rel_pct: float
    worst_domain: str


def tensor_label(tensor_name: str) -> str:
    return tensor_name.replace(".weight", "").replace(".", "_")


def exact_tensor_pattern(tensor_name: str) -> str:
    return f"^{re.escape(tensor_name)}$"


def load_candidates(path: Path) -> list[Candidate]:
    data = json.loads(path.read_text())
    rows = data.get("rows") or []
    candidates = []
    for row in rows:
        if "tensor" not in row or "class" not in row:
            continue
        candidates.append(
            Candidate(
                tensor=row["tensor"],
                label=row.get("label") or tensor_label(row["tensor"]),
                class_name=row["class"],
                weighted_rel_pct=float(row.get("weighted_rel_pct", 0.0)),
                max_rel_pct=float(row.get("max_rel_pct", 0.0)),
                worst_domain=str(row.get("worst_domain") or ""),
            )
        )
    return candidates


def select_top(candidates: list[Candidate], class_name: str, k: int) -> list[Candidate]:
    rows = [c for c in candidates if c.class_name == class_name]
    if class_name == "demotable":
        rows.sort(key=lambda c: (c.weighted_rel_pct, c.max_rel_pct, c.tensor))
    else:
        rows.sort(key=lambda c: (-c.weighted_rel_pct, -c.max_rel_pct, c.tensor))
    return rows[:k]


def pair_id(a: Candidate, b: Candidate) -> str:
    return f"{tensor_label(a.tensor)}__AND__{tensor_label(b.tensor)}"


def pair_override_text(a: Candidate, b: Candidate, qtype: str) -> str:
    return f"{exact_tensor_pattern(a.tensor)}={qtype}\n{exact_tensor_pattern(b.tensor)}={qtype}\n"


def build_plan(candidates: list[Candidate], args: argparse.Namespace) -> dict[str, Any]:
    demotable = select_top(candidates, "demotable", args.top_k)
    sacred = select_top(candidates, "sacred", args.top_k)
    tolerant = select_top(candidates, "tolerant", args.top_k)

    groups: list[tuple[str, list[Candidate], list[Candidate]]] = [
        ("sacred_x_demotable", sacred, demotable),
        ("demotable_x_demotable", demotable, demotable),
        ("sacred_x_tolerant", sacred, tolerant),
    ]
    pairs = []
    for group, left, right in groups:
        if group == "demotable_x_demotable":
            iterator = itertools.combinations(left, 2)
        else:
            iterator = itertools.product(left, right)
        for a, b in iterator:
            if a.tensor == b.tensor:
                continue
            expected = a.weighted_rel_pct + b.weighted_rel_pct
            pairs.append(
                {
                    "id": pair_id(a, b),
                    "group": group,
                    "tensor_a": a.tensor,
                    "tensor_b": b.tensor,
                    "class_a": a.class_name,
                    "class_b": b.class_name,
                    "weighted_rel_pct_a": a.weighted_rel_pct,
                    "weighted_rel_pct_b": b.weighted_rel_pct,
                    "expected_additive_weighted_rel_pct": expected,
                    "override_text": pair_override_text(a, b, args.ablate_type),
                    "result_ppl": None,
                    "observed_weighted_rel_pct": None,
                    "interference_pct": None,
                }
            )

    return {
        "schema_version": 1,
        "source_analysis": str(args.analysis),
        "top_k": args.top_k,
        "ablate_type": args.ablate_type,
        "candidate_counts": {
            "demotable": len(demotable),
            "sacred": len(sacred),
            "tolerant": len(tolerant),
        },
        "candidates": {
            "demotable": [c.__dict__ for c in demotable],
            "sacred": [c.__dict__ for c in sacred],
            "tolerant": [c.__dict__ for c in tolerant],
        },
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def write_override_files(plan: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in plan["pairs"]:
        path = output_dir / f"{pair['id']}.txt"
        path.write_text(pair["override_text"])
        pair["override_file"] = str(path)


def write_runner(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.runner:
        return
    args.runner.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(["echo 'Pairwise plan runner is dry-run only for now.'", "exit 0", ""])
    for pair in plan["pairs"]:
        if "override_file" not in pair:
            continue
        out_path = args.output_dir / "ggufs" / f"{pair['id']}.gguf"
        cmd = [
            args.quantize_bin,
            "--imatrix",
            str(args.imatrix),
            "--tensor-type-file",
            pair["override_file"],
            str(args.source_gguf),
            str(out_path),
            args.base_type,
        ]
        lines.extend([f"echo '=== {pair['id']} ==='", " ".join(shlex.quote(x) for x in cmd), ""])
    args.runner.write_text("\n".join(lines))
    args.runner.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="JSON from analyze_ablation_results.py")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for pair override files")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ablate-type", default="Q3_K")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--source-gguf", type=Path, default=Path("source.gguf"))
    parser.add_argument("--imatrix", type=Path, default=Path("imatrix.dat"))
    parser.add_argument("--base-type", default="Q4_K_M")
    parser.add_argument("--quantize-bin", default="/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(load_candidates(args.analysis), args)
    write_override_files(plan, args.output_dir)
    write_runner(plan, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        "planned {pairs} pairs from {counts}".format(
            pairs=plan["pair_count"],
            counts=plan["candidate_counts"],
        )
    )


if __name__ == "__main__":
    main()
