#!/usr/bin/env python3
"""Analyze Cerebellum ablation results and generate tensor override files.

The tool accepts the JSON emitted by scripts/ablate_multidomain.py and the
older single-domain ablation JSONs used by osmosis.cerebellum. It computes
per-tensor PPL deltas, classifies tensors by quantization sensitivity, and can
emit a llama-quantize tensor-type-file for the safe demotion candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EMIT_CLASSES = ("demotable",)

PROFILE_WEIGHTS = {
    "general": {
        "wiki": 0.25,
        "code": 0.25,
        "math": 0.25,
        "dialogue": 0.25,
        "chat": 0.25,
        "reasoning": 0.50,
    },
    "code": {
        "code": 0.70,
        "math": 0.20,
        "reasoning": 0.20,
        "chat": 0.10,
        "wiki": 0.05,
        "dialogue": 0.05,
    },
    "reason": {
        "reasoning": 0.60,
        "math": 0.25,
        "code": 0.10,
        "wiki": 0.10,
        "chat": 0.15,
        "dialogue": 0.05,
    },
    "chat": {
        "chat": 0.70,
        "dialogue": 0.70,
        "reasoning": 0.15,
        "code": 0.05,
        "math": 0.05,
        "wiki": 0.05,
    },
    "tools": {
        "code": 0.35,
        "reasoning": 0.35,
        "math": 0.15,
        "chat": 0.10,
        "dialogue": 0.10,
        "wiki": 0.05,
    },
}


@dataclass(frozen=True)
class Thresholds:
    beneficial_weighted: float
    safe_max: float
    safe_weighted: float
    tolerant_max: float
    tolerant_weighted: float
    fragile_max: float
    fragile_weighted: float
    critical_max: float
    critical_weighted: float


DEFAULT_THRESHOLDS = Thresholds(
    beneficial_weighted=-0.0025,
    safe_max=0.0100,
    safe_weighted=0.0050,
    tolerant_max=0.0200,
    tolerant_weighted=0.0150,
    fragile_max=0.0300,
    fragile_weighted=0.0200,
    critical_max=0.0800,
    critical_weighted=0.0500,
)


def exact_tensor_pattern(tensor_name: str) -> str:
    """Return an anchored regex pattern for llama-quantize tensor-type-file."""
    return f"^{re.escape(tensor_name)}$"


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_weights(spec: str | None, domains: list[str]) -> dict[str, float]:
    if not domains:
        raise ValueError("cannot infer weights without at least one baseline domain")

    if spec is None:
        weight = 1.0 / len(domains)
        return {domain: weight for domain in domains}

    weights: dict[str, float] = {}
    for part in parse_csv(spec):
        if ":" not in part:
            raise ValueError(f"invalid weight '{part}', expected domain:value")
        key, value = part.split(":", 1)
        weights[key.strip()] = float(value)

    missing = [domain for domain in weights if domain not in domains]
    if missing:
        raise ValueError(f"weights mention domains not in baseline: {missing}")
    total = sum(weights.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"weights must sum to 1.0, got {total:.4f}: {weights}")
    return weights


def profile_weights(profile: str, domains: list[str]) -> dict[str, float]:
    if profile not in PROFILE_WEIGHTS:
        raise ValueError(f"unknown profile '{profile}', expected one of {sorted(PROFILE_WEIGHTS)}")
    raw = PROFILE_WEIGHTS[profile]
    selected = {domain: raw[domain] for domain in domains if domain in raw and raw[domain] > 0}
    if not selected:
        raise ValueError(f"profile '{profile}' has no weights for domains {domains}")
    total = sum(selected.values())
    return {domain: value / total for domain, value in selected.items()}


def load_ablation(path: Path) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, str]]]:
    data = json.loads(path.read_text())
    if "baseline_ppl" not in data:
        raise ValueError(f"{path} has no baseline_ppl field")

    raw_baseline = data["baseline_ppl"]
    if isinstance(raw_baseline, dict):
        baseline = {str(k): float(v) for k, v in raw_baseline.items()}
    else:
        baseline = {"ppl": float(raw_baseline)}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    tests = data.get("tests") or {}
    for label, item in tests.items():
        if not isinstance(item, dict):
            skipped.append({"label": str(label), "reason": "test entry is not an object"})
            continue
        tensor = item.get("gguf_tensor") or str(label)
        if not str(tensor).endswith(".weight"):
            tensor = f"{tensor}.weight"

        if "error" in item:
            skipped.append({"label": str(label), "tensor": tensor, "reason": str(item["error"])})
            continue

        raw_ppl = item.get("ppl")
        if raw_ppl is None:
            skipped.append({"label": str(label), "tensor": tensor, "reason": "missing ppl"})
            continue
        if isinstance(raw_ppl, dict):
            ppl = {str(k): float(v) for k, v in raw_ppl.items() if k in baseline}
        else:
            ppl = {"ppl": float(raw_ppl)}
        if not ppl:
            skipped.append({"label": str(label), "tensor": tensor, "reason": "no matching domains"})
            continue

        rows.append({"label": str(label), "tensor": tensor, "ppl": ppl})

    return baseline, rows, skipped


def classify_row(max_rel: float, weighted_rel: float, thresholds: Thresholds) -> str:
    if weighted_rel <= thresholds.beneficial_weighted:
        return "demotable"
    if max_rel >= thresholds.critical_max or weighted_rel >= thresholds.critical_weighted:
        return "critical"
    if max_rel >= thresholds.fragile_max or weighted_rel >= thresholds.fragile_weighted:
        return "sacred"
    if max_rel <= thresholds.safe_max and weighted_rel <= thresholds.safe_weighted:
        return "demotable"
    if max_rel <= thresholds.tolerant_max and weighted_rel <= thresholds.tolerant_weighted:
        return "tolerant"
    return "sacred"


def classify_subclass(max_rel: float, weighted_rel: float, thresholds: Thresholds) -> str:
    if weighted_rel <= thresholds.beneficial_weighted:
        return "beneficial"
    if max_rel <= thresholds.safe_max and weighted_rel <= thresholds.safe_weighted:
        return "safe"
    return "sensitivity"


def analyze_rows(
    baseline: dict[str, float],
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    analyzed = []
    for row in rows:
        deltas_abs: dict[str, float] = {}
        deltas_rel: dict[str, float] = {}
        weighted_rel = 0.0
        for domain, ppl in row["ppl"].items():
            base = baseline[domain]
            delta_abs = ppl - base
            delta_rel = delta_abs / base
            deltas_abs[domain] = delta_abs
            deltas_rel[domain] = delta_rel
            weighted_rel += weights.get(domain, 0.0) * delta_rel

        worst_domain = max(deltas_rel, key=lambda d: deltas_rel[d])
        best_domain = min(deltas_rel, key=lambda d: deltas_rel[d])
        max_rel = deltas_rel[worst_domain]
        classification = classify_row(max_rel, weighted_rel, thresholds)
        analyzed.append(
            {
                "label": row["label"],
                "tensor": row["tensor"],
                "class": classification,
                "subclass": classify_subclass(max_rel, weighted_rel, thresholds),
                "ppl": row["ppl"],
                "deltas_abs": deltas_abs,
                "deltas_rel": deltas_rel,
                "deltas_rel_pct": {d: v * 100.0 for d, v in deltas_rel.items()},
                "weighted_rel": weighted_rel,
                "weighted_rel_pct": weighted_rel * 100.0,
                "max_rel": max_rel,
                "max_rel_pct": max_rel * 100.0,
                "worst_domain": worst_domain,
                "best_domain": best_domain,
            }
        )

    class_rank = {"critical": 0, "sacred": 1, "tolerant": 2, "demotable": 3}
    analyzed.sort(key=lambda r: (class_rank[r["class"]], -r["max_rel"], r["weighted_rel"], r["tensor"]))
    return analyzed


def count_classes(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in ("demotable", "tolerant", "sacred", "critical")}
    for row in rows:
        counts[row["class"]] += 1
    return counts


def build_overrides(rows: list[dict[str, Any]], emit_classes: set[str], qtype: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["class"] in emit_classes]
    selected.sort(key=lambda r: (r["weighted_rel"], r["max_rel"], r["tensor"]))
    return [
        {
            "tensor": row["tensor"],
            "pattern": exact_tensor_pattern(row["tensor"]),
            "qtype": qtype,
            "class": row["class"],
        }
        for row in selected
    ]


def write_tensor_type_file(path: Path, overrides: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{item['pattern']}={item['qtype']}" for item in overrides]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def format_pct(value: float) -> str:
    return f"{value:+.3f}%"


def write_summary(path: Path, result: dict[str, Any], limit: int) -> None:
    rows = result["rows"][:limit]
    lines = [
        "# Cerebellum Ablation Analysis",
        "",
        f"- input: `{result['input']}`",
        f"- rows: {result['tested_count']} tested, {result['skipped_count']} skipped",
        f"- overrides: {result['override_count']} tensors",
        f"- counts: {result['counts']}",
        "",
        "| class | tensor | weighted | worst | worst domain |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {class_} | `{tensor}` | {weighted} | {worst} | {worst_domain} |".format(
                class_=row["class"],
                tensor=row["tensor"],
                weighted=format_pct(row["weighted_rel_pct"]),
                worst=format_pct(row["max_rel_pct"]),
                worst_domain=row["worst_domain"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_result(
    input_path: Path,
    baseline: dict[str, float],
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    weights: dict[str, float],
    thresholds: Thresholds,
    emit_classes: set[str],
    override_type: str,
) -> dict[str, Any]:
    analyzed = analyze_rows(baseline, rows, weights, thresholds)
    overrides = build_overrides(analyzed, emit_classes, override_type)
    return {
        "schema_version": 1,
        "input": str(input_path),
        "baseline_ppl": baseline,
        "weights": weights,
        "thresholds": thresholds.__dict__,
        "emit_classes": sorted(emit_classes),
        "tested_count": len(analyzed),
        "skipped_count": len(skipped),
        "counts": count_classes(analyzed),
        "override_count": len(overrides),
        "overrides": overrides,
        "rows": analyzed,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True, help="Ablation results JSON")
    parser.add_argument("--output-json", type=Path, help="Write full analysis JSON")
    parser.add_argument("--summary-md", type=Path, help="Write compact markdown summary")
    parser.add_argument("--tensor-type-file", type=Path, help="Write llama-quantize override file")
    parser.add_argument("--override-type", default="Q3_K", help="Quant type for emitted override lines")
    parser.add_argument(
        "--emit-classes",
        default=",".join(DEFAULT_EMIT_CLASSES),
        help="Comma-separated classes to emit to tensor-type-file",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional domain weights, e.g. chat:0.25,reasoning:0.5,code:0.25. Default: equal or --profile.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_WEIGHTS),
        help="Named task profile. Ignored when --weights is provided.",
    )
    parser.add_argument("--beneficial-weighted", type=float, default=DEFAULT_THRESHOLDS.beneficial_weighted)
    parser.add_argument("--safe-max", type=float, default=DEFAULT_THRESHOLDS.safe_max)
    parser.add_argument("--safe-weighted", type=float, default=DEFAULT_THRESHOLDS.safe_weighted)
    parser.add_argument("--tolerant-max", type=float, default=DEFAULT_THRESHOLDS.tolerant_max)
    parser.add_argument("--tolerant-weighted", type=float, default=DEFAULT_THRESHOLDS.tolerant_weighted)
    parser.add_argument("--fragile-max", type=float, default=DEFAULT_THRESHOLDS.fragile_max)
    parser.add_argument("--fragile-weighted", type=float, default=DEFAULT_THRESHOLDS.fragile_weighted)
    parser.add_argument("--critical-max", type=float, default=DEFAULT_THRESHOLDS.critical_max)
    parser.add_argument("--critical-weighted", type=float, default=DEFAULT_THRESHOLDS.critical_weighted)
    parser.add_argument("--summary-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline, rows, skipped = load_ablation(args.ablation)
    weights = parse_weights(args.weights, list(baseline)) if args.weights else (
        profile_weights(args.profile, list(baseline)) if args.profile else parse_weights(None, list(baseline))
    )
    thresholds = Thresholds(
        beneficial_weighted=args.beneficial_weighted,
        safe_max=args.safe_max,
        safe_weighted=args.safe_weighted,
        tolerant_max=args.tolerant_max,
        tolerant_weighted=args.tolerant_weighted,
        fragile_max=args.fragile_max,
        fragile_weighted=args.fragile_weighted,
        critical_max=args.critical_max,
        critical_weighted=args.critical_weighted,
    )
    result = build_result(
        input_path=args.ablation,
        baseline=baseline,
        rows=rows,
        skipped=skipped,
        weights=weights,
        thresholds=thresholds,
        emit_classes=set(parse_csv(args.emit_classes)),
        override_type=args.override_type,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary_md:
        write_summary(args.summary_md, result, args.summary_limit)
    if args.tensor_type_file:
        write_tensor_type_file(args.tensor_type_file, result["overrides"])

    print(
        "analyzed {tested} tensors ({skipped} skipped): {counts}; overrides={overrides}".format(
            tested=result["tested_count"],
            skipped=result["skipped_count"],
            counts=result["counts"],
            overrides=result["override_count"],
        )
    )
    if args.tensor_type_file:
        print(f"tensor-type-file -> {args.tensor_type_file}")


if __name__ == "__main__":
    main()
