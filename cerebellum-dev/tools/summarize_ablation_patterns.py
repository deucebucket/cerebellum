#!/usr/bin/env python3
"""Summarize cross-model Cerebellum ablation patterns."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_INPUTS = [
    Path("osmosis-qwen35-9b/ablation_results.json"),
    Path("osmosis-qwen36-27b/ablation_results.json"),
    Path("osmosis-gemma4-e4b/ablation/ablation_results.json"),
    Path("osmosis-gemma4-26b/ablation/attn_k_layers/ablation_attn_k_analysis.json"),
    Path("osmosis-gemma4-26b/ablation/attn_o_layers/ablation_results.json"),
]


@dataclass(frozen=True)
class AblationRecord:
    model: str
    source_file: str
    schema_type: str
    tensor: str
    layer: int | None
    tensor_group: str
    baseline_ppl: float
    tensor_ppl: float
    delta_abs: float
    delta_pct: float
    classification: str


def model_name_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith(("osmosis-", "cerebellum-")):
            return part
    return path.parent.name


def layer_from_tensor(tensor: str) -> int | None:
    match = re.search(r"blk\.(\d+)\.", tensor)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|_)layer[_-]?(\d+)", tensor)
    if match:
        return int(match.group(1))
    return None


def tensor_group(tensor: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if "ffn_gate_up_exps" in tensor:
        return "moe_ffn_gate_up_exps"
    if "ffn_down_exps" in tensor:
        return "moe_ffn_down_exps"
    if "ffn_gate_shexp" in tensor:
        return "moe_shared_gate"
    if "ffn_down_shexp" in tensor:
        return "moe_shared_down"
    for name in (
        "attn_qkv",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ssm_out",
        "ssm_alpha",
        "ssm_beta",
        "ssm_a",
        "output",
        "token_embd",
    ):
        if name in tensor:
            return name
    return "other"


def classify_delta(delta_pct: float) -> str:
    if delta_pct <= -0.25:
        return "demotable"
    if delta_pct <= 1.0:
        return "tolerant"
    if delta_pct >= 5.0:
        return "critical"
    return "sacred"


def scalar_baseline(value: Any) -> float:
    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return mean(vals)
    return float(value)


def scalar_ppl(value: Any) -> float:
    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return mean(vals)
    return float(value)


def load_standard_schema(path: Path, data: dict[str, Any]) -> list[AblationRecord]:
    baseline = scalar_baseline(data["baseline_ppl"])
    model = model_name_from_path(path)
    rows = []
    for label, item in (data.get("tests") or {}).items():
        if not isinstance(item, dict) or "ppl" not in item or "error" in item:
            continue
        tensor = str(item.get("gguf_tensor") or label)
        if not tensor.endswith(".weight"):
            tensor = f"{tensor}.weight"
        ppl = scalar_ppl(item["ppl"])
        delta_abs = ppl - baseline
        delta_pct = delta_abs / baseline * 100.0 if baseline else 0.0
        rows.append(
            AblationRecord(
                model=model,
                source_file=str(path),
                schema_type="standard",
                tensor=tensor,
                layer=layer_from_tensor(tensor),
                tensor_group=tensor_group(tensor),
                baseline_ppl=baseline,
                tensor_ppl=ppl,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                classification=classify_delta(delta_pct),
            )
        )
    return rows


def load_analysis_schema(path: Path, data: dict[str, Any]) -> list[AblationRecord]:
    baseline = float(data["baseline_ppl"])
    group = str(data.get("tensor_group") or "other")
    model = model_name_from_path(path)
    rows = []
    for item in data.get("results") or []:
        layer = item.get("layer")
        tensor = f"blk.{layer}.{group}.weight" if layer is not None else group
        ppl = float(item["ppl"])
        delta_abs = float(item.get("delta", ppl - baseline))
        delta_pct = float(item.get("pct", delta_abs / baseline * 100.0 if baseline else 0.0))
        rows.append(
            AblationRecord(
                model=model,
                source_file=str(path),
                schema_type="analysis_results",
                tensor=tensor,
                layer=int(layer) if layer is not None else None,
                tensor_group=tensor_group(tensor, group),
                baseline_ppl=baseline,
                tensor_ppl=ppl,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                classification=classify_delta(delta_pct),
            )
        )
    return rows


def load_list_schema(path: Path, data: list[dict[str, Any]]) -> list[AblationRecord]:
    model = model_name_from_path(path)
    rows = []
    for item in data:
        group = str(item.get("group") or "other")
        layer = item.get("layer")
        tensor = f"blk.{layer}.{group}.weight" if layer is not None else group
        baseline = float(item["baseline_ppl"])
        ppl = float(item["ppl"])
        delta_abs = ppl - baseline
        delta_pct = float(item.get("delta_pct", delta_abs / baseline * 100.0 if baseline else 0.0))
        rows.append(
            AblationRecord(
                model=model,
                source_file=str(path),
                schema_type="list_results",
                tensor=tensor,
                layer=int(layer) if layer is not None else None,
                tensor_group=tensor_group(tensor, group),
                baseline_ppl=baseline,
                tensor_ppl=ppl,
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                classification=classify_delta(delta_pct),
            )
        )
    return rows


def load_records(path: Path) -> list[AblationRecord]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "baseline_ppl" in data and "tests" in data:
        return load_standard_schema(path, data)
    if isinstance(data, dict) and "baseline_ppl" in data and "results" in data:
        return load_analysis_schema(path, data)
    if isinstance(data, list):
        return load_list_schema(path, data)
    return []


def layer_bucket(layer: int | None, max_layer: int | None) -> str:
    if layer is None or max_layer is None:
        return "unknown"
    pos = layer / max(max_layer, 1)
    if pos < 1 / 3:
        return "early"
    if pos < 2 / 3:
        return "mid"
    return "late"


def aggregate(records: list[AblationRecord]) -> dict[str, Any]:
    max_layer_by_model: dict[str, int] = {}
    for row in records:
        if row.layer is not None:
            max_layer_by_model[row.model] = max(max_layer_by_model.get(row.model, row.layer), row.layer)

    def summarize(rows: list[AblationRecord]) -> dict[str, Any]:
        vals = [row.delta_pct for row in rows]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.classification] = counts.get(row.classification, 0) + 1
        return {
            "count": len(rows),
            "mean_delta_pct": mean(vals) if vals else None,
            "median_delta_pct": median(vals) if vals else None,
            "min_delta_pct": min(vals) if vals else None,
            "max_delta_pct": max(vals) if vals else None,
            "classes": counts,
        }

    by_model_group: dict[tuple[str, str], list[AblationRecord]] = {}
    by_model_bucket: dict[tuple[str, str], list[AblationRecord]] = {}
    for row in records:
        by_model_group.setdefault((row.model, row.tensor_group), []).append(row)
        bucket = layer_bucket(row.layer, max_layer_by_model.get(row.model))
        by_model_bucket.setdefault((row.model, bucket), []).append(row)

    return {
        "coverage": {
            model: sum(1 for row in records if row.model == model)
            for model in sorted({row.model for row in records})
        },
        "by_model_group": {
            f"{model}/{group}": summarize(rows)
            for (model, group), rows in sorted(by_model_group.items())
        },
        "by_model_layer_bucket": {
            f"{model}/{bucket}": summarize(rows)
            for (model, bucket), rows in sorted(by_model_bucket.items())
        },
    }


def write_csv(path: Path, records: list[AblationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AblationRecord.__dataclass_fields__))
        writer.writeheader()
        for row in records:
            writer.writerow(row.__dict__)


def render_markdown(records: list[AblationRecord], summary: dict[str, Any]) -> str:
    lines = ["# Cross-Model Ablation Pattern Summary", ""]
    lines.extend(["## Coverage", "", "| Model | Tests |", "| --- | ---: |"])
    for model, count in summary["coverage"].items():
        lines.append(f"| `{model}` | {count} |")

    lines.extend(["", "## Group Sensitivity", "", "| Model / Group | Count | Median Δ% | Classes |", "| --- | ---: | ---: | --- |"])
    for key, item in summary["by_model_group"].items():
        classes = ", ".join(f"{k}:{v}" for k, v in sorted(item["classes"].items()))
        lines.append(f"| `{key}` | {item['count']} | {item['median_delta_pct']:.3f} | {classes} |")

    lines.extend(["", "## Layer Buckets", "", "| Model / Bucket | Count | Median Δ% | Classes |", "| --- | ---: | ---: | --- |"])
    for key, item in summary["by_model_layer_bucket"].items():
        classes = ", ".join(f"{k}:{v}" for k, v in sorted(item["classes"].items()))
        lines.append(f"| `{key}` | {item['count']} | {item['median_delta_pct']:.3f} | {classes} |")

    lines.extend(["", "## Most Demotable", "", "| Model | Tensor | Group | Δ% |", "| --- | --- | --- | ---: |"])
    for row in sorted(records, key=lambda r: r.delta_pct)[:25]:
        lines.append(f"| `{row.model}` | `{row.tensor}` | `{row.tensor_group}` | {row.delta_pct:.3f} |")

    lines.extend(["", "## Most Sensitive", "", "| Model | Tensor | Group | Δ% |", "| --- | --- | --- | ---: |"])
    for row in sorted(records, key=lambda r: r.delta_pct, reverse=True)[:25]:
        lines.append(f"| `{row.model}` | `{row.tensor}` | `{row.tensor_group}` | {row.delta_pct:.3f} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Ablation JSON inputs. Defaults to known local artifacts.")
    parser.add_argument("--output-json", type=Path, help="Write summary JSON")
    parser.add_argument("--output-md", type=Path, help="Write Markdown report")
    parser.add_argument("--output-csv", type=Path, help="Write normalized long-format CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or [path for path in DEFAULT_INPUTS if path.exists()]
    records: list[AblationRecord] = []
    for path in inputs:
        if path.exists():
            records.extend(load_records(path))
        else:
            print(f"warning: missing {path}")
    summary = aggregate(records)

    if args.output_csv:
        write_csv(args.output_csv, records)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"records": [r.__dict__ for r in records], **summary}, indent=2) + "\n")
    markdown = render_markdown(records, summary)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    else:
        print(markdown)
    print(f"loaded {len(records)} ablation records from {len(inputs)} files")


if __name__ == "__main__":
    main()
