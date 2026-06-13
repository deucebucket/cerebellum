#!/usr/bin/env python3
"""Generate Markdown comparison reports from Cerebellum benchmark JSONs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCORE_FIELDS = (
    "accuracy",
    "pass_at_1_pct",
    "pass_at_1",
    "pass_at_1_plus",
    "pass_at_1_base",
)


@dataclass(frozen=True)
class BenchmarkRow:
    model: str
    benchmark: str
    score: float
    score_field: str
    path: str
    correct: int | None = None
    total: int | None = None
    timestamp: str | None = None
    audit_status: str = "unchecked"
    missing_artifacts: tuple[str, ...] = ()
    audit_artifacts: tuple[str, ...] = ()


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def normalize_score(key: str, value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if key.startswith("pass_at_1") and score <= 1.0:
        return score * 100.0
    if key == "accuracy" and score <= 1.0:
        return score * 100.0
    return score


def infer_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_results"):
        stem = stem[: -len("_results")]
    known_suffixes = [
        "mmlu_redux",
        "arc_challenge",
        "evalplus_humaneval_plus",
        "humaneval",
        "hellaswag",
        "mmlu",
        "arc",
    ]
    for suffix in known_suffixes:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)], suffix
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "unknown"


def result_prefix(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_results")] if stem.endswith("_results") else stem


def existing_artifact(path: Path) -> str | None:
    return str(path) if path.exists() else None


def audit_artifacts(path: Path, benchmark: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return (status, present, missing) for expected benchmark audit artifacts."""
    prefix = result_prefix(path)
    parent = path.parent
    present: list[str] = []
    missing: list[str] = []

    def require_any(label: str, candidates: list[Path]) -> None:
        hit = next((candidate for candidate in candidates if candidate.exists()), None)
        if hit:
            present.append(str(hit))
        else:
            missing.append(label)

    bench = benchmark.lower()
    if any(token in bench for token in ("arc", "hellaswag", "mmlu")):
        require_any("detailed_jsonl", [parent / f"{prefix}_detailed.jsonl"])
    elif "humaneval" in bench or "evalplus" in bench:
        require_any("samples_jsonl", [parent / f"{prefix}_samples.jsonl"])
        require_any(
            "eval_results",
            [
                parent / f"{prefix}_samples_eval_results.json",
                parent / f"{prefix}_samples.jsonl_results.jsonl",
                parent / f"{prefix}_eval_results.json",
            ],
        )
    else:
        return "unchecked", tuple(present), tuple(missing)

    return ("ok" if not missing else "needs_audit"), tuple(present), tuple(missing)


def load_result(path: Path) -> BenchmarkRow | None:
    data = json.loads(path.read_text())
    inferred_model, inferred_benchmark = infer_from_filename(path)
    model = str(data.get("model") or inferred_model)
    benchmark = str(data.get("benchmark") or inferred_benchmark)
    audit_status, present_artifacts, missing_artifacts = audit_artifacts(path, benchmark)

    for key in SCORE_FIELDS:
        if key not in data:
            continue
        score = normalize_score(key, data.get(key))
        if score is None:
            continue
        return BenchmarkRow(
            model=model,
            benchmark=benchmark,
            score=score,
            score_field=key,
            path=str(path),
            correct=data.get("correct"),
            total=data.get("total") or data.get("total_problems"),
            timestamp=data.get("timestamp"),
            audit_status=audit_status,
            missing_artifacts=missing_artifacts,
            audit_artifacts=present_artifacts,
        )
    return None


def load_results(paths: list[Path], models: set[str] | None, benchmarks: set[str] | None) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for path in paths:
        try:
            row = load_result(path)
        except Exception as exc:
            print(f"warning: skipped {path}: {exc}")
            continue
        if row is None:
            continue
        if models and row.model not in models:
            continue
        if benchmarks and row.benchmark not in benchmarks:
            continue
        rows.append(row)
    return rows


def collapse_latest(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    by_key: dict[tuple[str, str], BenchmarkRow] = {}
    for row in rows:
        key = (row.model, row.benchmark)
        old = by_key.get(key)
        if old is None or ((row.timestamp or ""), row.path) > ((old.timestamp or ""), old.path):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (r.model, r.benchmark))


def matrix(rows: list[BenchmarkRow]) -> tuple[list[str], list[str], dict[tuple[str, str], BenchmarkRow]]:
    models = sorted({row.model for row in rows})
    benchmarks = sorted({row.benchmark for row in rows})
    lookup = {(row.model, row.benchmark): row for row in rows}
    return models, benchmarks, lookup


def format_score(row: BenchmarkRow | None) -> str:
    if row is None:
        return "-"
    return f"{row.score:.2f}"


def delta_text(score: float, baseline: float | None) -> str:
    if baseline is None:
        return "-"
    delta = score - baseline
    pct = (delta / baseline * 100.0) if baseline else 0.0
    return f"{delta:+.2f} ({pct:+.1f}%)"


def bar(score: float, width: int = 24) -> str:
    filled = max(0, min(width, round(score / 100.0 * width)))
    return "#" * filled + "." * (width - filled)


def render_markdown(rows: list[BenchmarkRow], baseline_model: str | None) -> str:
    models, benchmarks, lookup = matrix(rows)
    lines = ["# Benchmark Comparison", ""]

    header = "| Model | " + " | ".join(benchmarks) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in benchmarks) + " |"
    lines.extend([header, sep])
    for model in models:
        values = [format_score(lookup.get((model, bench))) for bench in benchmarks]
        lines.append(f"| `{model}` | " + " | ".join(values) + " |")

    if baseline_model and baseline_model in models:
        lines.extend(["", f"## Deltas vs `{baseline_model}`", ""])
        lines.extend([header, sep])
        for model in models:
            values = []
            for bench in benchmarks:
                row = lookup.get((model, bench))
                base = lookup.get((baseline_model, bench))
                values.append(delta_text(row.score, base.score if base else None) if row else "-")
            lines.append(f"| `{model}` | " + " | ".join(values) + " |")

    lines.extend(["", "## ASCII Bars", ""])
    for bench in benchmarks:
        lines.append(f"### {bench}")
        for model in models:
            row = lookup.get((model, bench))
            if row:
                lines.append(f"`{model:30s}` {bar(row.score)} {row.score:.2f}")
        lines.append("")

    lines.extend(["## Sources", ""])
    for row in sorted(rows, key=lambda r: (r.model, r.benchmark, r.path)):
        count = f" ({row.correct}/{row.total})" if row.correct is not None and row.total is not None else ""
        missing = f"; missing audit: {', '.join(row.missing_artifacts)}" if row.missing_artifacts else ""
        lines.append(
            f"- `{row.model}` / `{row.benchmark}`: `{row.path}` via `{row.score_field}`{count}; "
            f"audit={row.audit_status}{missing}"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_summary(rows: list[BenchmarkRow], baseline_model: str | None) -> dict[str, Any]:
    models, benchmarks, lookup = matrix(rows)
    out: dict[str, Any] = {
        "models": models,
        "benchmarks": benchmarks,
        "rows": [row.__dict__ for row in rows],
        "scores": {
            model: {bench: lookup[(model, bench)].score for bench in benchmarks if (model, bench) in lookup}
            for model in models
        },
    }
    if baseline_model:
        out["baseline_model"] = baseline_model
        out["deltas"] = {}
        for model in models:
            out["deltas"][model] = {}
            for bench in benchmarks:
                row = lookup.get((model, bench))
                base = lookup.get((baseline_model, bench))
                if row and base:
                    out["deltas"][model][bench] = {
                        "absolute": row.score - base.score,
                        "relative_pct": (row.score - base.score) / base.score * 100.0 if base.score else 0.0,
                    }
    return out


def collect_paths(inputs: list[Path], pattern: str, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob(pattern) if recursive else item.glob(pattern)))
        else:
            paths.append(item)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Result JSON files or directories")
    parser.add_argument("--glob", default="*_results.json", help="Glob used for directory inputs")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover matching result files")
    parser.add_argument("--models", help="Comma-separated model names to include")
    parser.add_argument("--benchmarks", help="Comma-separated benchmark names to include")
    parser.add_argument("--baseline-model", help="Model name for delta table")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not collapse duplicate model/benchmark rows")
    parser.add_argument("--output-md", type=Path, help="Write Markdown report")
    parser.add_argument("--output-json", type=Path, help="Write machine-readable summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_paths(args.inputs, args.glob, args.recursive)
    rows = load_results(paths, parse_csv(args.models), parse_csv(args.benchmarks))
    if not args.keep_duplicates:
        rows = collapse_latest(rows)
    markdown = render_markdown(rows, args.baseline_model)
    summary = build_summary(rows, args.baseline_model)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    else:
        print(markdown)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"loaded {len(rows)} benchmark rows from {len(paths)} files")


if __name__ == "__main__":
    main()
