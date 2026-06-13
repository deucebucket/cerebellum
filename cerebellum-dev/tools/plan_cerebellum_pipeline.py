#!/usr/bin/env python3
"""Create a dry-run Cerebellum pipeline manifest and runner script."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DOMAINS = "wiki,code,math,dialogue"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    kind: str
    command: list[str]
    outputs: list[str]
    gpu: bool = False
    note: str = ""

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    out = args.output_dir
    runs = out / "runs"
    variants = out / "variants"
    bench = out / "benchmark_results"

    ablation_json = runs / "ablation_results.json"
    analysis_json = runs / "ablation_analysis.json"
    analysis_md = runs / "ablation_analysis.md"
    tensor_types = variants / "tensor_types_demotable.txt"
    final_gguf = variants / f"{args.model_name}-{args.base_type.lower()}-cerebellum.gguf"
    benchmark_md = bench / "benchmark_comparison.md"
    benchmark_json = bench / "benchmark_comparison.json"

    steps = [
        PipelineStep(
            name="ablate",
            kind="measurement",
            gpu=True,
            command=[
                "python",
                "scripts/ablate_multidomain.py",
                "--source-gguf",
                str(args.source_gguf),
                "--imatrix",
                str(args.imatrix),
                "--base-type",
                args.base_type,
                "--ablate-type",
                args.ablate_type,
                "--tensors-file",
                str(args.tensors_file),
                "--corpus-dir",
                str(args.corpus_dir),
                "--domains",
                args.domains,
                "--output",
                str(ablation_json),
                "--queue-depth",
                str(args.queue_depth),
                "--ppl-workers",
                str(args.ppl_workers),
                "--ctx-size",
                str(args.ctx_size),
                "--chunks",
                str(args.chunks),
            ],
            outputs=[str(ablation_json)],
            note="Measures per-tensor PPL deltas. Resume-safe if the JSON already exists.",
        ),
        PipelineStep(
            name="analyze-ablation",
            kind="analysis",
            command=[
                "python",
                "cerebellum-dev/tools/analyze_ablation_results.py",
                "--ablation",
                str(ablation_json),
                "--weights",
                args.weights,
                "--output-json",
                str(analysis_json),
                "--summary-md",
                str(analysis_md),
                "--tensor-type-file",
                str(tensor_types),
                "--override-type",
                args.ablate_type,
            ],
            outputs=[str(analysis_json), str(analysis_md), str(tensor_types)],
            note="Classifies tensors and emits exact anchored tensor-type overrides.",
        ),
        PipelineStep(
            name="final-quantize",
            kind="build",
            command=[
                args.quantize_bin,
                "--imatrix",
                str(args.imatrix),
                "--tensor-type-file",
                str(tensor_types),
                str(args.source_gguf),
                str(final_gguf),
                args.base_type,
            ],
            outputs=[str(final_gguf)],
            note="Builds the candidate GGUF using stock llama-quantize.",
        ),
        PipelineStep(
            name="benchmark-report",
            kind="report",
            command=[
                "python",
                "cerebellum-dev/tools/compare_benchmark_results.py",
                str(bench),
                "--recursive",
                "--baseline-model",
                args.baseline_model,
                "--output-md",
                str(benchmark_md),
                "--output-json",
                str(benchmark_json),
            ],
            outputs=[str(benchmark_md), str(benchmark_json)],
            note="Summarizes benchmark JSON artifacts after benchmark scripts have run.",
        ),
    ]
    return steps


def manifest(args: argparse.Namespace, steps: list[PipelineStep]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": args.model_name,
        "source_gguf": str(args.source_gguf),
        "imatrix": str(args.imatrix),
        "output_dir": str(args.output_dir),
        "base_type": args.base_type,
        "ablate_type": args.ablate_type,
        "domains": args.domains,
        "weights": args.weights,
        "steps": [
            {
                "name": step.name,
                "kind": step.kind,
                "gpu": step.gpu,
                "command": step.command,
                "shell": step.shell(),
                "outputs": step.outputs,
                "note": step.note,
            }
            for step in steps
        ],
    }


def write_runner(path: Path, steps: list[PipelineStep], execute: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if not execute:
        lines.extend(
            [
                "echo 'Dry-run runner. Review commands, then re-run planner with --execute-runner to remove exits.'",
                "exit 0",
                "",
            ]
        )
    for step in steps:
        lines.extend([f"echo '=== {step.name} ==='", step.shell(), ""])
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gguf", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--tensors-file", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-type", default="Q4_K_M")
    parser.add_argument("--ablate-type", default="Q3_K")
    parser.add_argument("--domains", default=DEFAULT_DOMAINS)
    parser.add_argument("--weights", default="wiki:0.25,code:0.25,math:0.25,dialogue:0.25")
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--ppl-workers", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--quantize-bin", default="/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize")
    parser.add_argument("--baseline-model", default="baseline")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--execute-runner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = build_steps(args)
    data = manifest(args, steps)

    manifest_path = args.manifest or args.output_dir / "cerebellum_pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")

    runner_path = args.runner or args.output_dir / "run_cerebellum_pipeline.sh"
    write_runner(runner_path, steps, execute=args.execute_runner)

    print(f"manifest -> {manifest_path}")
    print(f"runner -> {runner_path}")
    print(f"steps: {len(steps)}")


if __name__ == "__main__":
    main()
