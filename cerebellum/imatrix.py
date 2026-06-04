"""Public Cerebellum imatrix entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from osmosis.hillstep import default_data_root, slug


def project_source_root(data_root: Path, family: str, model_name: str, source_name: str) -> Path:
    return data_root / "families" / slug(family) / slug(model_name) / "sources" / slug(source_name)


def write_project_manifest(
    source_root: Path,
    args: argparse.Namespace,
    output: Path,
    next_command: str,
) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cerebellum.project_source.v1",
        "tool": "cerebellum",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "family": args.family,
        "model_name": args.model_name,
        "source_name": args.source_name,
        "hf_or_local_model": args.model,
        "source_gguf": args.source_gguf,
        "imatrix": str(output),
        "imatrix_mode": args.mode,
        "data_root": str(args.data_root),
        "next_command": next_command,
        "layout": {
            "source_root": str(source_root),
            "imatrix_dir": str(source_root / "imatrix"),
            "runs_dir": str(source_root / "runs"),
            "reports_dir": str(source_root / "reports"),
        },
    }
    path = source_root / "cerebellum_project.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Cerebellum/llama.cpp imatrix for quantization"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--output", default=None, help="Output imatrix file path; defaults to the Cerebellum project imatrix dir")
    parser.add_argument("--data-root", default=None, help="Cerebellum project data root")
    parser.add_argument("--family", default=None, help="Model family for Cerebellum project layout")
    parser.add_argument("--model-name", default=None, help="Model name for Cerebellum project layout")
    parser.add_argument("--source-name", default=None, help="Source name for Cerebellum project layout")
    parser.add_argument("--source-gguf", default=None, help="Optional source GGUF for printing the next cerebellum run command")
    parser.add_argument("--profile", default="wiki", help="PPL profile for the printed next cerebellum run command")
    parser.add_argument("--corpus", default=None, help="Optional corpus for the printed next cerebellum run command")
    parser.add_argument(
        "--mode",
        choices=["stream", "calibrated"],
        default="stream",
        help="stream uses safetensors one tensor at a time; calibrated loads the model and blends activation stats",
    )
    parser.add_argument("--no-calibrate", action="store_true", help="For --mode calibrated, skip activation calibration")
    parser.add_argument("--num-samples", type=int, default=8, help="Calibration prompt count for --mode calibrated")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = Path(args.data_root) if args.data_root else default_data_root()
    args.family = args.family or "unknown-family"
    args.model_name = args.model_name or slug(Path(str(args.model)).name)
    args.source_name = args.source_name or "hf-safetensors"

    source_root = project_source_root(args.data_root, args.family, args.model_name, args.source_name)
    output = Path(args.output) if args.output else source_root / "imatrix" / "cerebellum_imatrix.dat"
    output.parent.mkdir(parents=True, exist_ok=True)
    next_parts = [
        "cerebellum",
        "run",
        "--source-gguf",
        args.source_gguf or "SOURCE.gguf",
        "--imatrix",
        str(output),
        "--profile",
        args.profile,
        "--family",
        slug(args.family),
        "--model-name",
        slug(args.model_name),
        "--source-name",
        slug(args.source_name),
        "--data-root",
        str(args.data_root),
    ]
    if args.corpus:
        next_parts.extend(["--corpus", args.corpus])
    next_command = " ".join(next_parts)

    if args.mode == "stream":
        from osmosis.imatrix_stream import generate_imatrix_streaming

        generate_imatrix_streaming(args.model, str(output), verbose=args.verbose)
    else:
        from osmosis.imatrix_gen import generate_imatrix

        generate_imatrix(
            args.model,
            str(output),
            calibrate=not args.no_calibrate,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )

    write_project_manifest(source_root, args, output, next_command)
    print()
    print("Cerebellum project updated")
    print(f"  source root : {source_root}")
    print(f"  imatrix     : {output}")
    print("Next:")
    print(f"  {next_command}")


if __name__ == "__main__":
    main()
