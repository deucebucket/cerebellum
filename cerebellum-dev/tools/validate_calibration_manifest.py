#!/usr/bin/env python3
"""Validate Cerebellum calibration dataset manifests without downloading data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {"manifest_version", "policy", "seed"}
ENTRY_SECTIONS = ("text_corpora", "sparse_upcycling_jsonl")


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_entry(section: str, entry: dict[str, Any], ids: set[str], root: Path, errors: list[str], warnings: list[str]) -> None:
    prefix = f"{section}/{entry.get('id', '<missing-id>')}"
    for key in ("id", "hf_repo", "output"):
        require(key in entry and entry[key], errors, f"{prefix}: missing required field '{key}'")
    if "id" in entry:
        require(entry["id"] not in ids, errors, f"{prefix}: duplicate id '{entry['id']}'")
        ids.add(entry["id"])
    require("filename" in entry or "filename_template" in entry, errors, f"{prefix}: missing filename or filename_template")
    if "builder" in entry:
        builder = root / entry["builder"]
        if not builder.exists():
            warnings.append(f"{prefix}: builder path does not exist locally: {entry['builder']}")
    if "target_bytes" in entry:
        require(int(entry["target_bytes"]) > 0, errors, f"{prefix}: target_bytes must be positive")
    if "rows" in entry:
        require(int(entry["rows"]) > 0, errors, f"{prefix}: rows must be positive")


def validate_manifest(path: Path, root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    data = json.loads(path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        require(key in data, errors, f"missing top-level field '{key}'")
    require(data.get("policy") == "reuse_public_datasets_only", errors, "policy must be reuse_public_datasets_only")

    ids: set[str] = set()
    for section in ENTRY_SECTIONS:
        for entry in data.get(section, []):
            validate_entry(section, entry, ids, root, errors, warnings)

    aliases = data.get("aliases") or {}
    for alias, members in aliases.items():
        require(isinstance(members, list), errors, f"aliases/{alias}: must be a list")
        for member in members if isinstance(members, list) else []:
            require(member in ids, errors, f"aliases/{alias}: unknown dataset id '{member}'")

    return data, errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, errors, warnings = validate_manifest(args.manifest, args.repo_root)
    summary = {
        "manifest": str(args.manifest),
        "name": data.get("name"),
        "text_corpora": len(data.get("text_corpora", [])),
        "sparse_upcycling_jsonl": len(data.get("sparse_upcycling_jsonl", [])),
        "aliases": len(data.get("aliases", {})),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
