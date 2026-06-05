"""Cerebellum adapter for LiveCodeBench release_v6 code generation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def newest_json(root: Path) -> Path | None:
    files = [path for path in root.rglob("*.json") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    model = env_value("BENCH_MODEL", "model")
    port = int(env_value("BENCH_PORT", "8080"))
    results_dir = Path(env_value("RESULTS_DIR", "benchmark_results"))
    max_tokens = int(env_value("BENCH_MAX_TOKENS", "4096"))
    limit = env_value("BENCH_LIMIT")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = results_dir / f"{model}_livecodebench_v6"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = results_dir / f"{model}_livecodebench_v6_results.json"

    env = os.environ.copy()
    env.setdefault("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    env.setdefault("OPENAI_API_KEY", "cerebellum-local")
    cmd = [
        "python",
        "-m",
        "lcb_runner.runner.main",
        "--model",
        model,
        "--scenario",
        "codegeneration",
        "--evaluate",
        "--release_version",
        "release_v6",
        "--max_tokens",
        str(max_tokens),
        "--output_dir",
        str(output_dir),
    ]
    if limit:
        cmd.extend(["--start_date", "2025-04-01", "--end_date", "2025-04-30"])

    started = time.time()
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    elapsed = time.time() - started
    raw_path = newest_json(output_dir)
    payload = {
        "benchmark": "livecodebench_v6",
        "model": model,
        "harness": "LiveCodeBench",
        "release_version": "release_v6",
        "status": "completed" if proc.returncode == 0 else "failed",
        "command": " ".join(cmd),
        "output_dir": str(output_dir),
        "raw_result": str(raw_path) if raw_path else None,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
