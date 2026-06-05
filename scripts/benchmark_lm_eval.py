"""Cerebellum adapter for lm-evaluation-harness frontier tasks."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def find_metric(data: Any) -> tuple[str | None, float | None]:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"acc", "acc_norm", "exact_match", "f1", "score"} and isinstance(value, (int, float)):
                return key, float(value) * 100.0 if 0.0 <= float(value) <= 1.0 else float(value)
        for value in data.values():
            metric, score = find_metric(value)
            if metric is not None:
                return metric, score
    elif isinstance(data, list):
        for value in data:
            metric, score = find_metric(value)
            if metric is not None:
                return metric, score
    return None, None


def newest_json(root: Path) -> Path | None:
    files = [path for path in root.rglob("*.json") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    model = env_value("BENCH_MODEL", "model")
    port = int(env_value("BENCH_PORT", "8080"))
    workers = int(env_value("BENCH_WORKERS", "1"))
    results_dir = Path(env_value("RESULTS_DIR", "benchmark_results"))
    task = env_value("LM_EVAL_TASK", "mmlu_pro")
    benchmark = env_value("BENCHMARK_NAME", task)
    limit = env_value("BENCH_LIMIT")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = results_dir / f"{model}_{benchmark}_lm_eval"
    summary_path = results_dir / f"{model}_{benchmark}_results.json"

    model_args = (
        f"model={model},"
        f"base_url=http://127.0.0.1:{port}/v1/chat/completions,"
        f"num_concurrent={workers},max_retries=3,tokenized_requests=False"
    )
    cmd = [
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--apply_chat_template",
        "--batch_size",
        "auto",
        "--log_samples",
        "--output_path",
        str(output_dir),
    ]
    if limit:
        cmd.extend(["--limit", limit])

    started = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - started
    raw_path = newest_json(output_dir)
    raw: dict[str, Any] = {}
    if raw_path:
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
    metric, value = find_metric(raw)
    payload = {
        "benchmark": benchmark,
        "model": model,
        "task": task,
        "harness": "lm-evaluation-harness",
        "command": " ".join(cmd),
        "status": "completed" if proc.returncode == 0 else "failed",
        "metric": metric,
        "score": value,
        "elapsed_seconds": round(elapsed, 3),
        "output_dir": str(output_dir),
        "raw_result": str(raw_path) if raw_path else None,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
