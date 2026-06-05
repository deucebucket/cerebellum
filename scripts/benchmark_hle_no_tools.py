"""Cerebellum adapter for CAIS Humanity's Last Exam no-tools evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def pct_value(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number * 100.0 if 0.0 <= number <= 1.0 else number


def metric_payload_from_mapping(data: dict) -> dict[str, object]:
    for key in ["score", "accuracy", "final_score", "hle_score"]:
        if key in data:
            score = pct_value(data.get(key))
            if score is not None:
                return {"score": round(score, 4), "metric_source": key}
    correct = data.get("correct")
    total = data.get("total") or data.get("num_total") or data.get("n")
    try:
        correct_n = int(correct)  # type: ignore[arg-type]
        total_n = int(total)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    if total_n <= 0:
        return {}
    return {
        "score": round((correct_n / total_n) * 100.0, 4),
        "correct": correct_n,
        "total": total_n,
        "metric_source": "correct/total",
    }


def extract_hle_metrics(*texts: str) -> dict[str, object]:
    """Extract a leaderboard-friendly HLE score from common judge outputs."""
    for text in texts:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            candidates = [stripped]
            if "{" in stripped and "}" in stripped:
                candidates.append(stripped[stripped.find("{"): stripped.rfind("}") + 1])
            for candidate in candidates:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    metric = metric_payload_from_mapping(data)
                    if metric:
                        return metric
    return {}


def main() -> None:
    model = env_value("BENCH_MODEL", "model")
    port = int(env_value("BENCH_PORT", "8080"))
    workers = int(env_value("BENCH_WORKERS", "4"))
    max_tokens = int(env_value("BENCH_MAX_TOKENS", "8192"))
    results_dir = Path(env_value("RESULTS_DIR", "benchmark_results"))
    dataset = env_value("HLE_DATASET", "cais/hle")
    hle_dir = env_value("HLE_EVAL_DIR")
    judge_model = env_value("HLE_JUDGE_MODEL")
    max_samples = env_value("BENCH_LIMIT")
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions = results_dir / f"{model}_hle_no_tools_predictions.json"
    summary = results_dir / f"{model}_hle_no_tools_results.json"

    base_env = os.environ.copy()
    base_env.setdefault("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    base_env.setdefault("OPENAI_API_KEY", "cerebellum-local")
    cmd = [
        "python",
        "run_model_predictions.py" if hle_dir else "-m",
        "hle_eval.run_model_predictions" if not hle_dir else "",
        "--dataset",
        dataset,
        "--model",
        model,
        "--max_completion_tokens",
        str(max_tokens),
        "--num_workers",
        str(workers),
    ]
    cmd = [part for part in cmd if part]
    if max_samples:
        cmd.extend(["--max_samples", max_samples])
    started = time.time()
    proc = subprocess.run(cmd, cwd=hle_dir or None, env=base_env, text=True, capture_output=True)
    elapsed = time.time() - started

    judge_cmd: list[str] | None = None
    judge_proc: subprocess.CompletedProcess[str] | None = None
    if proc.returncode == 0 and judge_model:
        judge_cmd = [
            "python",
            "run_judge_results.py" if hle_dir else "-m",
            "hle_eval.run_judge_results" if not hle_dir else "",
            "--dataset",
            dataset,
            "--predictions",
            str(predictions),
            "--num_workers",
            str(workers),
            "--model",
            judge_model,
        ]
        judge_cmd = [part for part in judge_cmd if part]
        judge_proc = subprocess.run(judge_cmd, cwd=hle_dir or None, env=base_env, text=True, capture_output=True)

    payload = {
        "benchmark": "hle_no_tools",
        "model": model,
        "dataset": dataset,
        "harness": "centerforaisafety/hle",
        "status": "completed" if proc.returncode == 0 and (judge_proc is None or judge_proc.returncode == 0) else "failed",
        "command": " ".join(cmd),
        "judge_command": " ".join(judge_cmd) if judge_cmd else None,
        "predictions": str(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "judge_stdout_tail": (judge_proc.stdout[-4000:] if judge_proc else ""),
        "judge_stderr_tail": (judge_proc.stderr[-4000:] if judge_proc else ""),
    }
    payload.update(extract_hle_metrics(payload["judge_stdout_tail"], payload["judge_stderr_tail"], payload["stdout_tail"], payload["stderr_tail"]))
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "completed":
        raise SystemExit(proc.returncode or (judge_proc.returncode if judge_proc else 1))


if __name__ == "__main__":
    main()
