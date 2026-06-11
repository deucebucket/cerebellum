#!/usr/bin/env python3
"""EvalPlus (HumanEval+) benchmark for Cerebellum GGUFs.

Replaces the old HumanEval benchmark. Uses 80x more test cases per problem,
catches shortcut solutions that pass original HumanEval.

Requires the .bench-venv (Python 3.12) for evalplus/datasets compatibility.
Run via: .bench-venv/bin/python benchmark_evalplus.py [max_problems]
"""

import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
from evalplus.data import get_human_eval_plus, write_jsonl

PORT = int(os.environ.get("BENCH_PORT", 8084))
MODEL_NAME = os.environ.get("BENCH_MODEL", "cerebellum_v4")
# HumanEval+ runs sequentially. Code-completion benchmarks must not be
# parallelized — multi-slot llama-server delivers non-deterministic completions
# at temp=0 (slot affinity isn't fixed across runs), which breaks reproducibility.
# Override BENCH_WORKERS at your own risk; published rows must be from WORKERS=1.
WORKERS = int(os.environ.get("BENCH_WORKERS", 1))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", 4096))
API_URL = f"http://127.0.0.1:{PORT}/v1/completions"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).parent.parent / "osmosis-qwen35-122b" / "benchmark_results")))
SAMPLES_FILE = RESULTS_DIR / f"{MODEL_NAME}_evalplus_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / f"{MODEL_NAME}_evalplus_results.json"

# Stop sequences: catch where a function body should end. Includes "\n```"
# because chat-tuned models (granite-4.1, qwen) emit closing markdown fences
# even in completions mode — without this, the fence ends up in the function
# body and the resulting "function" fails to parse, scoring as wrong even
# when the code is correct.
STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\n# Test", "\nprint(", "\n```", "\n\n\n"]


def strip_trailing_fence(text: str) -> str:
    """Remove trailing markdown fences and anything after them.

    Defense-in-depth on top of the stop sequence. If a fence slipped through
    (e.g. mid-line), cut it off. Models sometimes emit ``` followed by an
    "Explanation" or test imports — none of that belongs in the function body.
    """
    # Find a line that is just a markdown fence (with optional language tag) — cut from there
    lines = text.split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "```" or stripped.startswith("```"):
            cut = i
            break
    return "\n".join(lines[:cut]).rstrip()


def generate_completion(prompt: str, retries=3) -> str:
    for attempt in range(retries):
        try:
            resp = httpx.post(
                API_URL,
                json={
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0,
                    "stop": STOP_SEQUENCES,
                },
                timeout=300.0,
            )
            resp.raise_for_status()
            break
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if attempt == retries - 1:
                # abort loudly: a fabricated "pass" scores as a model failure
                # and silently corrupts the run (2026-06-11 audit, BE-15)
                raise RuntimeError(f"API failed after {retries} attempts: {e}") from e
            time.sleep(2 * (attempt + 1))
    data = resp.json()
    content = data["choices"][0]["text"]
    content = strip_trailing_fence(content)
    return content


def load_existing() -> dict:
    done = {}
    if SAMPLES_FILE.exists():
        for line in SAMPLES_FILE.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["task_id"]] = entry
    return done


def pass_rates_from_evalplus(eval_data: dict) -> tuple[float | None, float | None]:
    """Parse both summary and per-task EvalPlus result formats."""
    base_pass = None
    plus_pass = None

    # Newer evalplus format: {"eval": {"HumanEval/0": [{"base_status": ...}]}}
    if "eval" in eval_data:
        samples = [
            sample
            for task_samples in eval_data["eval"].values()
            for sample in task_samples
        ]
        if samples:
            base_pass = sum(s.get("base_status") == "pass" for s in samples) / len(samples)
            plus_pass = sum(s.get("plus_status") == "pass" for s in samples) / len(samples)
            return base_pass, plus_pass

    # Older evalplus summary format:
    # {"humaneval": {"pass@1": X}, "humaneval+": {"pass@1": Y}}
    if "humaneval" in eval_data:
        base_pass = eval_data["humaneval"].get("pass@1")
    elif "pass@1" in eval_data:
        base_pass = eval_data["pass@1"]
    if "humaneval+" in eval_data:
        plus_pass = eval_data["humaneval+"].get("pass@1")

    return base_pass, plus_pass


def main():
    max_q = None
    if len(sys.argv) > 1:
        max_q = int(sys.argv[1])

    print(f"EVALPLUS (HumanEval+) BENCHMARK — {MODEL_NAME}", flush=True)
    print("Loading problems...", flush=True)

    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
    if max_q:
        task_ids = task_ids[:max_q]
    print(f"Loaded {len(task_ids)} problems", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done = load_existing()
    skip = len(done)
    if skip > 0:
        print(f"Resuming from {skip} existing completions", flush=True)

    remaining = [(tid, problems[tid]) for tid in task_ids if tid not in done]
    print(f"Workers: {WORKERS}, remaining: {len(remaining)}", flush=True)

    t0 = time.time()
    write_lock = threading.Lock()
    completed = skip

    def process(tid_problem):
        tid, problem = tid_problem
        prompt = problem["prompt"]
        completion = generate_completion(prompt)
        return {
            "task_id": tid,
            "solution": prompt + completion,
            "completion": completion,
        }

    with open(SAMPLES_FILE, "a") as f:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for entry in pool.map(process, remaining):
                with write_lock:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    completed += 1
                    elapsed = time.time() - t0
                    done_this_run = completed - skip
                    rate = done_this_run / elapsed if elapsed > 0 else 0
                    remaining_count = len(task_ids) - completed
                    eta = remaining_count / rate if rate > 0 else 0
                    print(f"  [{completed:3d}/{len(task_ids)}] {rate:.2f} q/s, ETA {eta:.0f}s — {entry['task_id']}", flush=True)

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {len(task_ids)} problems in {elapsed:.0f}s", flush=True)
    print(f"Samples: {SAMPLES_FILE}", flush=True)
    print(f"\nRunning EvalPlus evaluation...", flush=True)

    # Run evalplus evaluation — it writes results to a _eval_results.json next to samples
    from evalplus.evaluate import evaluate
    evaluate(dataset="humaneval", samples=str(SAMPLES_FILE))

    # Parse the results file evalplus writes
    eval_results_path = SAMPLES_FILE.with_name(SAMPLES_FILE.stem + "_eval_results.json")
    if not eval_results_path.exists():
        # evalplus sometimes puts it in a subdirectory
        candidates = list(RESULTS_DIR.glob("*eval_results*"))
        if candidates:
            eval_results_path = candidates[-1]

    base_pass = None
    plus_pass = None
    if eval_results_path.exists():
        eval_data = json.loads(eval_results_path.read_text())
        base_pass, plus_pass = pass_rates_from_evalplus(eval_data)

    summary = {
        "benchmark": "evalplus_humaneval_plus",
        "model": MODEL_NAME,
        "pass_at_1_plus": round(plus_pass * 100, 2) if plus_pass else None,
        "pass_at_1_base": round(base_pass * 100, 2) if base_pass else None,
        "total_problems": len(task_ids),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRESULTS:", flush=True)
    if base_pass is not None:
        print(f"  HumanEval (base):  {base_pass*100:.1f}%", flush=True)
    if plus_pass is not None:
        print(f"  HumanEval+ (hard): {plus_pass*100:.1f}%", flush=True)
    print(f"\nSaved to {RESULTS_FILE}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
