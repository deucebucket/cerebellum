#!/usr/bin/env python3
"""HumanEval benchmark for llama.cpp models via OpenAI-compatible API.

Writes each sample to disk as it completes. Resumes from existing progress.
"""

import json
import os
import sys
import time
import httpx
from pathlib import Path
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

PORT = int(os.environ.get("BENCH_PORT", 8084))
MODEL_NAME = os.environ.get("BENCH_MODEL", "cerebellum_v4")
API_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
OUTPUT_DIR = Path(__file__).parent.parent / "osmosis-qwen36-27b" / "benchmark_results"
SAMPLES_FILE = OUTPUT_DIR / f"{MODEL_NAME}_humaneval_samples.jsonl"
RESULTS_FILE = OUTPUT_DIR / f"{MODEL_NAME}_humaneval_results.json"


def generate_completion(prompt: str) -> str:
    resp = httpx.post(
        API_URL,
        json={
            "messages": [
                {"role": "system", "content":
                    "You are a Python coding assistant. Complete the function below. "
                    "Output ONLY the function body code, no explanations, no markdown, "
                    "no ```python blocks. Just the raw Python code that completes the function."},
                {"role": "user", "content":
                    f"Complete this Python function. Return ONLY the implementation code "
                    f"(the function body), nothing else:\n\n{prompt}"},
            ],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return normalize_indent(content)


def normalize_indent(completion: str) -> str:
    """Ensure completion is indented at 4 spaces (HumanEval function body level).

    Models sometimes return code at indent 0 or re-emit the function signature.
    Strip any function def/docstring prefix, then re-indent the body to 4 spaces.
    """
    lines = completion.split("\n")

    # Strip leading function signature if the model re-emitted it
    start = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            start = i + 1
            # Skip docstring after def
            if start < len(lines):
                doc_line = lines[start].lstrip()
                if doc_line.startswith('"""') or doc_line.startswith("'''"):
                    quote = doc_line[:3]
                    if doc_line.count(quote) >= 2:
                        start += 1
                    else:
                        for j in range(start + 1, len(lines)):
                            if quote in lines[j]:
                                start = j + 1
                                break
            break

    body_lines = lines[start:]
    if not body_lines:
        return completion

    # Find the minimum indentation of non-empty lines
    min_indent = None
    for line in body_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if min_indent is None or indent < min_indent:
            min_indent = indent

    if min_indent is None:
        return completion

    # Re-indent to 4 spaces
    result = []
    for line in body_lines:
        if not line.strip():
            result.append("")
        else:
            result.append("    " + line[min_indent:])

    return "\n".join(result)


def load_existing() -> dict[str, dict]:
    done = {}
    if SAMPLES_FILE.exists():
        for line in SAMPLES_FILE.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["task_id"]] = entry
    return done


def main():
    problems = read_problems()

    # Resume from existing progress
    done = load_existing()
    remaining = {k: v for k, v in problems.items() if k not in done}

    print(f"HumanEval: {len(problems)} total, {len(done)} already done, {len(remaining)} remaining", flush=True)
    if not remaining:
        print("All problems complete, running evaluation...", flush=True)
    else:
        print(f"API: {API_URL}", flush=True)

        try:
            r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5.0)
            if r.json().get("status") != "ok":
                print(f"Model not ready: {r.text}", flush=True)
                return
        except Exception as e:
            print(f"Model not reachable: {e}", flush=True)
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        start = time.time()
        completed = 0
        errors = 0

        with open(SAMPLES_FILE, "a") as f:
            for task_id, problem in remaining.items():
                try:
                    completion = generate_completion(problem["prompt"])
                    entry = {"task_id": task_id, "completion": completion}
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 10 == 0:
                        elapsed = time.time() - start
                        rate = completed / elapsed * 60
                        eta = (len(remaining) - completed) / (completed / elapsed)
                        print(f"  [{len(done)+completed}/{len(problems)}] {rate:.0f}/min, ETA {eta:.0f}s", flush=True)
                except Exception as e:
                    print(f"  ERROR {task_id}: {e}", flush=True)
                    entry = {"task_id": task_id, "completion": ""}
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    errors += 1

        elapsed = time.time() - start
        print(f"\nGeneration: {completed} ok, {errors} errors, {elapsed:.0f}s", flush=True)

    # Evaluate
    print("\nRunning evaluation (executing code)...", flush=True)
    results = evaluate_functional_correctness(str(SAMPLES_FILE))
    pct = results["pass@1"] * 100

    print(f"\n{'='*50}", flush=True)
    print(f"HumanEval pass@1: {pct:.1f}%", flush=True)
    print(f"{'='*50}", flush=True)

    result_data = {
        "benchmark": "humaneval",
        "model": MODEL_NAME,
        "pass_at_1": results["pass@1"],
        "pass_at_1_pct": round(pct, 1),
        "total_problems": len(problems),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "settings": {
            "temperature": 0,
            "max_tokens": 512,
            "enable_thinking": False,
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
