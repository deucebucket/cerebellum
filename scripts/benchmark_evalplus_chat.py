#!/usr/bin/env python3
"""EvalPlus (HumanEval+) through llama-server chat completions.

Gemma 4 is a chat/thinking model. The plain completion endpoint can produce
garbled continuations because it bypasses the chat template and reasoning
parser. This harness asks for a function body through /v1/chat/completions,
extracts the final code block, and evaluates with EvalPlus.
"""

import json
import os
import re
import sys
import time
import threading
import ast
import textwrap
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
from evalplus.data import get_human_eval_plus

PORT = int(os.environ.get("BENCH_PORT", 8084))
MODEL_NAME = os.environ.get("BENCH_MODEL", "cerebellum_chat")
WORKERS = int(os.environ.get("BENCH_WORKERS", 1))
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", 1536))
THINKING_BUDGET = int(os.environ.get("BENCH_THINKING_BUDGET", 256))
ENABLE_THINKING = os.environ.get("BENCH_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}
API_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).parent.parent / "benchmark_results")))
SAMPLES_FILE = RESULTS_DIR / f"{MODEL_NAME}_evalplus_chat_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / f"{MODEL_NAME}_evalplus_chat_results.json"


PROMPT_PREFIX = """Complete the following Python function.

Return only the function body, with indentation that fits inside the provided
function. Put the final answer inside <final_code> and </final_code>. Do not
include markdown fences.

"""


def strip_trailing_fence(text: str) -> str:
    lines = text.split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "```" or stripped.startswith("```"):
            cut = i
            break
    return "\n".join(lines[:cut]).rstrip()


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _indent_body(body: str) -> str:
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    pass"

    body = "\n".join(lines)
    dedented = textwrap.dedent(body).rstrip()
    return "\n".join(("    " + line if line.strip() else line) for line in dedented.splitlines()).rstrip()


def _shift_after_first_line(body: str) -> str:
    """Repair common Gemma body output where lines after the first are overindented."""
    lines = body.splitlines()
    shifted = []
    seen_first = False
    for line in lines:
        if not line.strip():
            shifted.append(line)
            continue
        if not seen_first:
            shifted.append(line)
            seen_first = True
            continue
        indent = _indent_width(line)
        shifted.append(line[4:] if indent >= 4 else line)
    return "\n".join(shifted).rstrip()


def _is_valid_solution(prompt: str | None, body: str) -> bool:
    if not prompt:
        return False
    try:
        ast.parse(prompt + body)
        return True
    except SyntaxError:
        return False


def _prompt_function_name(prompt: str | None) -> str | None:
    if not prompt:
        return None
    match = re.search(r"^\s*def\s+(\w+)\(", prompt, flags=re.MULTILINE)
    return match.group(1) if match else None


def normalize_body(text: str, prompt: str | None = None) -> str:
    content = text.strip()

    tag_match = re.search(r"<final_code>\s*(.*?)\s*</final_code>", content, flags=re.DOTALL | re.IGNORECASE)
    if tag_match:
        content = tag_match.group(1).strip("\n")
    elif re.match(r"\s*<final_code>", content, flags=re.IGNORECASE):
        content = re.sub(r"^\s*<final_code>\s*", "", content, flags=re.IGNORECASE).strip("\n")
    else:
        fence_match = re.search(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            content = fence_match.group(1).strip("\n")
        content = strip_trailing_fence(content)

    # If the model repeated the full function, keep the indented body after the
    # first docstring. This is a fallback for chat models that ignore "body only".
    first_def = re.match(r"^\s*def\s+(\w+)\(", content)
    if first_def and first_def.group(1) == _prompt_function_name(prompt):
        parts = content.split('"""')
        if len(parts) >= 3:
            content = '"""'.join(parts[2:]).strip("\n")
        else:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.lstrip().startswith(("return ", "for ", "if ", "while ", "res =", "result =", "ans =")):
                    content = "\n".join(lines[i:])
                    break

    lines = content.splitlines()
    # Drop common wrappers or commentary that leaked into content.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and lines[0].strip().lower() in {"python", "final code:", "final answer:"}:
        lines.pop(0)

    body = "\n".join(lines).rstrip()
    if not body:
        return "    pass"

    # Try syntax-valid variants before falling back. Gemma often returns a
    # body where the first line is correctly indented and all later lines are
    # shifted one level too deep, which EvalPlus counts as syntax failure.
    candidates = []
    shifted_once = _shift_after_first_line(body)
    shifted_twice = _shift_after_first_line(shifted_once)
    dedented = textwrap.dedent(body)
    dedented_shifted_once = _shift_after_first_line(dedented)
    dedented_shifted_twice = _shift_after_first_line(dedented_shifted_once)
    for candidate in (
        body,
        shifted_once,
        shifted_twice,
        dedented,
        dedented_shifted_once,
        dedented_shifted_twice,
    ):
        normalized = _indent_body(candidate)
        if normalized not in candidates:
            candidates.append(normalized)

    for candidate in candidates:
        if _is_valid_solution(prompt, candidate):
            return candidate

    return candidates[0]


def generate_completion(prompt: str, retries=3) -> tuple[str, str, str]:
    user_prompt = PROMPT_PREFIX + prompt
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        "thinking_budget_tokens": THINKING_BUDGET if ENABLE_THINKING else 0,
    }
    last_error = None
    for attempt in range(retries):
        try:
            resp = httpx.post(API_URL, json=payload, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""
            return normalize_body(content, prompt), content, reasoning
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    # abort loudly: a fabricated "pass" scores as a model failure and
    # silently corrupts the run (2026-06-11 audit, BE-15)
    raise RuntimeError(f"API failed after {retries} attempts: {last_error}")


def load_existing() -> dict:
    done = {}
    if SAMPLES_FILE.exists():
        for line in SAMPLES_FILE.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["task_id"]] = entry
    return done


def pass_rates_from_evalplus(eval_data: dict) -> tuple[float | None, float | None]:
    if "eval" in eval_data:
        samples = [sample for task_samples in eval_data["eval"].values() for sample in task_samples]
        if samples:
            base_pass = sum(s.get("base_status") == "pass" for s in samples) / len(samples)
            plus_pass = sum(s.get("plus_status") == "pass" for s in samples) / len(samples)
            return base_pass, plus_pass
    return None, None


def main():
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"EVALPLUS CHAT BENCHMARK - {MODEL_NAME}", flush=True)
    print(
        f"Workers: {WORKERS}, max_tokens: {MAX_TOKENS}, "
        f"enable_thinking: {ENABLE_THINKING}, thinking_budget: {THINKING_BUDGET if ENABLE_THINKING else 0}",
        flush=True,
    )

    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
    if max_q:
        task_ids = task_ids[:max_q]
    print(f"Loaded {len(task_ids)} problems", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_existing()
    skip = len(done)
    remaining = [(tid, problems[tid]) for tid in task_ids if tid not in done]
    if skip:
        print(f"Resuming from {skip} existing completions", flush=True)

    t0 = time.time()
    write_lock = threading.Lock()
    completed = skip

    def process(tid_problem):
        tid, problem = tid_problem
        completion, raw_content, reasoning = generate_completion(problem["prompt"])
        return {
            "task_id": tid,
            "solution": problem["prompt"] + completion,
            "completion": completion,
            "raw_content": raw_content[:2000],
            "reasoning": reasoning[:2000],
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
                    eta = (len(task_ids) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:3d}/{len(task_ids)}] {rate:.2f} q/s, ETA {eta:.0f}s - {entry['task_id']}", flush=True)

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {len(task_ids)} problems in {elapsed:.0f}s", flush=True)
    print(f"Samples: {SAMPLES_FILE}", flush=True)

    from evalplus.evaluate import evaluate
    evaluate(dataset="humaneval", samples=str(SAMPLES_FILE))

    eval_results_path = SAMPLES_FILE.with_name(SAMPLES_FILE.stem + "_eval_results.json")
    base_pass = plus_pass = None
    if eval_results_path.exists():
        base_pass, plus_pass = pass_rates_from_evalplus(json.loads(eval_results_path.read_text()))

    summary = {
        "benchmark": "evalplus_humaneval_plus_chat",
        "model": MODEL_NAME,
        "pass_at_1_base": round(base_pass * 100, 2) if base_pass is not None else None,
        "pass_at_1_plus": round(plus_pass * 100, 2) if plus_pass is not None else None,
        "total_problems": len(task_ids),
        "elapsed_seconds": round(elapsed, 1),
        "enable_thinking": ENABLE_THINKING,
        "thinking_budget_tokens": THINKING_BUDGET if ENABLE_THINKING else 0,
        "max_tokens": MAX_TOKENS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    RESULTS_FILE.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
