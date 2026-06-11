#!/usr/bin/env python3
"""RealWorldQA slice benchmark for local OpenAI-compatible VLM servers."""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx
from datasets import load_dataset


PORT = int(os.environ.get("BENCH_PORT", 7821))
MODEL_NAME = os.environ.get("BENCH_MODEL", "qwen36_35b_cerebellum_v2")
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        str(Path(__file__).parent.parent / "osmosis-qwen36-35b" / "benchmark_results_v2_vision"),
    )
)
API_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
DETAILS_FILE = RESULTS_DIR / f"{MODEL_NAME}_realworldqa_detailed.jsonl"
RESULTS_FILE = RESULTS_DIR / f"{MODEL_NAME}_realworldqa_results.json"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s,.;:!`*_\"']+", " ", text)
    return text.strip()


def first_answer_token(text: str) -> str:
    match = re.search(r"\b([a-d]|yes|no|left|right|middle|center|centre|up|down|green|red|blue|yellow|white|black|gray|grey|bus|car|truck|sedan)\b", normalize(text))
    return match.group(1) if match else normalize(text)


def is_correct(raw: str, expected: str) -> bool:
    raw_norm = normalize(raw)
    exp_norm = normalize(expected)
    if raw_norm == exp_norm:
        return True
    if len(exp_norm) == 1 and exp_norm in "abcd":
        return first_answer_token(raw_norm) == exp_norm
    return exp_norm == first_answer_token(raw_norm) or exp_norm in raw_norm.split()


def image_data_url(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def query(image, question: str) -> str:
    resp = httpx.post(
        API_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                        {"type": "text", "text": question.strip()},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 48,
            # 48-token budget: a <think> block would consume it all and return
            # empty content when the server runs with --jinja
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget_tokens": 0,
        },
        timeout=240,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content", "").strip()


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(os.environ.get("BENCH_SEED", "36635"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"REALWORLDQA BENCHMARK - {MODEL_NAME}", flush=True)
    print(f"Loading xai-org/RealworldQA test split, n={n}, seed={seed}", flush=True)
    ds = load_dataset("xai-org/RealworldQA", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n]

    correct = 0
    t0 = time.time()
    with DETAILS_FILE.open("w") as f:
        for pos, idx in enumerate(indices, 1):
            row = ds[idx]
            question = row["question"]
            expected = str(row["answer"]).strip()
            error = None
            try:
                raw = query(row["image"], question)
            except Exception as exc:
                raw = ""
                error = repr(exc)
            ok = error is None and is_correct(raw, expected)
            correct += int(ok)
            detail = {
                "dataset": "xai-org/RealworldQA",
                "split": "test",
                "index": idx,
                "question": question,
                "expected": expected,
                "raw_response": raw,
                "correct": ok,
                "error": error,
            }
            f.write(json.dumps(detail) + "\n")
            f.flush()
            print(f"[{pos:03d}/{n}] idx={idx} {'OK' if ok else 'MISS'} expected={expected!r} raw={raw!r}", flush=True)

    elapsed = time.time() - t0
    summary = {
        "benchmark": "realworldqa",
        "model": MODEL_NAME,
        "dataset": "xai-org/RealworldQA",
        "split": "test",
        "sample_size": n,
        "seed": seed,
        "accuracy": round(correct / n * 100, 2),
        "correct": correct,
        "total": n,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    RESULTS_FILE.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if correct == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
