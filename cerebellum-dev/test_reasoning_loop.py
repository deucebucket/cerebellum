#!/usr/bin/env python3
"""
Test for reasoning loop tendency in Qwen 3.6 27B quants.
Sends prompts that trigger thinking mode and checks:
  1. Does the model exit <think> tags cleanly?
  2. Does it degenerate into "I will be X" repetition?
  3. How many tokens does it waste thinking vs answering?

Used to compare different quant configurations and find which
tensor overrides reduce loop tendency.
"""
import json
import time
import requests
import sys
from collections import Counter

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8084
URL = f"http://127.0.0.1:{PORT}/v1/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "test"

LOOP_PROMPTS = [
    # Code review (the scenario from Reddit)
    {
        "system": "You are a helpful coding assistant. Review code for bugs and security issues. Be concise.",
        "user": 'Check this bash script:\n```bash\n#!/bin/bash\nread -p "Enter filename: " FILE\ncat $FILE | grep -i error > /tmp/errors.txt\nchmod 777 /tmp/errors.txt\n```',
    },
    # Short question (reported to loop more)
    {
        "system": "You are a helpful assistant.",
        "user": "What is the capital of France?",
    },
    # Coding task
    {
        "system": "You are an expert Python developer.",
        "user": "Write a function to check if a string is a valid IPv4 address.",
    },
    # Math reasoning
    {
        "system": "You are a math tutor.",
        "user": "Prove that the square root of 2 is irrational.",
    },
    # Multi-step reasoning
    {
        "system": "You are a helpful assistant.",
        "user": "I have a 3-gallon jug and a 5-gallon jug. How do I measure exactly 4 gallons?",
    },
    # Ambiguous/open-ended (most likely to loop)
    {
        "system": "You are an AI assistant.",
        "user": "What are your capabilities?",
    },
    # Long code to review
    {
        "system": "Review this code for issues.",
        "user": """```python
def process_data(items):
    results = []
    for i in range(len(items)):
        if items[i] > 0:
            results.append(items[i] * 2)
        else:
            results.append(0)
    return results

def fetch_url(url):
    import urllib.request
    response = urllib.request.urlopen(url)
    return response.read().decode()
```""",
    },
    # Instruction following under pressure
    {
        "system": "Answer in exactly one sentence. No thinking out loud.",
        "user": "Explain quantum entanglement.",
    },
]


def test_prompt(prompt_data, presence_penalty=0.0, max_tokens=2048):
    messages = f'<|im_start|>system\n{prompt_data["system"]}<|im_end|>\n<|im_start|>user\n{prompt_data["user"]}<|im_end|>\n<|im_start|>assistant\n'

    try:
        resp = requests.post(URL, json={
            "prompt": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": presence_penalty,
            "repeat_penalty": 1.0,
            "stop": ["<|im_end|>"],
        }, timeout=120)
        data = resp.json()
        text = data["choices"][0]["text"]
        finish = data["choices"][0].get("finish_reason", "?")
        tokens = data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        return {"error": str(e)}

    # Analyze
    think_end = text.find("</think>")
    has_think_exit = think_end > 0
    think_tokens = think_end if has_think_exit else len(text)
    answer = text[think_end + 8:].strip() if has_think_exit else ""

    # Detect "I will be" degeneration
    i_will_count = text.count("I will")
    loop_detected = i_will_count > 5

    # Detect any repeated line (3+ times)
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
    line_counts = Counter(lines)
    repeated_lines = [(l, c) for l, c in line_counts.most_common(3) if c >= 3]

    return {
        "tokens": tokens,
        "finish": finish,
        "has_think_exit": has_think_exit,
        "think_chars": think_tokens,
        "answer_chars": len(answer),
        "loop_i_will": loop_detected,
        "i_will_count": i_will_count,
        "repeated_lines": len(repeated_lines),
        "answer_preview": answer[:200] if answer else text[-200:],
    }


def run_suite(presence_penalty=0.0):
    print(f"\n{'='*70}")
    print(f"REASONING LOOP TEST — {MODEL} (presence_penalty={presence_penalty})")
    print(f"{'='*70}\n")

    results = []
    for i, prompt in enumerate(LOOP_PROMPTS):
        result = test_prompt(prompt, presence_penalty=presence_penalty)
        results.append(result)

        status = "OK" if result.get("has_think_exit") else "STUCK"
        if result.get("loop_i_will"):
            status = "LOOP"
        if result.get("error"):
            status = "ERROR"

        print(f"  [{i+1}/{len(LOOP_PROMPTS)}] {status:5s} | "
              f"tok={result.get('tokens', 0):4d} | "
              f"think={result.get('think_chars', 0):5d} | "
              f"answer={result.get('answer_chars', 0):4d} | "
              f"{prompt['user'][:50]}")

        if result.get("loop_i_will"):
            print(f"         ^^^ I WILL loop ({result['i_will_count']}x)")
        if result.get("repeated_lines"):
            print(f"         ^^^ Repeated lines detected")

    # Summary
    total = len(results)
    clean = sum(1 for r in results if r.get("has_think_exit"))
    loops = sum(1 for r in results if r.get("loop_i_will"))
    stuck = sum(1 for r in results if not r.get("has_think_exit") and not r.get("loop_i_will") and not r.get("error"))

    print(f"\n  SUMMARY: {clean}/{total} clean exits, {stuck} stuck in think, {loops} loops")
    return results


if __name__ == "__main__":
    # Test without presence penalty (should show the problem)
    results_base = run_suite(presence_penalty=0.0)

    # Test with presence penalty 1.5 (should fix it)
    results_fixed = run_suite(presence_penalty=1.5)

    # Save results
    outfile = f"/var/home/deucebucket/ai-drive/osmosis/osmosis-qwen36-27b/benchmark_results/{MODEL}_reasoning_loop.json"
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL,
            "base_results": results_base,
            "fixed_results": results_fixed,
        }, f, indent=2)
    print(f"\nSaved to: {outfile}")
