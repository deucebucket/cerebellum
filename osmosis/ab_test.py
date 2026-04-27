"""A/B quality comparison between osmosis variants.

Runs identical prompts against two llama-server endpoints and scores
the outputs. Designed to compare e.g. stock Q5_K_M vs osmosis 4GB vs
thinking-preserved variant.

Usage:
    python -m osmosis.ab_test \
        --model-a http://localhost:8095 --label-a "osmosis-4gb" \
        --model-b http://localhost:8096 --label-b "osmosis-think" \
        --output osmosis-qwen3.5-9b/ab_results.json -v
"""
import argparse
import json
import time
from pathlib import Path

import httpx

TEST_SUITE = [
    {
        "category": "factual",
        "prompt": "What are the three laws of thermodynamics? Be concise.",
        "check": lambda r: "entropy" in r.lower() and "energy" in r.lower(),
    },
    {
        "category": "factual",
        "prompt": "Who wrote 'One Hundred Years of Solitude' and what country were they from?",
        "check": lambda r: "marquez" in r.lower() or "márquez" in r.lower(),
    },
    {
        "category": "factual",
        "prompt": "What is the capital of Australia?",
        "check": lambda r: "canberra" in r.lower(),
    },
    {
        "category": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "check": lambda r: "9" in r,
    },
    {
        "category": "reasoning",
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "check": lambda r: "5 minute" in r.lower() or "five minute" in r.lower(),
    },
    {
        "category": "code",
        "prompt": "Write a Python function that checks if a string is a palindrome. Just the function, no explanation.",
        "check": lambda r: "def " in r and ("[::-1]" in r or "reversed" in r),
    },
    {
        "category": "code",
        "prompt": "Write a Python function to find the nth Fibonacci number using iteration. Just the function.",
        "check": lambda r: "def " in r and ("fib" in r.lower()) and ("for " in r or "while " in r),
    },
    {
        "category": "creative",
        "prompt": "Write a haiku about a GPU running out of memory.",
        "check": lambda r: len(r.strip()) > 20,
    },
    {
        "category": "instruction",
        "prompt": "List exactly 5 programming languages that start with the letter P.",
        "check": lambda r: r.lower().count("python") >= 1 and r.lower().count("p") >= 5,
    },
    {
        "category": "instruction",
        "prompt": "Explain what a hash table is in exactly 2 sentences.",
        "check": lambda r: len(r.strip()) > 30,
    },
    {
        "category": "math",
        "prompt": "What is 247 * 83?",
        "check": lambda r: "20501" in r.replace(",", "").replace(" ", ""),
    },
    {
        "category": "math",
        "prompt": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "check": lambda r: "150" in r,
    },
]


def query_model(base_url: str, prompt: str, max_tokens: int = 512, disable_thinking: bool = True) -> dict:
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{base_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content", "")
            return {
                "content": content or "",
                "finish_reason": choice.get("finish_reason", "unknown"),
                "tokens": data.get("usage", {}),
                "error": None,
            }
    except Exception as e:
        return {"content": "", "finish_reason": "error", "tokens": {}, "error": str(e)}


def run_ab_test(
    model_a_url: str,
    model_b_url: str,
    label_a: str,
    label_b: str,
    output_path: str,
    verbose: bool = False,
):
    results = {
        "model_a": {"url": model_a_url, "label": label_a},
        "model_b": {"url": model_b_url, "label": label_b},
        "tests": [],
        "summary": {},
    }

    scores = {"a": 0, "b": 0}
    category_scores = {}

    for i, test in enumerate(TEST_SUITE):
        cat = test["category"]
        prompt = test["prompt"]
        check_fn = test["check"]

        if verbose:
            print(f"\n[{i+1}/{len(TEST_SUITE)}] {cat}: {prompt[:60]}...")

        resp_a = query_model(model_a_url, prompt)
        resp_b = query_model(model_b_url, prompt)

        pass_a = bool(check_fn(resp_a["content"])) if resp_a["content"] else False
        pass_b = bool(check_fn(resp_b["content"])) if resp_b["content"] else False

        if pass_a:
            scores["a"] += 1
        if pass_b:
            scores["b"] += 1

        if cat not in category_scores:
            category_scores[cat] = {"a": 0, "b": 0, "total": 0}
        category_scores[cat]["total"] += 1
        if pass_a:
            category_scores[cat]["a"] += 1
        if pass_b:
            category_scores[cat]["b"] += 1

        result = {
            "category": cat,
            "prompt": prompt,
            "model_a": {
                "content": resp_a["content"][:500],
                "pass": pass_a,
                "finish_reason": resp_a["finish_reason"],
                "error": resp_a["error"],
            },
            "model_b": {
                "content": resp_b["content"][:500],
                "pass": pass_b,
                "finish_reason": resp_b["finish_reason"],
                "error": resp_b["error"],
            },
        }
        results["tests"].append(result)

        if verbose:
            status_a = "PASS" if pass_a else "FAIL"
            status_b = "PASS" if pass_b else "FAIL"
            print(f"  {label_a}: {status_a} | {label_b}: {status_b}")
            if not pass_a:
                print(f"    {label_a} answer: {resp_a['content'][:100]}...")
            if not pass_b:
                print(f"    {label_b} answer: {resp_b['content'][:100]}...")

    results["summary"] = {
        "total_tests": len(TEST_SUITE),
        "model_a_pass": scores["a"],
        "model_b_pass": scores["b"],
        "model_a_pct": round(scores["a"] / len(TEST_SUITE) * 100, 1),
        "model_b_pct": round(scores["b"] / len(TEST_SUITE) * 100, 1),
        "by_category": category_scores,
    }

    print(f"\n{'='*60}")
    print(f"A/B Test Results")
    print(f"  {label_a}: {scores['a']}/{len(TEST_SUITE)} ({results['summary']['model_a_pct']}%)")
    print(f"  {label_b}: {scores['b']}/{len(TEST_SUITE)} ({results['summary']['model_b_pct']}%)")
    print(f"\n  By category:")
    for cat, cs in sorted(category_scores.items()):
        print(f"    {cat:15s}: {label_a}={cs['a']}/{cs['total']}  {label_b}={cs['b']}/{cs['total']}")

    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Osmosis A/B quality comparison")
    parser.add_argument("--model-a", required=True, help="Base URL for model A")
    parser.add_argument("--model-b", required=True, help="Base URL for model B")
    parser.add_argument("--label-a", default="model-a")
    parser.add_argument("--label-b", default="model-b")
    parser.add_argument("--output", default="ab_results.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run_ab_test(args.model_a, args.model_b, args.label_a, args.label_b, args.output, args.verbose)


if __name__ == "__main__":
    main()
