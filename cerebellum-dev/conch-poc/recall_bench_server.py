"""
recall_bench_server.py -- HTTP-server-based recall benchmark for dead-block GGUF.

Reuses all scoring/selection logic from recall_bench_compiled.py (imported).
Generation via HTTP against a running llama-server instead of llama_cpp.Llama.

Server lifecycle per model:
  - Launch distrobox ai -- llama-server ... --port 8089 in background
  - Poll /health until ok (120s timeout)
  - Bench (POST /completion, single-client sequential)
  - Kill server, wait for port release before next model

Usage:
    python recall_bench_server.py \\
        --model-a qwen2.5-3b-brainloop.gguf \\
        --model-b cerebellum-deadblock-python.gguf \\
        [--n 200] [--seed 42] [--out recall_results_deadblock.json]
"""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import random
import torch

# Reuse all scoring/selection logic from recall_bench_compiled
from recall_bench_compiled import (
    POST_CUTOFF_MODULES,
    STATUS_FILE,
    token_overlap,
    content_overlap,
    get_symbol_module,
    is_post_cutoff,
    load_docs_get_symbol,
    compute_stats,
    format_markdown_table,
    append_status,
)

LLAMA_BIN = "/var/home/deucebucket/ai-drive/llama.cpp-stock/build/bin/llama-server"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8089
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def wait_for_port_release(timeout: int = 30) -> None:
    """Block until port 8089 is no longer in use. HARD ERROR if it stays occupied —
    a stale server here means the next model's bench silently hits the wrong brain."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1)
            s.close()
            time.sleep(1)
        except (ConnectionRefusedError, OSError):
            return
    raise RuntimeError(
        f"port {SERVER_PORT} still occupied after {timeout}s — stale llama-server "
        f"alive (distrobox wrapper kill does not reach it). Refusing to continue."
    )


def assert_loaded_model(expected_model_path: str) -> None:
    """Ask the running server which model it actually loaded; hard-fail on mismatch.
    This is the wiring assertion: no prompts are sent to an unverified server."""
    import os
    expected = os.path.basename(expected_model_path)
    found = None
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=5) as r:
            props = json.loads(r.read())
        found = props.get("model_path") or props.get("default_generation_settings", {}).get("model")
    except Exception:
        pass
    if not found:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/v1/models", timeout=5) as r:
                found = json.loads(r.read())["data"][0]["id"]
        except Exception as e:
            raise RuntimeError(f"cannot verify loaded model identity: {e}")
    if expected not in str(found):
        raise RuntimeError(
            f"WIRING ERROR: server reports model '{found}', expected '{expected}'. "
            f"A stale server from the previous model is still answering."
        )
    print(f"[+] Wiring verified: server is running {expected}")


def start_server(model_path: str) -> subprocess.Popen:
    cmd = [
        "distrobox", "enter", "ai", "--",
        LLAMA_BIN,
        "-m", model_path,
        "-ngl", "99",
        "--parallel", "4",
        "-c", "24576",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--log-disable",
    ]
    print(f"[*] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    # Terminating the distrobox wrapper does NOT kill llama-server inside the
    # container — kill the actual server by pattern, then verify the port died.
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    subprocess.run(["pkill", "-f", f"llama-server.*--port {SERVER_PORT}"],
                   capture_output=True)
    time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", f"llama-server.*--port {SERVER_PORT}"],
                   capture_output=True)
    wait_for_port_release()


def complete_http(prompt: str, max_tokens: int = 60) -> str:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read().decode())
    return body.get("content", "")


def run_model_http(docs: list, indices: list, max_tokens: int,
                   overlap_threshold: float, label: str) -> list:
    results = []
    total = len(indices)
    hits = 0

    for rank, idx in enumerate(indices):
        symbol, doc = load_docs_get_symbol(docs, idx)
        prompt = f"Question: How do I use {symbol} in Python?\nAnswer: "

        completion = complete_http(prompt, max_tokens)
        overlap = content_overlap(doc, completion)
        is_hit = overlap >= overlap_threshold
        if is_hit:
            hits += 1

        results.append({
            "idx": idx,
            "symbol": symbol,
            "doc": doc,
            "completion": completion.strip(),
            "overlap": round(overlap, 4),
            "hit": is_hit,
            "post_cutoff": is_post_cutoff(symbol),
        })

        if (rank + 1) % 20 == 0 or (rank + 1) == total:
            print(f"  [{label}] {rank+1}/{total}  running recall={hits/(rank+1):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HTTP-server recall benchmark: model-A vs model-B."
    )
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="recall_results_deadblock.json")
    parser.add_argument("--docs", default="python_13k_docs.pt")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load docs and sample
    print(f"[*] Loading docs from {args.docs}...")
    docs = torch.load(args.docs, map_location="cpu", weights_only=False)
    print(f"[*] Loaded {len(docs)} docs.")

    random.seed(args.seed)
    indices = random.sample(range(len(docs)), min(args.n, len(docs)))
    print(f"[*] Sampled {len(indices)} symbols (seed={args.seed}).")

    pc_sample = [i for i in indices
                 if is_post_cutoff(load_docs_get_symbol(docs, i)[0])]
    print(f"[*] Post-cutoff symbols in sample: {len(pc_sample)}")

    # ---- Model A ----
    print(f"\n[*] Launching server for model A: {args.model_a}")
    proc_a = start_server(args.model_a)
    try:
        if not wait_for_server(120):
            print("[!] Server A failed to start. Aborting.")
            stop_server(proc_a)
            sys.exit(1)
        assert_loaded_model(args.model_a)
        print("[*] Server A ready. Running bench...")
        results_a = run_model_http(docs, indices, args.max_tokens,
                                   args.overlap_threshold, label="A")
    finally:
        print("[*] Stopping server A...")
        stop_server(proc_a)

    # ---- Model B ----
    print(f"\n[*] Launching server for model B: {args.model_b}")
    proc_b = start_server(args.model_b)
    try:
        if not wait_for_server(120):
            print("[!] Server B failed to start. Aborting.")
            stop_server(proc_b)
            sys.exit(1)
        assert_loaded_model(args.model_b)
        print("[*] Server B ready. Running bench...")
        results_b = run_model_http(docs, indices, args.max_tokens,
                                   args.overlap_threshold, label="B")
    finally:
        print("[*] Stopping server B...")
        stop_server(proc_b)

    # ---- Wiring self-check: identical A/B under greedy decoding = miswired ----
    n_same = sum(1 for a, b in zip(results_a, results_b)
                 if a["completion"] == b["completion"])
    print(f"[*] Identical-completion fraction A vs B: {n_same}/{len(results_a)}")
    if results_a and n_same == len(results_a) and args.model_a != args.model_b:
        raise RuntimeError(
            "WIRING ERROR: 100% identical completions between supposedly different "
            "models — both passes hit the same server. Results discarded."
        )

    # ---- Stats ----
    stats_a = compute_stats(results_a, baseline_results=results_a)
    stats_b = compute_stats(results_b, baseline_results=results_a)

    table = format_markdown_table(stats_a, stats_b, args.model_a, args.model_b)
    print("\n" + table)

    output = {
        "meta": {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "n": args.n,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "overlap_threshold": args.overlap_threshold,
            "timestamp": datetime.datetime.now().isoformat(),
            "server_binary": LLAMA_BIN,
        },
        "stats_a": stats_a,
        "stats_b": stats_b,
        "results_a": results_a,
        "results_b": results_b,
    }
    with open(args.out, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[+] Results saved to {args.out}")

    append_status(table, args.model_a, args.model_b, args.n, args.seed)


if __name__ == "__main__":
    main()
