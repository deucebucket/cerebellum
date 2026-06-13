"""
bench_humaneval_server.py -- HumanEval+ via stock llama-server HTTP (distrobox/CUDA).

Prompts exactly as bench_humaneval.py:
  "<|im_start|>user\nSolve this Python coding problem:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
n_predict=512, temperature=0, stop=["<|im_end|>"], ```python extraction.
SINGLE worker, sequential (HumanEval determinism rule).

Server lifecycle: distrobox enter ai -- llama-server ... --port 8089, kill+wait between models.

Usage:
    python bench_humaneval_server.py --model <path.gguf> --out <samples.jsonl>

Score with:
    python -m evalplus.evaluate --dataset humaneval --samples <samples.jsonl>
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from tqdm import tqdm
from evalplus.data import get_human_eval_plus

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
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1)
            s.close()
            time.sleep(1)
        except (ConnectionRefusedError, OSError):
            return


def start_server(model_path: str) -> subprocess.Popen:
    cmd = [
        "distrobox", "enter", "ai", "--",
        LLAMA_BIN,
        "-m", model_path,
        "-ngl", "99",
        "--parallel", "1",      # single slot for HumanEval determinism
        "-c", "24576",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--log-disable",
    ]
    print(f"[*] Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    wait_for_port_release()


def complete(prompt: str, max_tokens: int = 512) -> str:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stop": ["<|im_end|>"],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read().decode())
    return body.get("content", "")


def extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    elif "```" in text:
        return text.split("```")[1].split("```")[0]
    return text


def run_bench(model_path: str, out_path: str, max_new_tokens: int = 512) -> None:
    proc = start_server(model_path)
    try:
        print("[*] Waiting for server to be ready...")
        if not wait_for_server(timeout=120):
            print("[!] Server did not start in time.")
            stop_server(proc)
            sys.exit(1)
        print("[*] Server ready.")

        dataset = get_human_eval_plus()
        samples = []

        for task_id, problem in tqdm(dataset.items(), desc="HumanEval+"):
            prompt_text = (
                f"<|im_start|>user\nSolve this Python coding problem:\n"
                f"{problem['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
            raw = complete(prompt_text, max_tokens=max_new_tokens)
            solution = extract_code(raw)
            samples.append({"task_id": task_id, "completion": solution})
            # Incremental write — never buffer until end
            with open(out_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

        print(f"[+] {len(samples)} samples written to {out_path}")

    finally:
        print("[*] Stopping server...")
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--out", required=True, help="Output JSONL file for evalplus")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    run_bench(args.model, args.out, args.max_new_tokens)


if __name__ == "__main__":
    main()
