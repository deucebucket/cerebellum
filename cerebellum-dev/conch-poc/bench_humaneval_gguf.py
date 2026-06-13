"""
HumanEval+ benchmark runner using llama-server HTTP API (compiled + CUDA path).

Launches llama-server for each model, runs all 164 HumanEval+ problems greedy,
writes a samples.jsonl for evalplus scoring.

Usage:
    python bench_humaneval_gguf.py --model <path.gguf> --out <samples.jsonl>

Score with:
    python -m evalplus.evaluate --dataset humaneval --samples <samples.jsonl>
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from tqdm import tqdm
from evalplus.data import get_human_eval_plus

LLAMA_SERVER = "/home/deucebucket/ai-drive/llama-prismml/build/bin/llama-server"
# CUDA 12 shared libs — prismml binary was linked against these
LLAMA_LIB_DIR = "/home/deucebucket/ai-drive/llama-prismml/build/bin"
_CUDA12_DIRS = [
    "/var/home/deucebucket/.local/lib/python3.10/site-packages/nvidia/cublas/lib",
    "/var/home/deucebucket/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    "/var/home/deucebucket/.local/lib/python3.10/site-packages/nvidia/nccl/lib",
]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout: int = 120) -> bool:
    """Poll until llama-server /health returns 200 or timeout."""
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


def start_server(model_path: str) -> subprocess.Popen:
    """Start llama-server with CUDA and return the Popen handle."""
    env = os.environ.copy()
    # Prepend CUDA 12 shared lib dirs so the prismml build resolves libcublas.so.12 etc.
    ld_parts = [LLAMA_LIB_DIR] + _CUDA12_DIRS + [env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_parts if p)

    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-ngl", "99",          # all layers on GPU
        "--parallel", "1",     # single slot for deterministic HumanEval+ (per CLAUDE.md)
        "-c", "4096",          # context window
        "--log-disable",       # suppress verbose server logs
    ]
    print(f"[*] Starting llama-server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def complete(prompt: str, max_tokens: int = 512) -> str:
    """Send a completion request to the running llama-server."""
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|im_start|>"],
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
    """Extract Python code from model output, same logic as bench_humaneval.py."""
    if "```python" in text:
        return text.split("```python")[1].split("```")[0]
    elif "```" in text:
        return text.split("```")[1].split("```")[0]
    return text


def run_bench(model_path: str, out_path: str, max_new_tokens: int = 512) -> None:
    # Start server
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
            # Incremental save
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
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per problem")
    args = parser.parse_args()

    run_bench(args.model, args.out, args.max_new_tokens)


if __name__ == "__main__":
    main()
