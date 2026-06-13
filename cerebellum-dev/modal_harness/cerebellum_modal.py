"""Cerebellum Modal harness — run finished-recipe GGUF quantization (CPU) and
small-model PPL (T4) on Modal's free monthly credits instead of the local PC.

BUDGET DISCIPLINE
-----------------
This account runs on Modal's free $30/month credits. Every function here has an
explicit timeout and uses the smallest hardware that works:

  - quantize: 8 CPU cores, 16 GiB RAM, timeout 3600 s.
    Cost at list price (2026-06): CPU $0.0000131/core/s + RAM $0.00000222/GiB/s
    => ~$0.13/hr fully busy. A 3 GB GGUF re-quant takes ~3-5 min (~$0.01-0.02).
  - ppl: 1x T4 ($0.000164/s => $0.59/hr), 4 cores, timeout 1800 s.
    A small-model PPL on a few chunks takes ~2-4 min (~$0.02-0.04).

IMAGE STRATEGY (documented choice)
----------------------------------
quantize (CPU): debian_slim + the official llama.cpp Linux release tarball
  (ggml-org/llama.cpp release asset `llama-<TAG>-bin-ubuntu-x64.tar.gz`).
  Chosen over compiling from source because: 15 MB download vs ~10 min compile,
  deterministic (pinned tag), and the binaries were verified to need only
  glibc >= 2.34 / GLIBCXX <= 3.4.29 / libgomp1 / libssl3 — all satisfied by
  debian bookworm. The tarball is flat: binaries + .so files in one dir, so
  LD_LIBRARY_PATH must point at it.

ppl (GPU): the official CUDA docker image ghcr.io/ggml-org/llama.cpp:full-cuda-<TAG>
  via Image.from_registry (no Linux CUDA binary tarball is published upstream).
  add_python="3.11" because Modal needs a Python it controls; .entrypoint([])
  clears the image's tools.sh ENTRYPOINT which would otherwise confuse Modal.

PREREQUISITES (one-time)
------------------------
  modal secret create hf-token HF_TOKEN="$(hf auth token)"

USAGE (one-liners)
------------------
  # Quantize a source GGUF on Modal CPU and upload the result to HF:
  modal run cerebellum_modal.py::quantize \
      --source-repo deucebucket/SomeModel-GGUF \
      --source-filename model-f16.gguf \
      --override-file /path/to/tensor_types.txt \
      --base-type Q3_K_M \
      --output-filename model-cerebellum-Q3_K_M.gguf \
      --target-repo deucebucket/SomeModel-Cerebellum-GGUF

  # add --allow-requantize when the source is already quantized (lossy; mechanics/tests only)
  # add --imatrix-repo R --imatrix-filename F to pull an imatrix .dat from HF
  # add --no-private to create the target repo public (default: private)

  # PPL of a GGUF on a T4 against a local corpus file:
  modal run cerebellum_modal.py::ppl \
      --repo deucebucket/SomeModel-GGUF \
      --filename model-Q3_K_M.gguf \
      --corpus-file /var/home/deucebucket/ai-drive/cerebellum/wikitext-test.txt \
      --ctx-size 2048 --chunks 8

NOTES / GOTCHAS
---------------
- First invocation of each function builds its image. The CPU image builds in
  ~1-2 min; the CUDA image pull is several GB and takes ~5-8 min the first
  time. Both are cached afterwards.
- Override-file content is sanitized before use: comment/blank lines are
  stripped because llama-quantize parses `#` lines as tensor names and silently
  dumps help text with exit 0 (known upstream bug we hit before). If nothing
  remains, --tensor-type-file is omitted entirely.
- Modal has no CLI surface for billing/spending limits. Check
  https://modal.com/settings/<workspace>/usage in the dashboard. On the free
  plan with no payment method attached, compute simply stops when credits run
  out — there is no overage risk.
- For big F16 sources (>60 GB), raise `memory`/`ephemeral_disk` on the
  quantize function below (and expect ~2x source size in scratch disk).
"""

import os
import re
import subprocess
import sys
import time

import modal

# Pinned llama.cpp build (verified 2026-06-12: linux x64 CPU tarball present,
# full-cuda docker tag present, --tensor-type-file supported).
LLAMA_CPP_TAG = "b9603"
LLAMA_DIR = f"/opt/llama-{LLAMA_CPP_TAG}"
TARBALL_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_TAG}/llama-{LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz"
)
CUDA_IMAGE_REF = f"ghcr.io/ggml-org/llama.cpp:full-cuda-{LLAMA_CPP_TAG}"

app = modal.App("cerebellum-quant")

hf_secret = modal.Secret.from_name("hf-token")

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "libgomp1", "libssl3")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "LD_LIBRARY_PATH": LLAMA_DIR})
    .run_commands(
        f"curl -fsSL -o /tmp/llama.tar.gz {TARBALL_URL}",
        "tar xzf /tmp/llama.tar.gz -C /opt && rm /tmp/llama.tar.gz",
        # build-time sanity check: --help exits 1 by design, so grep for usage text
        f"{LLAMA_DIR}/llama-quantize --help 2>&1 | grep -q usage",
    )
    .pip_install("huggingface_hub[hf_transfer]>=0.30")
)

gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])  # clear the image's tools.sh ENTRYPOINT
    .pip_install("huggingface_hub[hf_transfer]>=0.30")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


# ---------------------------------------------------------------------------
# Helpers (run inside containers)
# ---------------------------------------------------------------------------

def _sanitize_override(text: str) -> str:
    """Drop comment/blank lines — llama-quantize chokes on `#` lines (see module docstring)."""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln and not ln.startswith("#"))


def _hf_download(repo: str, filename: str, dest_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo, filename=filename, local_dir=dest_dir,
        token=os.environ["HF_TOKEN"],
    )


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, **kw)


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    cpu=8.0,
    memory=16384,  # MiB
    timeout=3600,
    secrets=[hf_secret],
)
def quantize_remote(
    source_repo: str,
    source_filename: str,
    override_text: str,
    base_type: str,
    output_filename: str,
    target_repo: str,
    private: bool = True,
    allow_requantize: bool = False,
    imatrix_repo: str = "",
    imatrix_filename: str = "",
    nthreads: int = 8,
) -> dict:
    """Download source GGUF from HF, run llama-quantize with a tensor-type
    override file, upload the result to `target_repo`. Returns metadata dict."""
    from huggingface_hub import HfApi

    t0 = time.time()
    work = "/work"
    os.makedirs(work, exist_ok=True)

    src = _hf_download(source_repo, source_filename, work)
    t_dl = time.time() - t0
    print(f"downloaded {src} ({os.path.getsize(src)/1e9:.2f} GB) in {t_dl:.0f}s")

    imatrix_path = None
    if imatrix_repo and imatrix_filename:
        imatrix_path = _hf_download(imatrix_repo, imatrix_filename, work)

    out = os.path.join(work, output_filename)
    cmd = [f"{LLAMA_DIR}/llama-quantize"]
    if allow_requantize:
        cmd.append("--allow-requantize")
    if imatrix_path:
        cmd += ["--imatrix", imatrix_path]
    clean = _sanitize_override(override_text)
    if clean:
        ovr = os.path.join(work, "tensor_types.txt")
        with open(ovr, "w") as f:
            f.write(clean + "\n")
        print(f"override file ({len(clean.splitlines())} lines):\n{clean}")
        cmd += ["--tensor-type-file", ovr]
    cmd += [src, out, base_type, str(nthreads)]

    t1 = time.time()
    proc = _run(cmd, capture_output=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    print(tail)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            f"llama-quantize failed (rc={proc.returncode}, output exists="
            f"{os.path.exists(out)}). Tail:\n{tail}"
        )
    t_quant = time.time() - t1
    out_size = os.path.getsize(out)
    print(f"quantized -> {out} ({out_size/1e9:.2f} GB) in {t_quant:.0f}s")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(target_repo, private=private, exist_ok=True, repo_type="model")
    t2 = time.time()
    api.upload_file(
        path_or_fileobj=out, path_in_repo=output_filename,
        repo_id=target_repo, repo_type="model",
    )
    t_up = time.time() - t2

    return {
        "target_repo": target_repo,
        "output_filename": output_filename,
        "output_bytes": out_size,
        "source_bytes": os.path.getsize(src),
        "download_s": round(t_dl, 1),
        "quantize_s": round(t_quant, 1),
        "upload_s": round(t_up, 1),
        "total_s": round(time.time() - t0, 1),
        "llama_cpp_tag": LLAMA_CPP_TAG,
    }


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=1800,
    secrets=[hf_secret],
)
def ppl_remote(
    repo: str,
    filename: str,
    corpus_text: str,
    ctx_size: int = 2048,
    chunks: int = -1,
    ngl: int = 99,
) -> dict:
    """Download a GGUF from HF, run llama-perplexity on `corpus_text` (CUDA),
    parse `Final estimate: PPL = X`, return the float plus timings."""
    t0 = time.time()
    work = "/work"
    os.makedirs(work, exist_ok=True)

    model = _hf_download(repo, filename, work)
    t_dl = time.time() - t0

    corpus = os.path.join(work, "corpus.txt")
    with open(corpus, "w") as f:
        f.write(corpus_text)

    # locate llama-perplexity inside the official image
    binary = None
    for cand in ("/app/llama-perplexity", "/llama-perplexity",
                 "/usr/local/bin/llama-perplexity"):
        if os.path.exists(cand):
            binary = cand
            break
    if binary is None:
        found = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "llama-perplexity",
             "-type", "f"], capture_output=True, text=True,
        ).stdout.split()
        if not found:
            raise RuntimeError("llama-perplexity not found in CUDA image")
        binary = found[0]

    cmd = [
        binary, "--model", model, "--file", corpus,
        "-ngl", str(ngl), "--ctx-size", str(ctx_size), "--chunks", str(chunks),
    ]
    t1 = time.time()
    proc = _run(cmd, capture_output=True)
    output = proc.stdout + proc.stderr
    print("\n".join(output.splitlines()[-20:]))
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
    if not m:
        raise RuntimeError(
            f"PPL parse failed (rc={proc.returncode}). Tail:\n"
            + "\n".join(output.splitlines()[-30:])
        )
    return {
        "ppl": float(m.group(1)),
        "download_s": round(t_dl, 1),
        "ppl_s": round(time.time() - t1, 1),
        "total_s": round(time.time() - t0, 1),
        "ctx_size": ctx_size,
        "chunks": chunks,
        "llama_cpp_tag": LLAMA_CPP_TAG,
    }


# ---------------------------------------------------------------------------
# Local entrypoints (the driver CLI)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def quantize(
    source_repo: str,
    source_filename: str,
    override_file: str,
    base_type: str,
    output_filename: str,
    target_repo: str,
    private: bool = True,
    allow_requantize: bool = False,
    imatrix_repo: str = "",
    imatrix_filename: str = "",
):
    """Quantize on Modal CPU. --override-file is a local tensor_types.txt path."""
    with open(override_file) as f:
        override_text = f.read()
    result = quantize_remote.remote(
        source_repo=source_repo,
        source_filename=source_filename,
        override_text=override_text,
        base_type=base_type,
        output_filename=output_filename,
        target_repo=target_repo,
        private=private,
        allow_requantize=allow_requantize,
        imatrix_repo=imatrix_repo,
        imatrix_filename=imatrix_filename,
    )
    print("\n=== quantize result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    cpu_cost = result["total_s"] * 8 * 0.0000131
    mem_cost = result["total_s"] * 16 * 0.00000222
    print(f"  est_cost_usd: {cpu_cost + mem_cost:.4f} (8 cores + 16 GiB, list price)")


@app.local_entrypoint()
def ppl(
    repo: str,
    filename: str,
    corpus_file: str,
    ctx_size: int = 2048,
    chunks: int = -1,
    ngl: int = 99,
):
    """Run PPL on a T4. --corpus-file is a local text file path."""
    with open(corpus_file) as f:
        corpus_text = f.read()
    result = ppl_remote.remote(
        repo=repo, filename=filename, corpus_text=corpus_text,
        ctx_size=ctx_size, chunks=chunks, ngl=ngl,
    )
    print("\n=== ppl result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    gpu_cost = result["total_s"] * 0.000164
    print(f"  est_cost_usd: {gpu_cost:.4f} (T4 list price, excl. small CPU/RAM adder)")


if __name__ == "__main__":
    sys.exit(
        "Run via Modal, e.g.:\n"
        "  modal run cerebellum_modal.py::quantize --source-repo ... "
        "--source-filename ... --override-file ... --base-type Q3_K_M "
        "--output-filename ... --target-repo ...\n"
        "  modal run cerebellum_modal.py::ppl --repo ... --filename ... "
        "--corpus-file ...\n"
        "See module docstring for details."
    )
