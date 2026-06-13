"""Cerebellum North-Mini-Code-1.0 campaign — runs ENTIRELY on Modal.

THE METHOD (OG bench-gated Cerebellum formula — nothing else):
  group ablation (crush one tensor-type GROUP to Q2_K, measure 4-domain PPL
  delta vs the uniform Q4_K_M base) -> protect/demote verdicts -> STOP.
  Allocation builds and benchmarks happen LOCALLY later (code model —
  BigCodeBench gates need the local harness).  NO hillstep.

MODEL FACTS (MEASURED 2026-06-12 against the local BF16 conversion at
/var/home/deucebucket/games/cerebellum-north-mini-code/north-mini-code-bf16.gguf,
read with PR #24260 gguf-py):
  - GGUF arch: `cohere2moe` (Cohere2MoeForCausalLM, llama.cpp PR #24260 —
    OPEN, not merged; pinned commit d9320477de5549e53a9452296f468d32a1d81d26).
  - block_count=49 (blk.0..blk.48).  blk.0 is the single leading dense FFN
    layer (leading_dense_block_count=1): ffn_{gate,up,down}.weight only there.
  - blk.1..blk.48 MoE: ffn_{gate,up,down}_exps (128 experts, 8 used,
    sigmoid gating func=2, expert_weights_norm=False), ffn_gate_inp router.
    NO shared expert (hypothesis confirmed).
  - attention: plain attn_q/attn_k/attn_v/attn_output (49 each). One
    attn_norm per layer (Cohere parallel block), no ffn_norm.
  - vocab 262144; **output.weight ABSENT -> embeddings TIED**.  The
    hypothesized `output_head` group does not exist; token_embd ablation
    hits both the embedding AND the LM head.
  - NO nextn/MTP tensors, no blk.49+.  442 tensors total.
  - imatrix: unsloth/North-Mini-Code-1.0-GGUF `imatrix_unsloth.gguf_file`
    (122 MB, GGUF-format imatrix, 391 covered tensors — verified locally to
    read with PR gguf-py and to match PR tensor naming).  PRE-MERGE
    provenance caveat is logged in cerebellum-north-mini-code/OPS_LOG.md.

IMAGE STRATEGY (the critical difference from Flash):
  cohere2moe is NOT in the pinned b9603 release image, so we build a custom
  image: nvidia/cuda:12.6.3-devel-ubuntu22.04 + add_python 3.11, clone
  llama.cpp, fetch pull/24260/head, checkout the pinned PR commit (fallback:
  PR head), cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;89"
  (T4=sm_75, L4/L40S=sm_89) -DGGML_NATIVE=OFF (portable: build host CPU !=
  runtime CPU), build llama-quantize/llama-perplexity/llama-cli, keep
  convert_hf_to_gguf.py + gguf-py importable, add CPU torch + transformers
  for conversion.  DECISION: this ONE image is reused for CPU containers
  (convert / quantize / verify) instead of building a CPU-only variant —
  containers are billed by CPU/RAM, not image size, so reuse costs the same,
  saves a second 10-20 min image build, and guarantees a single binary
  provenance for every artifact.  Build ~10-20 min, cached after.

PIPELINE (artifacts on the `cerebellum-north` Volume; driver runs ON MODAL
so it survives anything local):
  phase 0: download CohereLabs/North-Mini-Code-1.0 safetensors in-datacenter
           (61 GB), convert to BF16 onto the Volume with the PR converter;
           fetch the unsloth imatrix; VERIFY measured tensor inventory +
           imatrix coverage — hard stop on any surprise (nextn present,
           output.weight present, unmatched/missing tensors).
  phase 1: uniform Q8_0 / Q4_K_M / Q3_K_M / Q2_K baselines + 4-domain
           calibration PPL + wiki.test.raw.  Q8_0 (~30 GB) deleted after PPL.
  phase 2: for each of 8 measured groups: Q4_K_M base + group->Q2_K
           candidate (imatrix), 4-domain PPL, DELETE candidate.  CPU builds
           pipelined 2-in-flight ahead of the GPU; ONE GPU container at a
           time; the stage-4/flash GPU gate (below) is checked before EVERY
           GPU dispatch.
  phase 3: ablation_summary.md with PROTECT/TOLERANT/DEMOTABLE verdicts,
           STOP.  No allocation builds, no benches (local gates).

GPU GATE (shared-account discipline): a Flash stage-4 bench job may be
running on this workspace (app description contains 'stage4' or 'flash').
Before every GPU dispatch the driver polls `modal app list --json` (via the
modal-token secret) and waits while any such app is running/ephemeral.
CPU work overlaps freely.  If polling itself fails 3x in a row we proceed
with a loud log (the local watchdog still enforces the $4/hr velocity kill
and the $28.50 doors-close).

WRITE-RACE FIX (Flash lesson, cost real money): Flash's ppl containers
sometimes saw "model not on volume" right after the quantize container's
vol.commit() and escalated to L40S.  Here the reader does vol.reload() +
exponential backoff (6 attempts) before declaring the file missing; the
L40S fallback remains only for genuine GPU failures (OOM etc.).

BUDGET DISCIPLINE:
  retries=0 on every function.  Cost ledger on the Volume; the driver
  refuses to start a phase that would project past budget (default $10;
  phase estimates P0 $1.00, P1 $2.50, P2 $5.00) and stops CLEANLY at the
  last completed group — partial maps are still data.  Concurrency is
  capped at 2 CPU builders (cpu=12 each) + 1 GPU so combined velocity with
  a concurrent stage-4 L40S stays under the watchdog's $4/hr kill.
  Gate locally before launch:  python3 modal_credits.py --gate 8

LAUNCH (detached — survives local death):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  modal volume create cerebellum-north   # once
  modal volume put cerebellum-north <corpora...> corpora/<name>   # once
  modal secret create modal-token MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
  nohup setsid modal run --detach cerebellum_north_campaign.py::launch \
      > ../../cerebellum-north-mini-code/logs/modal_launch.log 2>&1 &

MONITOR:
  modal app list                                   # cerebellum-north-campaign
  modal app logs cerebellum-north-campaign
  modal volume ls cerebellum-north results
  python3 modal_credits.py
"""

import json
import os
import re
import subprocess
import time

import modal

app = modal.App("cerebellum-north-campaign")

hf_secret = modal.Secret.from_name("hf-token")
modal_token_secret = modal.Secret.from_name("modal-token")
vol = modal.Volume.from_name("cerebellum-north", create_if_missing=True)
V = "/vol"

HF_MODEL = "CohereLabs/North-Mini-Code-1.0"
IMATRIX_REPO = "unsloth/North-Mini-Code-1.0-GGUF"
IMATRIX_FILE = "imatrix_unsloth.gguf_file"

LLAMA_PR = "24260"
LLAMA_PR_COMMIT = "d9320477de5549e53a9452296f468d32a1d81d26"
BUILD_DIR = "/opt/llama.cpp/build/bin"          # CUDA build: llama-perplexity (GPU containers)
CPU_BUILD_DIR = "/opt/llama.cpp/build-cpu/bin"  # CPU build: llama-quantize/llama-cli (no CUDA link)

BF16 = f"{V}/north-mini-code-bf16.gguf"
IMATRIX = f"{V}/imatrix/{IMATRIX_FILE}"
CORPORA_DIR = f"{V}/corpora"
RESULTS_DIR = f"{V}/results"
DOMAINS = ["wiki", "code", "math", "dialogue"]
WIKITEST = "wiki.test.raw"

ABLATE_TYPE = "Q2_K"
BASE_TYPE = "Q4_K_M"

# Tensor groups — MEASURED against the local BF16 conversion (counts exact).
# Reconciled from cerebellum-north-mini-code/ablation_groups.txt: the
# hypothesized output_head group is DROPPED (output.weight absent — tied).
GROUPS = [
    ("routed_exps_gate_up", r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$", 96),
    ("routed_exps_down", r"^blk\.\d+\.ffn_down_exps\.weight$", 48),
    ("attn_q", r"^blk\.\d+\.attn_q\.weight$", 49),
    ("attn_k", r"^blk\.\d+\.attn_k\.weight$", 49),
    ("attn_v", r"^blk\.\d+\.attn_v\.weight$", 49),
    ("attn_output", r"^blk\.\d+\.attn_output\.weight$", 49),
    ("dense_ffn_l0", r"^blk\.0\.ffn_(gate|up|down)\.weight$", 3),
    ("token_embd", r"^token_embd\.weight$", 1),  # TIED: also the LM head
]

# Tensors that exist but are never ablated (norms / router).  Together with
# GROUPS these must account for EVERY tensor in the GGUF (verify hard-stops
# otherwise).
EXCLUDED = [
    (r"^blk\.\d+\.attn_norm\.weight$", 49),
    (r"^blk\.\d+\.ffn_gate_inp\.weight$", 48),
    (r"^output_norm\.weight$", 1),
]
EXPECTED_TOTAL = 442

# ---------------------------------------------------------------------------
# Cost model (list prices, $/sec)
# ---------------------------------------------------------------------------
CPU_S = 0.0000131  # per core
MEM_S = 0.00000222  # per GiB
GPU_S = {"T4": 0.000164, "L4": 0.000222, "L40S": 0.000542}

RATES = {
    "convert": 8 * CPU_S + 64 * MEM_S,
    "verify": 4 * CPU_S + 16 * MEM_S,
    "quantize": 12 * CPU_S + 32 * MEM_S,
    "ppl_t4": GPU_S["T4"] + 4 * CPU_S + 16 * MEM_S,
    "ppl_l4": GPU_S["L4"] + 4 * CPU_S + 24 * MEM_S,
    "ppl_l40s": GPU_S["L40S"] + 4 * CPU_S + 48 * MEM_S,
    "driver": 1 * CPU_S + 2 * MEM_S,
}
PHASE_EST = {"phase0": 1.00, "phase1": 2.50, "phase2": 5.00}

# Concurrent PPL measurements in phase 2. Same total GPU-seconds, ~half the
# wall clock. L40S-class PPLs always run solo regardless (velocity cap).
PPL_LANES = 2

# ---------------------------------------------------------------------------
# Images — ONE custom CUDA image for all heavy work (see IMAGE STRATEGY)
# ---------------------------------------------------------------------------

llama_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "curl", "ca-certificates")
    # Build split into separate layers so completed steps CACHE across
    # retries, with -j capped: the 07:02 launch died at ~31% with runner
    # exit -1 — `-j $(nproc)` on the shared builder spawns more nvcc procs
    # than its RAM supports. Arch 89 only (L4/L40S): dropping sm75 halves
    # the CUDA template instances; pick_ppl no longer routes to T4.
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        # pinned PR commit; fall back to PR head if upstream force-pushed
        f"cd /opt/llama.cpp && git fetch origin pull/{LLAMA_PR}/head:pr && "
        f"(git checkout {LLAMA_PR_COMMIT} || git checkout pr) && "
        "git rev-parse HEAD | tee /opt/LLAMA_SHA",
    )
    .run_commands(
        # CPU-only tool build: llama-quantize/llama-cli run on GPU-less
        # containers, so they must not link CUDA at all (the local dual-build
        # convention, build-cpu vs build). A CUDA-linked quantize would fail
        # to exec on CPU containers (no libcuda.so.1 there).
        "cd /opt/llama.cpp && cmake -B build-cpu "
        "-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF "
        "-DCMAKE_BUILD_TYPE=Release && "
        "cmake --build build-cpu -j 8 --target llama-quantize llama-cli",
        f"{CPU_BUILD_DIR}/llama-quantize --help 2>&1 | grep -q tensor-type-file",
    )
    .run_commands(
        # CUDA build for llama-perplexity only. The builder has no GPU
        # driver, so the CUDA driver API resolves against the toolkit stubs
        # (CXX_STANDARD_LIBRARIES lands at the END of the link line — exe
        # linker flags would precede the objects and be skipped by ld).
        # Runtime GPU containers inject the real libcuda.so.1.
        "cd /opt/llama.cpp && cmake -B build "
        "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='89' "
        "-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_STANDARD_LIBRARIES='-L/usr/local/cuda/lib64/stubs -lcuda'",
    )
    .run_commands(
        # the CUDA bulk, alone in its own cacheable layer
        "cd /opt/llama.cpp && cmake --build build -j 8 --target ggml",
    )
    .run_commands(
        "cd /opt/llama.cpp && cmake --build build -j 8 --target llama-perplexity",
    )
    .run_commands(
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "transformers>=4.46",
        "sentencepiece",
        "safetensors",
        "numpy",
        "huggingface_hub[hf_transfer,hf_xet]>=0.30",
    )
    .run_commands(
        # VOCAB FIX (root cause of the 2026-06-12 15:17 PPL wedge): at PR head
        # d9320477 the converter writes tokenizer.ggml.pre="cohere2moe"
        # (conversion/base.py chkhsh 52df12b4...) but the PR never touches
        # src/llama-vocab.cpp, so NO runtime accepts that string — every GGUF
        # converted at head fails to load ("unknown pre-tokenizer type:
        # 'cohere2moe'"). llama-quantize worked only because quantize never
        # loads the vocab. PR commit 0eada9a ("Removed cohere2-moe as a
        # tokenizer type and kept as tiny_aya") shows the intended mapping:
        # cohere2moe pre-tokenizes as TINY_AYA. One-line patch + relink —
        # ggml CUDA bulk stays cached, only llama-vocab.cpp recompiles.
        'cd /opt/llama.cpp && '
        'sed -i \'s@tokenizer_pre == "tiny_aya")@tokenizer_pre == "tiny_aya" || tokenizer_pre == "cohere2moe")@\' '
        'src/llama-vocab.cpp && grep -q \'tokenizer_pre == "cohere2moe"\' src/llama-vocab.cpp',
        "cd /opt/llama.cpp && cmake --build build -j 8 --target llama-perplexity",
        # build-cpu llama-cli gets the same fix so the CPU vocab smoke-test
        # (and any future CPU-side load) sees the patched vocab too.
        "cd /opt/llama.cpp && cmake --build build-cpu -j 8 --target llama-cli",
    )
    .env({"PYTHONPATH": "/opt/llama.cpp/gguf-py", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("cerebellum_north_campaign")
)

driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_north_campaign"
)


def _llama_sha() -> str:
    try:
        with open("/opt/LLAMA_SHA") as f:
            return f.read().strip()
    except Exception:
        return "unknown"


def _sanitize_override(text: str) -> str:
    """Strip comment/blank lines — llama-quantize parses `#` lines as tensor
    names and silently dumps help text with exit 0 (known upstream bug)."""
    return "\n".join(
        ln for ln in (text or "").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )


def _wait_on_volume(path: str, attempts: int = 6) -> bool:
    """Reader-side fix for the Volume write race (Flash lesson): reload +
    exponential backoff before declaring a file missing."""
    for i in range(attempts):
        vol.reload()
        if os.path.exists(path):
            return True
        wait = 10 * (2 ** min(i, 3))
        print(f"  not visible yet ({path}), backoff {wait}s [{i+1}/{attempts}]",
              flush=True)
        time.sleep(wait)
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=llama_image,
    cpu=8.0,
    memory=65536,
    timeout=7200,
    retries=0,
    secrets=[hf_secret],
    volumes={V: vol},
)
def convert_and_fetch() -> dict:
    """Download North-Mini-Code-1.0 safetensors inside Modal, convert to BF16
    GGUF onto the Volume with the PR #24260 converter, fetch the unsloth
    imatrix.  Idempotent."""
    from huggingface_hub import hf_hub_download, snapshot_download

    t0 = time.time()
    out = {"llama_sha": _llama_sha()}

    os.makedirs(f"{V}/imatrix", exist_ok=True)
    if not os.path.exists(IMATRIX):
        print(f"fetching imatrix {IMATRIX_REPO}/{IMATRIX_FILE}", flush=True)
        p = hf_hub_download(
            repo_id=IMATRIX_REPO, filename=IMATRIX_FILE,
            local_dir=f"{V}/imatrix", token=os.environ["HF_TOKEN"],
        )
        out["imatrix_bytes"] = os.path.getsize(p)
        vol.commit()

    if os.path.exists(BF16) and os.path.getsize(BF16) > 55e9:
        print(f"BF16 already on volume ({os.path.getsize(BF16)/1e9:.1f} GB), skipping convert")
        out["bf16_bytes"] = os.path.getsize(BF16)
        return out

    src = "/work/src"
    os.makedirs(src, exist_ok=True)
    print(f"downloading {HF_MODEL} ...", flush=True)
    snapshot_download(
        repo_id=HF_MODEL, local_dir=src, token=os.environ["HF_TOKEN"],
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.jinja", "*.model"],
        ignore_patterns=["*.pt"],
    )
    t_dl = time.time() - t0
    print(f"download done in {t_dl:.0f}s", flush=True)

    tmp_out = BF16 + ".part"
    cmd = [
        "python", "/opt/llama.cpp/convert_hf_to_gguf.py", src,
        "--outfile", tmp_out, "--outtype", "bf16",
    ]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
    print(tail, flush=True)
    if proc.returncode != 0 or not os.path.exists(tmp_out):
        raise RuntimeError(f"convert failed rc={proc.returncode}:\n{tail}")
    os.replace(tmp_out, BF16)
    vol.commit()
    out["bf16_bytes"] = os.path.getsize(BF16)
    out["download_s"] = round(t_dl, 1)
    out["total_s"] = round(time.time() - t0, 1)
    print(f"BF16 on volume: {out['bf16_bytes']/1e9:.1f} GB, total {out['total_s']:.0f}s")
    return out


@app.function(
    image=llama_image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    retries=0,
    volumes={V: vol},
)
def verify_assets() -> dict:
    """Reconcile MEASURED GGUF tensor names against GROUPS/EXCLUDED and check
    imatrix coverage.  Returns {'ok': bool, 'surprises': [...], ...} — the
    driver hard-stops on surprises (nextn tensors, untied output.weight,
    unmatched/missing tensors, imatrix gaps)."""
    from gguf import GGUFReader

    vol.reload()
    if not _wait_on_volume(BF16):
        raise RuntimeError(f"BF16 not on volume: {BF16}")
    if not _wait_on_volume(IMATRIX):
        raise RuntimeError(f"imatrix not on volume: {IMATRIX}")

    r = GGUFReader(BF16)
    names = [t.name for t in r.tensors]
    meta = {}
    for f in r.fields.values():
        if any(s in f.name for s in
               ("architecture", "block_count", "expert", "vocab_size", "leading")):
            try:
                meta[f.name] = f.contents()
            except Exception:
                pass

    surprises = []
    if [n for n in names if "nextn" in n]:
        surprises.append(f"NEXTN TENSORS PRESENT: {[n for n in names if 'nextn' in n][:5]}")
    if "output.weight" in names:
        surprises.append("output.weight PRESENT — embeddings UNTIED (hypothesis was tied); "
                         "output_head group missing from GROUPS")
    if len(names) != EXPECTED_TOTAL:
        surprises.append(f"tensor count {len(names)} != expected {EXPECTED_TOTAL}")

    patterns = [(n, rx, cnt) for (n, rx, cnt) in GROUPS] + [
        (f"excluded_{i}", rx, cnt) for i, (rx, cnt) in enumerate(EXCLUDED)
    ]
    unmatched = list(names)
    counts = {}
    for pname, rx, expected in patterns:
        hit = [n for n in names if re.match(rx, n)]
        counts[pname] = len(hit)
        if len(hit) != expected:
            surprises.append(f"group {pname}: matched {len(hit)}, expected {expected}")
        unmatched = [n for n in unmatched if not re.match(rx, n)]
    if unmatched:
        surprises.append(f"UNMATCHED TENSORS ({len(unmatched)}): {unmatched[:10]}")

    # imatrix coverage: every ablatable matmul tensor except token_embd must
    # have activation stats (unsloth imatrix is GGUF-format; entries are
    # <tensor>.in_sum2 / <tensor>.counts)
    im = GGUFReader(IMATRIX)
    im_names = {re.sub(r"\.(in_sum2|counts)$", "", t.name) for t in im.tensors}
    need = [n for n in names
            if any(re.match(rx, n) for (g, rx, c) in GROUPS) and n != "token_embd.weight"]
    gaps = [n for n in need if n not in im_names]
    if gaps:
        surprises.append(f"IMATRIX COVERAGE GAPS ({len(gaps)}): {gaps[:10]}")

    out = {
        "ok": not surprises,
        "surprises": surprises,
        "tensor_total": len(names),
        "group_counts": counts,
        "imatrix_tensors": len(im_names),
        "meta": {k: str(v) for k, v in meta.items()},
        "llama_sha": _llama_sha(),
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(
    image=llama_image,
    cpu=12.0,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={V: vol},
)
def quantize_vol(out_rel: str, base_type: str, override_text: str = "") -> dict:
    """llama-quantize Volume BF16 -> Volume GGUF with optional tensor-type
    override (group ablation).  Skips if output already exists (resume)."""
    out_path = f"{V}/{out_rel}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e9:
        print(f"exists, skipping: {out_path}")
        return {"out": out_rel, "bytes": os.path.getsize(out_path), "skipped": True}
    if not _wait_on_volume(BF16):
        raise RuntimeError(f"BF16 not on volume: {BF16}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    cmd = [f"{CPU_BUILD_DIR}/llama-quantize", "--imatrix", IMATRIX]
    clean = _sanitize_override(override_text)
    if clean:
        ovr = "/tmp/tensor_types.txt"
        with open(ovr, "w") as f:
            f.write(clean + "\n")
        print(f"override:\n{clean}", flush=True)
        cmd += ["--tensor-type-file", ovr]
    cmd += [BF16, tmp, base_type, "12"]

    t0 = time.time()
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    tail = "\n".join(output.splitlines()[-20:])
    print(tail, flush=True)
    if proc.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"quantize {out_rel} failed rc={proc.returncode}:\n{tail}")
    os.replace(tmp, out_path)
    vol.commit()

    fallbacks = sorted({ln.strip() for ln in output.splitlines()
                        if "fall" in ln.lower() or "requantiz" in ln.lower()})[:20]
    return {
        "out": out_rel,
        "bytes": os.path.getsize(out_path),
        "secs": round(time.time() - t0, 1),
        "fallback_lines": fallbacks,
        "converting_counts": _count_conversions(output),
        "llama_sha": _llama_sha(),
    }


def _count_conversions(output: str) -> dict:
    """Histogram of 'converting to <type>' lines — the proof that a group
    override actually matched (the 12B no-op lesson)."""
    counts: dict = {}
    for m in re.finditer(r"converting to (\w+)", output):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def _ppl_impl(filename: str, corpora: list[str], ngl: int) -> dict:
    binary = f"{BUILD_DIR}/llama-perplexity"
    model = f"{V}/{filename}"
    if not _wait_on_volume(model):
        raise RuntimeError(f"model not on volume after backoff: {model}")
    results, timings = {}, {}
    for c in corpora:
        corpus = f"{CORPORA_DIR}/{c}"
        if not os.path.exists(corpus):
            raise RuntimeError(f"corpus not on volume: {corpus}")
        cmd = [binary, "--model", model, "--file", corpus,
               "-ngl", str(ngl), "--ctx-size", "2048", "--chunks", "-1"]
        t0 = time.time()
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
        output = proc.stdout + proc.stderr
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
        if not m:
            print("\n".join(output.splitlines()[-30:]), flush=True)
            raise RuntimeError(f"PPL parse failed for {filename} on {c} (rc={proc.returncode})")
        results[c] = float(m.group(1))
        timings[c] = round(time.time() - t0, 1)
        print(f"  {c}: PPL {results[c]:.4f} in {timings[c]:.0f}s", flush=True)
    return {"ppl": results, "secs": timings, "llama_sha": _llama_sha()}


@app.function(image=llama_image, cpu=4.0, memory=24576, timeout=1800,
              retries=0, volumes={V: vol})
def vocab_smoke(filename: str = "baselines/north-mini-code-Q2_K.gguf") -> dict:
    """CPU-only proof that the cohere2moe vocab patch took: load the smallest
    baseline with build-cpu llama-cli and generate 1 token.  Pennies vs
    burning an L40S spin-up to find out the image is still broken."""
    model = f"{V}/{filename}"
    if not _wait_on_volume(model):
        raise RuntimeError(f"model not on volume: {model}")
    cmd = [f"{CPU_BUILD_DIR}/llama-cli", "--model", model, "-p", "hi",
           "-n", "1", "-ngl", "0", "--no-warmup", "-no-cnv"]
    # -no-cnv is load-bearing: with a chat-template model llama-cli defaults
    # to conversation mode and blocks on stdin forever (the first smoke run
    # burned a 1500s timeout exactly this way). stdin=DEVNULL as a backstop.
    print("+ " + " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900, stdin=subprocess.DEVNULL)
        output = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        def _s(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        output = _s(e.stdout) + _s(e.stderr)
        rc = -1
        print("TIMEOUT — partial output below", flush=True)
    tail = "\n".join(output.splitlines()[-25:])
    print(tail, flush=True)
    if "unknown pre-tokenizer" in output:
        raise RuntimeError(f"vocab smoke FAILED — pre-tokenizer still unknown:\n{tail}")
    if rc != 0:
        raise RuntimeError(f"vocab smoke FAILED rc={rc}:\n{tail}")
    return {"ok": True, "llama_sha": _llama_sha()}


@app.function(image=llama_image, gpu="T4", cpu=4.0, memory=16384,
              timeout=3600, retries=0, volumes={V: vol})
def ppl_t4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=llama_image, gpu="L4", cpu=4.0, memory=24576,
              timeout=3600, retries=0, volumes={V: vol})
def ppl_l4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=llama_image, gpu="L40S", cpu=4.0, memory=49152,
              timeout=5400, retries=0, volumes={V: vol})
def ppl_l40s(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


# ---------------------------------------------------------------------------
# Driver (runs ON MODAL — survives anything local)
# ---------------------------------------------------------------------------


@app.function(
    image=driver_image,
    cpu=1.0,
    memory=2048,
    timeout=int(8.5 * 3600),  # under the local watchdog's 9 h app kill
    retries=0,
    secrets=[modal_token_secret],
    volumes={V: vol},
)
def campaign_driver(budget_usd: float = 10.0) -> dict:
    t_driver = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ledger_path = f"{RESULTS_DIR}/cost_ledger.json"
    ledger = _load_json(ledger_path, {"entries": [], "total_usd": 0.0})

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(f"{RESULTS_DIR}/progress.log", "a") as f:
            f.write(line + "\n")

    def add_cost(kind: str, secs: float, note: str = ""):
        usd = RATES[kind] * secs
        ledger["entries"].append({"kind": kind, "secs": round(secs, 1),
                                  "usd": round(usd, 4), "note": note})
        ledger["total_usd"] = round(sum(e["usd"] for e in ledger["entries"]), 4)
        _dump_json(ledger_path, ledger)
        vol.commit()
        return usd

    def gate(phase: str):
        driver_usd = RATES["driver"] * (time.time() - t_driver)
        projected = ledger["total_usd"] + driver_usd + PHASE_EST[phase]
        if projected > budget_usd:
            log(f"BUDGET ABORT before {phase}: spent~${ledger['total_usd']:.2f} "
                f"+ driver ${driver_usd:.2f} + est ${PHASE_EST[phase]:.2f} "
                f"> budget ${budget_usd:.2f}")
            _dump_json(f"{RESULTS_DIR}/ABORTED.json",
                       {"phase": phase, "ledger": ledger})
            vol.commit()
            raise SystemExit(f"budget abort before {phase}")
        log(f"gate OK for {phase}: ledger ${ledger['total_usd']:.2f}, "
            f"est +${PHASE_EST[phase]:.2f}, budget ${budget_usd:.2f}")

    def timed(kind: str, fn, *a, note: str = "", **kw):
        t0 = time.time()
        r = fn.remote(*a, **kw)
        usd = add_cost(kind, time.time() - t0, note)
        log(f"  {note or kind}: {time.time()-t0:.0f}s  ~${usd:.3f}  "
            f"(cum ${ledger['total_usd']:.2f})")
        return r

    # --- GPU gate: REMOVED 2026-06-12. The Flash stage-4 app is long dead,
    # and the poll never worked anyway: Modal IGNORES MODAL_TOKEN_ID/SECRET
    # env vars inside containers, so the in-container `modal app list --json`
    # was always unauthenticated and emitted non-JSON — every dispatch burned
    # 3x180s of retry spam before proceeding. The local watchdog still
    # enforces the $4/hr velocity kill and the $28.50 doors-close.
    def wait_gpu_clear():
        return

    def pick_ppl(nbytes: int):
        # T4 tier removed: the image compiles sm89 only (see llama_image).
        if nbytes < 21.0e9:
            return ppl_l4, "ppl_l4"
        return ppl_l40s, "ppl_l40s"

    def run_ppl(filename: str, nbytes: int, corpora: list[str], note: str) -> dict:
        wait_gpu_clear()
        fn, kind = pick_ppl(nbytes)
        t0 = time.time()
        try:
            r = fn.remote(filename, corpora)
        except Exception as e:
            add_cost(kind, time.time() - t0, note + " (failed)")
            log(f"  {note}: {kind} failed ({e}); retrying on L40S")
            wait_gpu_clear()
            t0 = time.time()
            r = ppl_l40s.remote(filename, corpora)
            kind = "ppl_l40s"
        add_cost(kind, time.time() - t0, note)
        log(f"  {note}: {r['ppl']}  (cum ${ledger['total_usd']:.2f})")
        return r

    def rm(rel: str):
        vol.reload()
        p = f"{V}/{rel}"
        if os.path.exists(p):
            os.remove(p)
            vol.commit()
            log(f"  deleted {rel} from volume")

    log(f"=== Cerebellum North-Mini-Code-1.0 campaign driver "
        f"(budget ${budget_usd:.2f}) ===")
    log(f"groups: {[g[0] for g in GROUPS]}")

    # ---------------- phase 0: convert + imatrix + verify ----------------
    gate("phase0")
    vol.reload()
    missing = [c for c in [f"cerebellum_calibration_{d}.txt" for d in DOMAINS] + [WIKITEST]
               if not os.path.exists(f"{CORPORA_DIR}/{c}")]
    if missing:
        log(f"FATAL: corpora missing from volume: {missing} — "
            f"upload with `modal volume put cerebellum-north <file> corpora/<name>`")
        raise SystemExit("missing corpora")
    if not (os.path.exists(BF16) and os.path.getsize(BF16) > 55e9):
        log("phase 0: converting on Modal (download 61 GB + PR #24260 converter)")
        r = timed("convert", convert_and_fetch, note="convert+imatrix")
        log(f"phase 0 convert done: { {k: v for k, v in r.items()} }")
    else:
        log("phase 0: BF16 already on volume, skipping convert")
        if not os.path.exists(IMATRIX):
            timed("convert", convert_and_fetch, note="imatrix fetch")
    vol.reload()

    verify_path = f"{RESULTS_DIR}/verify_report.json"
    ver = _load_json(verify_path, None)
    if not (ver and ver.get("ok")):
        ver = timed("verify", verify_assets, note="verify tensor inventory")
        _dump_json(verify_path, ver)
        vol.commit()
    if not ver["ok"]:
        log(f"HARD STOP — architecture surprises: {ver['surprises']}")
        _dump_json(f"{RESULTS_DIR}/SURPRISE.json", ver)
        vol.commit()
        raise SystemExit("verify hard stop — see results/SURPRISE.json")
    log(f"phase 0 verify OK: {ver['tensor_total']} tensors, "
        f"groups {ver['group_counts']}, llama.cpp {ver.get('llama_sha', '?')[:12]}")

    # ---------------- phase 1: uniform baselines ----------------
    gate("phase1")
    baselines_path = f"{RESULTS_DIR}/baselines.json"
    baselines = _load_json(baselines_path, {})
    base_specs = [("Q8_0", "baselines/north-mini-code-Q8_0.gguf"),
                  ("Q4_K_M", "baselines/north-mini-code-Q4_K_M.gguf"),
                  ("Q3_K_M", "baselines/north-mini-code-Q3_K_M.gguf"),
                  ("Q2_K", "baselines/north-mini-code-Q2_K.gguf")]
    # build with AT MOST 2 CPU containers in flight (watchdog velocity cap)
    todo = [(q, rel) for q, rel in base_specs
            if not (q in baselines and "ppl" in baselines[q])]
    qi = 0
    inflight: dict = {}

    def spawn_base(i):
        if i < len(todo):
            q, rel = todo[i]
            inflight[i] = (quantize_vol.spawn(rel, q), time.time(), q)
            log(f"  spawned baseline build {q}")

    spawn_base(0)
    spawn_base(1)
    while qi < len(todo):
        h, t0, q = inflight.pop(qi)
        b = h.get()
        add_cost("quantize", time.time() - t0, f"build {q}")
        baselines.setdefault(q, {})["build"] = b
        _dump_json(baselines_path, baselines)
        vol.commit()
        log(f"  built {q}: {b['bytes']/1e9:.2f} GB")
        qi += 1
        spawn_base(qi + 1)
    vol.reload()
    cal_corpora = [f"cerebellum_calibration_{d}.txt" for d in DOMAINS]
    for qtype, rel in base_specs:
        if "ppl" in baselines.get(qtype, {}):
            log(f"  skip PPL {qtype} (already measured)")
            continue
        nbytes = baselines[qtype]["build"]["bytes"]
        r = run_ppl(rel, nbytes, cal_corpora + [WIKITEST], f"PPL {qtype}")
        baselines[qtype]["ppl"] = r["ppl"]
        baselines[qtype]["ppl_secs"] = r["secs"]
        _dump_json(baselines_path, baselines)
        vol.commit()
        if qtype == "Q8_0":
            rm(rel)  # ~30 GB — reference only, free the storage
    log(f"phase 1 done: {json.dumps({k: v.get('ppl') for k, v in baselines.items()})}")

    # ---------------- phase 2: group ablation ----------------
    gate("phase2")
    abl_path = f"{RESULTS_DIR}/ablation_results_multidomain.json"
    abl = _load_json(abl_path, {})
    abl.setdefault("model", HF_MODEL)
    abl.setdefault("arch", "cohere2moe (PR #24260)")
    abl.setdefault("base_type", BASE_TYPE)
    abl.setdefault("ablate_type", ABLATE_TYPE)
    abl.setdefault("llama_cpp", f"PR#{LLAMA_PR}@{LLAMA_PR_COMMIT[:12]}")
    abl["baseline_ppl"] = {d: baselines[BASE_TYPE]["ppl"][f"cerebellum_calibration_{d}.txt"]
                           for d in DOMAINS}
    abl["baseline_bytes"] = baselines[BASE_TYPE]["build"]["bytes"]
    abl.setdefault("corpus_versions",
                   {d: f"volume:corpora/cerebellum_calibration_{d}.txt" for d in DOMAINS})
    abl.setdefault("notes", {
        "token_embd": "embeddings are TIED — token_embd ablation also crushes the LM head",
        "imatrix": "unsloth imatrix_unsloth.gguf_file (pre-merge PR provenance)",
    })
    tests = abl.setdefault("tests", {})
    _dump_json(abl_path, abl)
    vol.commit()

    pending = [(n, rx, cnt) for (n, rx, cnt) in GROUPS
               if not (n in tests and isinstance(tests[n].get("ppl"), dict))]
    log(f"phase 2: {len(pending)} groups pending (of {len(GROUPS)})")

    build_handles: dict = {}

    def spawn_build(i: int):
        if i >= len(pending):
            return
        name, regex, _ = pending[i]
        rel = f"candidates/ablate_{name}.gguf"
        override = f"{regex}={ABLATE_TYPE}"
        build_handles[i] = (quantize_vol.spawn(rel, BASE_TYPE, override), time.time(), rel)
        log(f"  spawned build [{i+1}/{len(pending)}] {name}")

    spawn_build(0)
    spawn_build(1)

    # PPL pipeline: up to PPL_LANES measurements in flight alongside the CPU
    # builds. Same total GPU-seconds, ~half the wall clock. An L40S-class PPL
    # always runs solo — two concurrent L40S would trip the $4/hr watchdog;
    # T4/L4 pairs stay well under it.
    ppl_q: list = []

    def ppl_record(item, r):
        name = item["name"]
        ppl = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"] for d in DOMAINS}
        tests[name] = {
            "gguf_tensor": item["regex"],
            "tensor_count": item["cnt"],
            "ppl": ppl,
            "delta": {d: round(ppl[d] - abl["baseline_ppl"][d], 4) for d in DOMAINS},
            "delta_pct": {d: round(100 * (ppl[d] / abl["baseline_ppl"][d] - 1), 2)
                          for d in DOMAINS},
            "candidate_bytes": item["bytes"],
            "bytes_saved_vs_base": abl["baseline_bytes"] - item["bytes"],
            "q2k_converted": item["crushed"],
            "fallback_lines": item["fallback_lines"],
        }

    def ppl_finish(item):
        name = item["name"]
        try:
            r = item["handle"].get()
            add_cost(item["kind"], time.time() - item["t0"], f"PPL ablate {name}")
            ppl_record(item, r)
        except Exception as e:
            add_cost(item["kind"], time.time() - item["t0"],
                     f"PPL ablate {name} (failed)")
            log(f"  PPL {name}: {item['kind']} failed ({e}); retrying on L40S solo")
            ppl_drain(all_lanes=True)  # never pair the L40S with another lane
            wait_gpu_clear()
            t0 = time.time()
            try:
                r = ppl_l40s.remote(item["rel"], cal_corpora)
                add_cost("ppl_l40s", time.time() - t0, f"PPL ablate {name} (retry)")
                ppl_record(item, r)
            except Exception as e2:
                add_cost("ppl_l40s", time.time() - t0,
                         f"PPL ablate {name} (retry failed)")
                log(f"  PPL {name} FAILED: {e2}")
                tests[name] = {"gguf_tensor": item["regex"],
                               "tensor_count": item["cnt"],
                               "error": f"ppl_failed: {e2}"}
        _dump_json(abl_path, abl)
        vol.commit()
        rm(item["rel"])
        log(f"[{item['i']+1}/{len(pending)}] {name} done: "
            f"{tests[name].get('delta_pct', tests[name].get('error'))}")

    def ppl_drain(all_lanes: bool = False):
        while ppl_q and (all_lanes or len(ppl_q) >= PPL_LANES):
            ppl_finish(ppl_q.pop(0))

    def ppl_spawn(item):
        fn, kind = pick_ppl(item["bytes"])
        if kind == "ppl_l40s":
            ppl_drain(all_lanes=True)  # L40S runs solo (velocity cap)
        wait_gpu_clear()
        item["handle"] = fn.spawn(item["rel"], cal_corpora)
        item["t0"] = time.time()
        item["kind"] = kind
        ppl_q.append(item)
        log(f"  PPL spawned [{item['i']+1}/{len(pending)}] {item['name']} on {kind}")

    for i, (name, regex, cnt) in enumerate(pending):
        # per-group budget check: stop CLEANLY at the last completed group
        # (margin covers this group plus whatever is still in flight)
        driver_usd = RATES["driver"] * (time.time() - t_driver)
        margin = 0.60 * (1 + len(ppl_q))
        if ledger["total_usd"] + driver_usd + margin > budget_usd:
            log(f"BUDGET STOP inside phase 2 before group {name}: "
                f"ledger ${ledger['total_usd']:.2f} + driver ${driver_usd:.2f} "
                f"+ ~${margin:.2f} in flight > ${budget_usd:.2f}. "
                f"Partial map is still data.")
            break
        h, t0, rel = build_handles.pop(i)
        try:
            b = h.get()
        except Exception as e:
            add_cost("quantize", time.time() - t0, f"build {name} (failed)")
            log(f"  build {name} FAILED: {e}")
            tests[name] = {"gguf_tensor": regex, "tensor_count": cnt,
                           "error": f"quantize_failed: {e}"}
            _dump_json(abl_path, abl)
            vol.commit()
            spawn_build(i + 2)
            continue
        add_cost("quantize", time.time() - t0, f"build {name}")
        spawn_build(i + 2)  # keep 2 builds in flight while the GPUs work

        conv = b.get("converting_counts", {})
        crushed = sum(v for k, v in conv.items()
                      if k.lower() == ABLATE_TYPE.lower())
        if not b.get("skipped") and crushed == 0:
            log(f"  WARNING {name}: no tensors converted to {ABLATE_TYPE} — "
                f"override may not have matched! conv={conv}")
        vol.reload()
        ppl_drain()  # free a lane if both are busy
        ppl_spawn({"i": i, "name": name, "regex": regex, "cnt": cnt,
                   "rel": rel, "bytes": b["bytes"], "crushed": crushed,
                   "fallback_lines": b.get("fallback_lines", [])})
    ppl_drain(all_lanes=True)

    # ---------------- phase 3: summary + STOP ----------------
    add_cost("driver", time.time() - t_driver, "driver wall time")
    summary = _render_summary(abl, baselines, ledger)
    with open(f"{RESULTS_DIR}/ablation_summary.md", "w") as f:
        f.write(summary)
    _dump_json(abl_path, abl)
    vol.commit()
    log("=== phase 2 complete. STOPPING per the method: allocation builds and "
        "benchmarks gate LOCALLY (code model — BigCodeBench needs the local "
        "harness). See cerebellum-north-mini-code/RUN_PLAN.md. ===")
    log(f"total estimated spend: ${ledger['total_usd']:.2f}")
    print(summary, flush=True)
    return {"ledger_usd": ledger["total_usd"], "tests": len(tests)}


# ---------------------------------------------------------------------------
# Summary / verdicts
# ---------------------------------------------------------------------------


def _verdict(delta_pct: dict) -> str:
    mx = max(delta_pct.values())
    if mx >= 5.0:
        return "PROTECT"
    if mx >= 1.0:
        return "TOLERANT"
    return "DEMOTABLE"


def _render_summary(abl: dict, baselines: dict, ledger: dict) -> str:
    lines = [
        "# North-Mini-Code-1.0 group ablation — multi-domain summary",
        "",
        f"- model: {abl['model']}  (arch {abl['arch']})",
        f"- base {abl['base_type']} + group->{abl['ablate_type']}, "
        f"imatrix: unsloth (pre-merge PR provenance), "
        f"llama.cpp {abl.get('llama_cpp', '?')}",
        f"- baseline PPL ({abl['base_type']}): "
        + ", ".join(f"{d}={abl['baseline_ppl'][d]:.4f}" for d in DOMAINS),
        f"- estimated Modal spend: ${ledger['total_usd']:.2f}",
        "- embeddings TIED: token_embd row doubles as the LM-head ablation",
        "",
        "## Uniform baselines",
        "",
        "| build | size GB | " + " | ".join(DOMAINS) + " | wikitext |",
        "|---|---|" + "---|" * (len(DOMAINS) + 1),
    ]
    for q, v in baselines.items():
        if "ppl" not in v:
            continue
        p = v["ppl"]
        lines.append(
            f"| {q} | {v['build']['bytes']/1e9:.2f} | "
            + " | ".join(f"{p[f'cerebellum_calibration_{d}.txt']:.4f}" for d in DOMAINS)
            + f" | {p.get(WIKITEST, float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Group ablation (crush group to Q2_K, ΔPPL% vs Q4_K_M base)",
        "",
        "| group | tensors | saved GB | " + " | ".join(f"Δ%{d}" for d in DOMAINS)
        + " | verdict |",
        "|---|---|---|" + "---|" * (len(DOMAINS) + 1),
    ]
    for name, _, _ in GROUPS:
        t = abl["tests"].get(name)
        if not t:
            lines.append(f"| {name} | - | - | " + " | ".join("-" for _ in DOMAINS)
                         + " | UNMEASURED |")
            continue
        if "error" in t:
            lines.append(f"| {name} | {t['tensor_count']} | - | "
                         + " | ".join("-" for _ in DOMAINS)
                         + f" | ERROR: {t['error'][:40]} |")
            continue
        dp = t["delta_pct"]
        lines.append(
            f"| {name} | {t['tensor_count']} | {t['bytes_saved_vs_base']/1e9:.2f} | "
            + " | ".join(f"{dp[d]:+.2f}" for d in DOMAINS)
            + f" | {_verdict(dp)} |"
        )
    lines += [
        "",
        "Verdict rule: PROTECT if any domain ΔPPL ≥ +5%; TOLERANT if ≥ +1%; "
        "else DEMOTABLE (incl. negative = regularizing).  Rows marked "
        "UNMEASURED were skipped (budget stop) — partial map, still data.",
        "",
        "NEXT (human review required — no-bullshit rule): allocation builds "
        "from these verdicts, then LOCAL gates (HumanEval+/BigCodeBench vs "
        "same-size uniform baselines).  See cerebellum-north-mini-code/"
        "RUN_PLAN.md.  Re-imatrix after PR #24260 merges before finals.",
    ]
    return "\n".join(lines) + "\n"


def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _dump_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Local entrypoint — spawn the driver and exit (use with `modal run --detach`)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def launch(budget_usd: float = 10.0):
    fc = campaign_driver.spawn(budget_usd=budget_usd)
    print(f"campaign driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-north-campaign")
    print("results:  modal volume ls cerebellum-north results")


@app.local_entrypoint()
def smoke():
    """Run only the CPU vocab smoke-test (no driver, no GPU)."""
    print(json.dumps(vocab_smoke.remote(), indent=2))
