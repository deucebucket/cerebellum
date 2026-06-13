"""Cerebellum Qwen 3.6 35B-A3B v4 campaign — tensor-level sensitivity pass.

WHY: the flagship (Qwen3.6-35B-A3B, shipped v3) ships on a coarse GROUP-level
map (360 overrides, 9 whole groups -> Q2_K on a Q3_K_M base).  It never got
the tensor-level budget treatment that made 27B v4 (sparse single-tensor PPL
probes + allocator extrapolation -> 181 graded overrides).  This campaign
produces that tensor-level sensitivity data and STOPS — allocation and
benchmark gates happen later, locally, human-reviewed.

PROVENANCE (verified 2026-06-12 against local artifacts):
  - v3 source model: Qwen/Qwen3.6-35B-A3B (BASE, not -Instruct) — from
    games/qwen36-35b-v2/download_and_convert.py and WINNING_METHOD.md §C.
  - v3 imatrix: games/qwen36-35b-v2/imatrix_unsloth.gguf_file — 192 MB,
    GGUF-format, 1020 tensors, 76 chunks, Unsloth coder calibration.  The
    EXACT local file is uploaded to the Volume (not re-fetched from HF, so
    provenance is byte-identical to the shipped v3 build).
  - v3 override file: games/qwen36-35b-v2/cerebellum_v3_overrides.txt
    (360 lines, all Q2_K: ffn_{gate,up,down}_exps, ffn_{gate,up,down}_shexp,
    attn_gate, ssm_alpha, ssm_beta x blk.0-39), uploaded verbatim.
  - v3 build: base Q3_K_M + that override file -> 11.96 GB shipped GGUF.
    Known anchors (local protocol): v3 wiki PPL 7.4307, uniform Q3_K_M 7.1758.

MODEL FACTS (measured from the local shipped v3 GGUF):
  - arch `qwen35moe`, block_count=40, 256 experts / 8 used, hybrid SSM:
    30 layers are linear-attention/SSM (fused attn_qkv + attn_gate + ssm_*),
    10 layers are full attention (separate attn_q/k/v/output).  733 tensors.
  - THE THREE HEAVY GROUPS for the tensor pass (per the OG guide): attn_qkv
    (30 tensors), ffn_down_exps (40), ffn_up_exps (40) = 110 tensors.
  - llama.cpp b9603 release supports qwen35moe (local b8925+ already does;
    b9603 is newer) — verify_assets hard-stops on any surprise anyway.

P2 PROBE DESIGN (the point of the campaign):
  Probes are deltas vs the rebuilt v3 reference, all on the v3 override base:
  - attn_qkv.blk.N (Q3_K-class in v3, protected): DEMOTE probe — append
    `blk.N.attn_qkv.weight=Q2_K`.  Small +delta => demotable layer (savings).
  - ffn_{down,up}_exps.blk.N (Q2_K in v3): PROMOTE probe — replace that
    line with `=Q4_K`.  Strong -delta => best place to spend v4 budget.
  120 exhaustive probes would cost $10-14; instead: wave 1 samples every 4th
  layer (~28 probes, ~$3), then adaptive drilling of the layers whose
  neighbor deltas vary most, hard-capped at P2_CAP_USD ($6).  Partial maps
  are still data.  4-domain calibration PPL per probe; candidate GGUF
  deleted after PPL.

PIPELINE (driver ON MODAL — survives local death; artifacts on the
`cerebellum-35b-v4` Volume; cost ledger + resume state on the Volume):
  phase 0: download Qwen/Qwen3.6-35B-A3B in-datacenter, convert BF16
           (~65 GB) onto the Volume with the b9603 converter; verify tensor
           inventory + qkv-layer enumeration + imatrix coverage (hard stop
           on surprises).  Imatrix/overrides/corpora are PRE-UPLOADED from
           local via `modal volume put` (driver hard-stops if missing).
  phase 1: rebuild the v3 reference (Q3_K_M + v3 overrides) + uniform
           Q3_K_M baseline; 4-domain + wiki.test.raw PPL on both; sanity
           anchor vs the shipped numbers (warn-only: different llama.cpp
           build).  Both GGUFs KEPT on the volume for later allocation work.
  phase 2: tensor probes as above.  2 CPU builds pipelined ahead of the GPU;
           ONE GPU container at a time (account-wide velocity discipline —
           the local launcher already serializes vs the North phase 3 app).
  phase 3: v4_tensor_sensitivity.json + summary, STOP.  Allocation + benches
           later, locally, reviewed.

BUDGET DISCIPLINE: retries=0 everywhere; cost ledger on the Volume; phase
gates (P0 $1.20 / P1 $1.00 / P2 $6.00) against budget (default $9.00);
per-probe budget check stops CLEANLY mid-P2.
  Gate locally before EVERY launch:  python3 modal_credits.py --gate 9

LAUNCH (two-step; GPU phases wait for the North phase-3 app LOCALLY):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  modal volume create cerebellum-35b-v4                       # once
  modal volume put cerebellum-35b-v4 <corpora...> corpora/<name>   # once
  modal volume put cerebellum-35b-v4 .../imatrix_unsloth.gguf_file imatrix/
  modal volume put cerebellum-35b-v4 .../cerebellum_v3_overrides.txt overrides/
  nohup setsid modal run --detach cerebellum_35b_v4_campaign.py::launch_p0 \
      > ../../cerebellum-qwen36-35b-v4/logs/modal_p0_launch.log 2>&1 &
  # then the local sequencer (cerebellum-qwen36-35b-v4/launch_when_clear.sh)
  # polls `modal app list` until NO app is running, re-gates credits, and:
  nohup setsid modal run --detach cerebellum_35b_v4_campaign.py::launch \
      > ../../cerebellum-qwen36-35b-v4/logs/modal_launch.log 2>&1 &

MONITOR:
  modal app list                              # cerebellum-35b-v4-campaign
  modal app logs cerebellum-35b-v4-campaign
  modal volume ls cerebellum-35b-v4 results
  python3 modal_credits.py
"""

import json
import os
import re
import subprocess
import time

import modal

from cerebellum_modal import (
    CUDA_IMAGE_REF,
    LLAMA_CPP_TAG,
    LLAMA_DIR,
    TARBALL_URL,
    _sanitize_override,
)

app = modal.App("cerebellum-35b-v4-campaign")

hf_secret = modal.Secret.from_name("hf-token")
vol = modal.Volume.from_name("cerebellum-35b-v4", create_if_missing=True)
V = "/vol"

HF_MODEL = "Qwen/Qwen3.6-35B-A3B"  # the exact v3 base (NOT -Instruct)

BF16 = f"{V}/qwen36-35b-a3b-bf16.gguf"
IMATRIX = f"{V}/imatrix/imatrix_unsloth.gguf_file"
V3_OVERRIDES_PATH = f"{V}/overrides/cerebellum_v3_overrides.txt"
CORPORA_DIR = f"{V}/corpora"
RESULTS_DIR = f"{V}/results"
DOMAINS = ["wiki", "code", "math", "dialogue"]
CAL = [f"cerebellum_calibration_{d}.txt" for d in DOMAINS]
WIKITEST = "wiki.test.raw"

BASE_TYPE = "Q3_K_M"          # v3 base
PROBE_DEMOTE_TYPE = "Q2_K"    # attn_qkv demote probes
PROBE_PROMOTE_TYPE = "Q4_K"   # exps promote probes

# Expected inventory, measured from the local shipped v3 GGUF (733 tensors).
EXPECTED_TOTAL = 733
EXPECTED_COUNTS = {
    r"^blk\.\d+\.attn_qkv\.weight$": 30,
    r"^blk\.\d+\.attn_q\.weight$": 10,
    r"^blk\.\d+\.attn_k\.weight$": 10,
    r"^blk\.\d+\.attn_v\.weight$": 10,
    r"^blk\.\d+\.attn_output\.weight$": 10,
    r"^blk\.\d+\.ffn_down_exps\.weight$": 40,
    r"^blk\.\d+\.ffn_up_exps\.weight$": 40,
    r"^blk\.\d+\.ffn_gate_exps\.weight$": 40,
}

# Sanity anchors from the shipped v3 evidence (local protocol, ctx 2048).
ANCHOR_V3_WIKI = 7.4307
ANCHOR_Q3KM_WIKI = 7.1758
ANCHOR_TOL = 0.25  # warn-only: b9603 != the local build that measured these

# ---------------------------------------------------------------------------
# Cost model (list prices, $/sec)
# ---------------------------------------------------------------------------
CPU_S = 0.0000131
MEM_S = 0.00000222
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
PHASE_EST = {"phase0": 1.20, "phase1": 1.00, "phase2": 6.00}
P2_CAP_USD = 6.00
BUDGET_USD = 9.00

# ---------------------------------------------------------------------------
# Images — b9603 release artifacts (the validated harness strategy; qwen35moe
# is supported at this tag).  No custom compile needed (unlike North's PR).
# ---------------------------------------------------------------------------

quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "libgomp1", "libssl3")
    .env({"LD_LIBRARY_PATH": LLAMA_DIR})
    .run_commands(
        f"curl -fsSL -o /tmp/llama.tar.gz {TARBALL_URL}",
        "tar xzf /tmp/llama.tar.gz -C /opt && rm /tmp/llama.tar.gz",
        f"{LLAMA_DIR}/llama-quantize --help 2>&1 | grep -q usage",
    )
    .add_local_python_source("cerebellum_modal", "cerebellum_35b_v4_campaign")
)

convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu",
        f"git clone --depth 1 --branch {LLAMA_CPP_TAG} "
        "https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
    )
    .pip_install(
        "transformers>=4.46",
        "sentencepiece",
        "safetensors",
        "numpy",
        "huggingface_hub[hf_transfer,hf_xet]>=0.30",
    )
    .env({"PYTHONPATH": "/opt/llama.cpp/gguf-py", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("cerebellum_modal", "cerebellum_35b_v4_campaign")
)

gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])
    .add_local_python_source("cerebellum_modal", "cerebellum_35b_v4_campaign")
)

driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_modal", "cerebellum_35b_v4_campaign"
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


def _count_conversions(output: str) -> dict:
    counts: dict = {}
    for m in re.finditer(r"converting to (\w+)", output):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=convert_image,
    cpu=8.0,
    memory=65536,
    timeout=10800,
    retries=0,
    secrets=[hf_secret],
    volumes={V: vol},
)
def convert_to_bf16() -> dict:
    """Download Qwen/Qwen3.6-35B-A3B safetensors inside Modal and convert to
    BF16 GGUF onto the Volume.  Idempotent.  The imatrix and v3 overrides are
    NOT fetched here — they are the exact local v3 artifacts, pre-uploaded."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    out = {}
    vol.reload()
    if os.path.exists(BF16) and os.path.getsize(BF16) > 55e9:
        print(f"BF16 already on volume ({os.path.getsize(BF16)/1e9:.1f} GB), skipping")
        return {"bf16_bytes": os.path.getsize(BF16), "skipped": True}

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
    cmd = ["python", "/opt/llama.cpp/convert_hf_to_gguf.py", src,
           "--outfile", tmp_out, "--outtype", "bf16"]
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
    image=convert_image,
    cpu=4.0,
    memory=16384,
    timeout=1800,
    retries=0,
    volumes={V: vol},
)
def verify_assets() -> dict:
    """Reconcile the converted BF16 against the measured v3 inventory,
    enumerate the qkv layers, check imatrix coverage of the probe tensors,
    and validate the uploaded v3 override file.  Hard-stop on surprises."""
    from gguf import GGUFReader

    vol.reload()
    for p in (BF16, IMATRIX, V3_OVERRIDES_PATH):
        if not _wait_on_volume(p):
            raise RuntimeError(f"asset not on volume: {p}")

    r = GGUFReader(BF16)
    names = [t.name for t in r.tensors]
    meta = {}
    for f in r.fields.values():
        if any(s in f.name for s in
               ("architecture", "block_count", "expert", "ssm", "vocab_size")):
            try:
                meta[f.name] = f.contents()
            except Exception:
                pass

    surprises = []
    arch = str(meta.get("general.architecture", ""))
    if arch != "qwen35moe":
        surprises.append(f"arch {arch!r} != qwen35moe")
    if [n for n in names if "nextn" in n] or any(n.startswith("blk.40.") for n in names):
        surprises.append("MTP/nextn or blk.40+ tensors present (heretic lesson — fatal)")
    if len(names) != EXPECTED_TOTAL:
        surprises.append(f"tensor count {len(names)} != expected {EXPECTED_TOTAL}")
    counts = {}
    for rx, expected in EXPECTED_COUNTS.items():
        hit = [n for n in names if re.match(rx, n)]
        counts[rx] = len(hit)
        if len(hit) != expected:
            surprises.append(f"{rx}: matched {len(hit)}, expected {expected}")

    qkv_layers = sorted(int(m.group(1)) for n in names
                        if (m := re.match(r"^blk\.(\d+)\.attn_qkv\.weight$", n)))
    exps_layers = sorted(int(m.group(1)) for n in names
                         if (m := re.match(r"^blk\.(\d+)\.ffn_down_exps\.weight$", n)))

    # v3 override file: must be 360 non-comment lines, every named tensor
    # must exist in the GGUF
    with open(V3_OVERRIDES_PATH) as f:
        ovr_lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if len(ovr_lines) != 360:
        surprises.append(f"v3 override file has {len(ovr_lines)} lines, expected 360")
    name_set = set(names)
    bad = [ln for ln in ovr_lines if ln.split("=")[0] not in name_set]
    if bad:
        surprises.append(f"override names not in GGUF ({len(bad)}): {bad[:5]}")

    # imatrix coverage for every probe tensor (GGUF-format imatrix:
    # <tensor>.in_sum2 / <tensor>.counts)
    im = GGUFReader(IMATRIX)
    im_names = {re.sub(r"\.(in_sum2|counts)$", "", t.name) for t in im.tensors}
    probe_tensors = (
        [f"blk.{n}.attn_qkv.weight" for n in qkv_layers]
        + [f"blk.{n}.ffn_down_exps.weight" for n in exps_layers]
        + [f"blk.{n}.ffn_up_exps.weight" for n in exps_layers]
    )
    gaps = [n for n in probe_tensors if n not in im_names]
    if gaps:
        surprises.append(f"IMATRIX COVERAGE GAPS ({len(gaps)}): {gaps[:10]}")

    out = {
        "ok": not surprises,
        "surprises": surprises,
        "tensor_total": len(names),
        "counts": counts,
        "qkv_layers": qkv_layers,
        "exps_layers": exps_layers,
        "imatrix_tensors": len(im_names),
        "override_lines": len(ovr_lines),
        "meta": {k: str(v) for k, v in meta.items()},
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(
    image=quant_image,
    cpu=12.0,
    memory=32768,
    timeout=5400,
    retries=0,
    volumes={V: vol},
)
def quantize_vol(out_rel: str, base_type: str, override_text: str = "") -> dict:
    """llama-quantize Volume BF16 -> Volume GGUF with tensor-type overrides.
    Skips if output already exists (resume)."""
    out_path = f"{V}/{out_rel}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e9:
        print(f"exists, skipping: {out_path}")
        return {"out": out_rel, "bytes": os.path.getsize(out_path), "skipped": True}
    if not _wait_on_volume(BF16):
        raise RuntimeError(f"BF16 not on volume: {BF16}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    cmd = [f"{LLAMA_DIR}/llama-quantize", "--imatrix", IMATRIX]
    clean = _sanitize_override(override_text)
    if clean:
        ovr = "/tmp/tensor_types.txt"
        with open(ovr, "w") as f:
            f.write(clean + "\n")
        print(f"override ({len(clean.splitlines())} lines), head:\n"
              + "\n".join(clean.splitlines()[:6]), flush=True)
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
    }


def _ppl_impl(filename: str, corpora: list[str], ngl: int) -> dict:
    binary = None
    for cand in ("/app/llama-perplexity", "/llama-perplexity",
                 "/usr/local/bin/llama-perplexity",
                 "/app/full/llama-perplexity"):
        if os.path.exists(cand):
            binary = cand
            break
    if binary is None:
        r = subprocess.run(["find", "/", "-maxdepth", "3", "-name",
                            "llama-perplexity"], capture_output=True, text=True)
        cands = r.stdout.split()
        if not cands:
            raise RuntimeError("llama-perplexity not found in CUDA image")
        binary = cands[0]
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)
        output = proc.stdout + proc.stderr
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
        if not m:
            print("\n".join(output.splitlines()[-30:]), flush=True)
            raise RuntimeError(f"PPL parse failed for {filename} on {c} "
                               f"(rc={proc.returncode})")
        results[c] = float(m.group(1))
        timings[c] = round(time.time() - t0, 1)
        print(f"  {c}: PPL {results[c]:.4f} in {timings[c]:.0f}s", flush=True)
    return {"ppl": results, "secs": timings}


@app.function(image=gpu_image, gpu="T4", cpu=4.0, memory=16384,
              timeout=3600, retries=0, volumes={V: vol})
def ppl_t4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=gpu_image, gpu="L4", cpu=4.0, memory=24576,
              timeout=3600, retries=0, volumes={V: vol})
def ppl_l4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=gpu_image, gpu="L40S", cpu=4.0, memory=49152,
              timeout=5400, retries=0, volumes={V: vol})
def ppl_l40s(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


# ---------------------------------------------------------------------------
# Probe override construction
# ---------------------------------------------------------------------------


def build_probe_override(v3_text: str, tensor: str, group: str) -> str:
    """attn_qkv: DEMOTE (append =Q2_K — no existing line matches attn_qkv).
    ffn_{down,up}_exps: PROMOTE (replace that tensor's existing =Q2_K line)."""
    lines = [ln for ln in v3_text.splitlines() if ln.strip()]
    if group == "attn_qkv":
        if any(ln.split("=")[0] == tensor for ln in lines):
            raise ValueError(f"unexpected existing override for {tensor}")
        lines.append(f"{tensor}={PROBE_DEMOTE_TYPE}")
    else:
        old = f"{tensor}={PROBE_DEMOTE_TYPE}"
        if old not in lines:
            raise ValueError(f"v3 override line not found: {old}")
        lines[lines.index(old)] = f"{tensor}={PROBE_PROMOTE_TYPE}"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver (runs ON MODAL)
# ---------------------------------------------------------------------------


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


@app.function(
    image=driver_image,
    cpu=1.0,
    memory=2048,
    timeout=int(8.5 * 3600),  # under the local watchdog's 9 h app kill
    retries=0,
    volumes={V: vol},
)
def campaign_driver(budget_usd: float = BUDGET_USD, gpu_phases: bool = True) -> dict:
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

    def pick_ppl(nbytes: int):
        if nbytes < 12.5e9:
            return ppl_t4, "ppl_t4"
        if nbytes < 21.0e9:
            return ppl_l4, "ppl_l4"
        return ppl_l40s, "ppl_l40s"

    def run_ppl(filename: str, nbytes: int, corpora: list[str], note: str) -> dict:
        """ONE GPU at a time, escalating fallback T4 -> L4 -> L40S."""
        tiers = [(ppl_t4, "ppl_t4"), (ppl_l4, "ppl_l4"), (ppl_l40s, "ppl_l40s")]
        tiers = [t for t in tiers
                 if t[1] == pick_ppl(nbytes)[1] or
                 ["ppl_t4", "ppl_l4", "ppl_l40s"].index(t[1])
                 > ["ppl_t4", "ppl_l4", "ppl_l40s"].index(pick_ppl(nbytes)[1])]
        last = None
        for fn, kind in tiers:
            t0 = time.time()
            try:
                r = fn.remote(filename, corpora)
                add_cost(kind, time.time() - t0, note)
                log(f"  {note} on {kind}: {r['ppl']} "
                    f"(cum ${ledger['total_usd']:.2f})")
                return r
            except Exception as e:
                add_cost(kind, time.time() - t0, f"{note} FAILED on {kind}")
                log(f"  {note} FAILED on {kind}: {e} — escalating")
                last = e
        raise last

    def rm(rel: str):
        vol.reload()
        p = f"{V}/{rel}"
        if os.path.exists(p):
            os.remove(p)
            vol.commit()
            log(f"  deleted {rel} from volume")

    log(f"=== Cerebellum Qwen3.6-35B-A3B v4 tensor-level campaign "
        f"(budget ${budget_usd:.2f}, gpu_phases={gpu_phases}) ===")

    # ---------------- phase 0: convert + verify ----------------
    gate("phase0")
    vol.reload()
    missing = [c for c in CAL + [WIKITEST]
               if not os.path.exists(f"{CORPORA_DIR}/{c}")]
    for p, what in ((IMATRIX, "imatrix"), (V3_OVERRIDES_PATH, "v3 overrides")):
        if not os.path.exists(p):
            missing.append(f"{what}: {p}")
    if missing:
        log(f"FATAL: assets missing from volume: {missing} — upload with "
            f"`modal volume put cerebellum-35b-v4 ...`")
        raise SystemExit("missing assets")

    if not (os.path.exists(BF16) and os.path.getsize(BF16) > 55e9):
        log("phase 0: downloading + converting on Modal (~70 GB download)")
        t0 = time.time()
        r = convert_to_bf16.remote()
        add_cost("convert", time.time() - t0, "download+convert BF16")
        log(f"phase 0 convert done: {r}")
    else:
        log("phase 0: BF16 already on volume, skipping convert")
    vol.reload()

    verify_path = f"{RESULTS_DIR}/verify_report.json"
    ver = _load_json(verify_path, None)
    if not (ver and ver.get("ok")):
        t0 = time.time()
        ver = verify_assets.remote()
        add_cost("verify", time.time() - t0, "verify inventory + imatrix + overrides")
        _dump_json(verify_path, ver)
        vol.commit()
    if not ver["ok"]:
        log(f"HARD STOP — surprises: {ver['surprises']}")
        _dump_json(f"{RESULTS_DIR}/SURPRISE.json", ver)
        vol.commit()
        raise SystemExit("verify hard stop — see results/SURPRISE.json")
    qkv_layers = ver["qkv_layers"]
    exps_layers = ver["exps_layers"]
    log(f"phase 0 verify OK: {ver['tensor_total']} tensors, "
        f"{len(qkv_layers)} qkv layers {qkv_layers[:8]}..., "
        f"imatrix tensors {ver['imatrix_tensors']}")

    if not gpu_phases:
        add_cost("driver", time.time() - t_driver, "p0 driver wall time")
        log("phase 0 complete; gpu_phases=False — stopping (the local "
            "sequencer launches the full driver once the GPU lane is clear).")
        return {"phase": "p0_done", "ledger_usd": ledger["total_usd"]}

    vol.reload()
    with open(V3_OVERRIDES_PATH) as f:
        v3_text = "\n".join(ln.strip() for ln in f
                            if ln.strip() and not ln.startswith("#"))

    # ---------------- phase 1: v3 reference + uniform baseline ----------------
    gate("phase1")
    baselines_path = f"{RESULTS_DIR}/baselines.json"
    baselines = _load_json(baselines_path, {})
    p1_specs = [
        ("v3_reference", "baselines/qwen36-35b-cerebellum-v3-rebuild.gguf", v3_text),
        ("Q3_K_M", "baselines/qwen36-35b-Q3_K_M.gguf", ""),
    ]
    handles = {}
    for name, rel, ovr in p1_specs:
        if "ppl" in baselines.get(name, {}):
            continue
        handles[name] = (quantize_vol.spawn(rel, BASE_TYPE, ovr), time.time())
        log(f"  spawned build {name}")
    for name, rel, ovr in p1_specs:
        if name not in handles:
            log(f"  skip {name} (already measured)")
            continue
        h, t0 = handles.pop(name)
        b = h.get()
        add_cost("quantize", time.time() - t0, f"build {name}")
        conv = b.get("converting_counts", {})
        if name == "v3_reference" and not b.get("skipped"):
            q2 = sum(v for k, v in conv.items() if k.lower() == "q2_k")
            if q2 < 300:
                log(f"  WARNING v3_reference: only {q2} Q2_K conversions "
                    f"(expected ~360) — override may not have fully applied! "
                    f"conv={conv}")
        baselines.setdefault(name, {})["build"] = b
        _dump_json(baselines_path, baselines)
        vol.commit()
        log(f"  built {name}: {b['bytes']/1e9:.2f} GB conv={conv}")
        r = run_ppl(rel, b["bytes"], CAL + [WIKITEST], f"PPL {name}")
        baselines[name]["ppl"] = r["ppl"]
        baselines[name]["ppl_secs"] = r["secs"]
        _dump_json(baselines_path, baselines)
        vol.commit()
    # sanity anchors (warn-only — b9603 vs the local build that shipped v3)
    try:
        v3_wiki = baselines["v3_reference"]["ppl"][WIKITEST]
        q3_wiki = baselines["Q3_K_M"]["ppl"][WIKITEST]
        for got, want, tag in ((v3_wiki, ANCHOR_V3_WIKI, "v3 rebuild"),
                               (q3_wiki, ANCHOR_Q3KM_WIKI, "uniform Q3_K_M")):
            if abs(got - want) > ANCHOR_TOL:
                log(f"  ANCHOR WARNING {tag}: wikitext PPL {got:.4f} vs "
                    f"shipped {want:.4f} (|Δ|>{ANCHOR_TOL}) — investigate "
                    f"before trusting probe deltas")
            else:
                log(f"  anchor OK {tag}: {got:.4f} vs shipped {want:.4f}")
    except Exception as e:
        log(f"  anchor check skipped: {e}")
    log("phase 1 done (both baseline GGUFs KEPT on the volume)")

    # ---------------- phase 2: tensor probes ----------------
    gate("phase2")
    probes_path = f"{RESULTS_DIR}/v4_tensor_probes.json"
    probes = _load_json(probes_path, {})
    probes.setdefault("model", HF_MODEL)
    probes.setdefault("arch", "qwen35moe")
    probes.setdefault("base", "v3 reference (Q3_K_M + 360 v3 overrides)")
    probes.setdefault("llama_cpp", LLAMA_CPP_TAG)
    probes.setdefault("design", {
        "attn_qkv": f"DEMOTE single tensor to {PROBE_DEMOTE_TYPE} "
                    f"(+delta = cost of demoting; small = v4 savings)",
        "ffn_down_exps": f"PROMOTE single tensor to {PROBE_PROMOTE_TYPE} "
                         f"(-delta = improvement; strong = v4 spend target)",
        "ffn_up_exps": "same as ffn_down_exps",
    })
    probes["reference_ppl"] = {
        d: baselines["v3_reference"]["ppl"][f"cerebellum_calibration_{d}.txt"]
        for d in DOMAINS}
    probes["reference_bytes"] = baselines["v3_reference"]["build"]["bytes"]
    tests = probes.setdefault("tests", {})
    _dump_json(probes_path, probes)
    vol.commit()

    ref_ppl = probes["reference_ppl"]
    p2_start_usd = ledger["total_usd"]

    def p2_spent():
        return ledger["total_usd"] - p2_start_usd

    def probe_key(group, layer):
        return f"{group}.blk.{layer}"

    def tensor_of(group, layer):
        return f"blk.{layer}.{group}.weight"

    # wave 1: every 4th layer per group, ROUND-ROBIN interleaved across the
    # three groups so a budget stop still leaves even coverage of all three
    GROUPS3 = ["attn_qkv", "ffn_down_exps", "ffn_up_exps"]
    per_group = {g: (qkv_layers if g == "attn_qkv" else exps_layers)[::4]
                 for g in GROUPS3}
    wave1 = []
    for i in range(max(len(v) for v in per_group.values())):
        for g in GROUPS3:
            if i < len(per_group[g]):
                wave1.append((g, per_group[g][i]))

    def measured(g):
        return sorted((int(k.rsplit(".", 1)[1]), tests[k])
                      for k in tests
                      if k.startswith(g + ".blk.") and "delta_pct" in tests[k])

    def drill_candidates():
        """Unprobed layers ranked by the delta gap between their nearest
        probed neighbors (per group) — drill where the map varies most."""
        cands = []
        for g in GROUPS3:
            layers = qkv_layers if g == "attn_qkv" else exps_layers
            done = measured(g)
            if len(done) < 2:
                continue
            for (l0, t0_), (l1, t1_) in zip(done, done[1:]):
                gap_layers = [layer for layer in layers if l0 < layer < l1
                              and probe_key(g, layer) not in tests]
                if not gap_layers:
                    continue
                d0 = max(abs(v) for v in t0_["delta_pct"].values())
                d1 = max(abs(v) for v in t1_["delta_pct"].values())
                spread = abs(d1 - d0)
                mid = gap_layers[len(gap_layers) // 2]
                cands.append((spread, g, mid))
        return [(g, layer) for _s, g, layer in
                sorted(cands, key=lambda x: -x[0])]

    log(f"phase 2: wave 1 = {len(wave1)} probes (every 4th layer x 3 groups, "
        f"round-robin; {len(qkv_layers)} qkv + 2x{len(exps_layers)} exps "
        f"tensors total), then variance drilling; P2 cap ${P2_CAP_USD:.2f}")

    # Pipeline: up to 2 CPU builds in flight ahead of ONE sequential GPU
    # lane (run_ppl is .remote). Wave-1 first, then adaptive drilling —
    # drill picks lag the in-flight builds by design, that's fine.
    inflight: list = []
    spawned: set = set()
    budget_stopped = False

    def next_unscheduled():
        for g, layer in wave1:
            k = probe_key(g, layer)
            if k not in tests and k not in spawned:
                return g, layer
        for g, layer in drill_candidates():
            if probe_key(g, layer) not in spawned:
                return g, layer
        return None

    def try_spawn() -> str:
        """'spawned' | 'budget' | 'empty'."""
        nonlocal budget_stopped
        driver_usd = RATES["driver"] * (time.time() - t_driver)
        margin = 0.30 * (len(inflight) + 1)
        if (p2_spent() + margin > P2_CAP_USD
                or ledger["total_usd"] + driver_usd + margin > budget_usd):
            if not budget_stopped:
                log(f"P2 BUDGET STOP: p2 ${p2_spent():.2f}/${P2_CAP_USD:.2f}, "
                    f"ledger ${ledger['total_usd']:.2f}/${budget_usd:.2f} "
                    f"({len(inflight)} in flight). Partial map is still data.")
            budget_stopped = True
            return "budget"
        nxt = next_unscheduled()
        if nxt is None:
            return "empty"
        g, layer = nxt
        key = probe_key(g, layer)
        tensor = tensor_of(g, layer)
        rel = f"candidates/probe_{g}_blk{layer}.gguf"
        spawned.add(key)
        try:
            override = build_probe_override(v3_text, tensor, g)
        except ValueError as e:
            tests[key] = {"tensor": tensor, "error": f"override: {e}"}
            _dump_json(probes_path, probes)
            vol.commit()
            log(f"  probe {key} override FAILED: {e}")
            return "spawned"
        inflight.append({
            "key": key, "g": g, "tensor": tensor, "rel": rel,
            "handle": quantize_vol.spawn(rel, BASE_TYPE, override),
            "t0": time.time(),
        })
        log(f"  spawned build {key} ({len(inflight)} in flight)")
        return "spawned"

    while True:
        while not budget_stopped and len(inflight) < 2:
            if try_spawn() != "spawned":
                break
        if not inflight:
            log("phase 2: pipeline empty — done "
                + ("(budget stop)" if budget_stopped else "(no candidates left)"))
            break
        item = inflight.pop(0)
        key, g = item["key"], item["g"]
        try:
            b = item["handle"].get()
            add_cost("quantize", time.time() - item["t0"], f"build probe {key}")
        except Exception as e:
            add_cost("quantize", time.time() - item["t0"],
                     f"build probe {key} FAILED")
            tests[key] = {"tensor": item["tensor"],
                          "error": f"quantize_failed: {e}"}
            _dump_json(probes_path, probes)
            vol.commit()
            log(f"  probe {key} build FAILED: {e}")
            continue
        # keep the build lanes full while the GPU works
        while not budget_stopped and len(inflight) < 2:
            if try_spawn() != "spawned":
                break
        vol.reload()
        try:
            r = run_ppl(item["rel"], b["bytes"], CAL, f"PPL probe {key}")
            ppl = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"] for d in DOMAINS}
            tests[key] = {
                "tensor": item["tensor"],
                "direction": ("demote->" + PROBE_DEMOTE_TYPE if g == "attn_qkv"
                              else "promote->" + PROBE_PROMOTE_TYPE),
                "ppl": ppl,
                "delta": {d: round(ppl[d] - ref_ppl[d], 4) for d in DOMAINS},
                "delta_pct": {d: round(100 * (ppl[d] / ref_ppl[d] - 1), 3)
                              for d in DOMAINS},
                "bytes": b["bytes"],
                "bytes_delta_vs_ref": b["bytes"] - probes["reference_bytes"],
            }
            log(f"  probe {key}: Δ% {tests[key]['delta_pct']} "
                f"(cum ${ledger['total_usd']:.2f})")
        except Exception as e:
            tests[key] = {"tensor": item["tensor"], "error": f"ppl_failed: {e}"}
            log(f"  probe {key} PPL FAILED: {e}")
        _dump_json(probes_path, probes)
        vol.commit()
        rm(item["rel"])

    # ---------------- phase 3: summary + STOP ----------------
    add_cost("driver", time.time() - t_driver, "driver wall time")
    summary = _render_summary(probes, baselines, ledger, qkv_layers, exps_layers)
    with open(f"{RESULTS_DIR}/v4_sensitivity_summary.md", "w") as f:
        f.write(summary)
    _dump_json(probes_path, probes)
    vol.commit()
    log("=== phase 2 complete. STOPPING per the method: v4 allocation + "
        "benchmark gates happen LOCALLY, human-reviewed. ===")
    log(f"total estimated spend: ${ledger['total_usd']:.2f}, "
        f"probes measured: {len([k for k in tests if 'delta_pct' in tests[k]])}")
    print(summary, flush=True)
    return {"ledger_usd": ledger["total_usd"], "probes": len(tests)}


def _render_summary(probes, baselines, ledger, qkv_layers, exps_layers) -> str:
    tests = probes.get("tests", {})
    lines = [
        "# Qwen3.6-35B-A3B v4 — tensor-level sensitivity (3 heavy groups)",
        "",
        f"- model: {probes['model']} (arch {probes['arch']}), "
        f"llama.cpp {probes['llama_cpp']}",
        f"- base: {probes['base']}",
        "- imatrix: the exact shipped-v3 unsloth GGUF imatrix "
        "(1020 tensors / 76 chunks)",
        "- reference PPL: " + ", ".join(
            f"{d}={probes['reference_ppl'][d]:.4f}" for d in DOMAINS),
        f"- estimated Modal spend: ${ledger['total_usd']:.2f}",
        "",
        "## Baselines (kept on volume)",
        "",
        "| build | size GB | " + " | ".join(DOMAINS) + " | wikitext |",
        "|---|---|" + "---|" * (len(DOMAINS) + 1),
    ]
    for name, v in baselines.items():
        if "ppl" not in v:
            continue
        p = v["ppl"]
        lines.append(
            f"| {name} | {v['build']['bytes']/1e9:.2f} | "
            + " | ".join(f"{p[f'cerebellum_calibration_{d}.txt']:.4f}"
                         for d in DOMAINS)
            + f" | {p.get(WIKITEST, float('nan')):.4f} |")
    for g, direction in (("attn_qkv", f"demote to {PROBE_DEMOTE_TYPE} "
                                      f"(+ = damage; small + = demotable)"),
                         ("ffn_down_exps", f"promote to {PROBE_PROMOTE_TYPE} "
                                           f"(- = improvement; strong - = spend here)"),
                         ("ffn_up_exps", f"promote to {PROBE_PROMOTE_TYPE}")):
        layers = qkv_layers if g == "attn_qkv" else exps_layers
        lines += ["", f"## {g} — {direction}", "",
                  "| layer | ΔGB | " + " | ".join(f"Δ%{d}" for d in DOMAINS) + " |",
                  "|---|---|" + "---|" * len(DOMAINS)]
        for layer in layers:
            t = tests.get(f"{g}.blk.{layer}")
            if not t:
                continue
            if "error" in t:
                lines.append(f"| {layer} | - | "
                             + " | ".join("-" for _ in DOMAINS)
                             + f" | ERROR {t['error'][:40]}")
                continue
            lines.append(
                f"| {layer} | {t['bytes_delta_vs_ref']/1e9:+.3f} | "
                + " | ".join(f"{t['delta_pct'][d]:+.3f}" for d in DOMAINS) + " |")
    lines += [
        "",
        f"- probed {len([k for k in tests if 'delta_pct' in tests[k]])} of "
        f"{len(qkv_layers) + 2*len(exps_layers)} heavy-group tensors "
        f"(wave-1 every-4th-layer + variance drilling, budget-capped)",
        "",
        "NEXT (human review required — no-bullshit rule): v4 allocation from "
        "this map (graded overrides, 27B-v4 style), then LOCAL gates "
        "(HumanEval+/ARC/HellaSwag/MMLU + BigCodeBench vs v3 and same-size "
        "uniform baselines) before any ship.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def launch_p0():
    """Phase 0 only (CPU: download + convert + verify) — safe to overlap with
    another campaign's GPU work."""
    fc = campaign_driver.spawn(budget_usd=BUDGET_USD, gpu_phases=False)
    print(f"p0 driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-35b-v4-campaign")


@app.local_entrypoint()
def launch(budget_usd: float = BUDGET_USD):
    """Full campaign (resumes past completed phases via volume checkpoints).
    Only launch once `modal app list` shows the GPU lane is clear."""
    fc = campaign_driver.spawn(budget_usd=budget_usd, gpu_phases=True)
    print(f"campaign driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-35b-v4-campaign")
    print("results:  modal volume ls cerebellum-35b-v4 results")
