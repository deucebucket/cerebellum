"""Cerebellum GLM-4.7-Flash campaign — runs ENTIRELY on Modal.

THE METHOD (OG bench-gated Cerebellum formula — nothing else):
  group ablation (crush one tensor-type GROUP to Q2_K, measure 4-domain PPL
  delta vs the uniform base) -> protect/demote verdicts -> tensor-type
  override file -> stock llama-quantize -> candidate selection gated on
  BENCHMARKS later (human-reviewed; phase 3 of this driver only writes the
  plan and STOPS).  NO hillstep. NO PPL-only ship decisions.

MODEL FACTS (verified 2026-06-12 against the local BF16 conversion at
/var/home/deucebucket/games/cerebellum-glm47-flash/glm-4.7-flash-bf16.gguf):
  - GGUF arch: `deepseek2` (convert_hf_to_gguf maps Glm4MoeLiteForCausalLM
    onto the DeepSeek-V2/MLA graph). block_count=47 (blk.0..blk.46).
  - NO nextn/MTP tensors in the conversion (no blk.47, no *.nextn.*) —
    nothing to exclude, and no MTP-contamination risk (heretic lesson).
  - blk.0 is the single leading dense FFN layer (leading_dense_block_count=1):
    ffn_gate/up/down.weight exist only on blk.0.
  - blk.1..blk.46 are MoE: ffn_{gate,up,down}_exps (64 experts, 4 used),
    ffn_{gate,up,down}_shexp (1 shared expert), ffn_gate_inp (router) and
    exp_probs_b.bias (sigmoid gating bias; expert_weights_norm=true,
    scale=1.8).  Router + gating bias + all norms/biases are EXCLUDED from
    ablation (tiny / known-fragile aux signals).
  - MLA attention per layer: attn_q_a, attn_q_b, attn_kv_a_mqa, attn_k_b,
    attn_v_b, attn_output (47 each).
  - vocab 154880; token_embd and output are separate (untied).
  - 844 tensors total.

PIPELINE (all heavy work in Modal containers; artifacts on the
`cerebellum-flash` Volume; the driver itself runs ON MODAL so it survives
anything local):
  phase 0: download zai-org/GLM-4.7-Flash safetensors inside Modal, run
           convert_hf_to_gguf.py -> BF16 on the Volume; fetch bartowski's
           imatrix; verify calibration corpora (uploaded from local via
           `modal volume put` — they are small text files).
  phase 1: uniform Q8_0 / Q4_K_M / Q3_K_M baselines (CPU quantize from the
           Volume BF16) + PPL on 4 calibration domains + full wiki.test.raw.
           Q8_0 (~33 GB) is deleted from the Volume after its PPL.
  phase 2: for each of 10 tensor groups: build Q4_K_M-base candidate with
           the group forced to Q2_K (imatrix), run 4-domain PPL, DELETE the
           candidate.  CPU build of group N+1/N+2 overlaps GPU PPL of group
           N (2 builds in flight, 1 GPU at a time).  Results JSON mirrors
           scripts/ablate_multidomain.py's schema.
  phase 3: write summary + verdicts + allocation plan, STOP.  No allocation
           candidates, no benchmarks — those happen after human review.

BUDGET DISCIPLINE:
  Every function has an explicit timeout and the cheapest hardware that
  holds the model.  GPU per fit rule: <12.5 GB -> T4, <21 GB -> L4 (24 GB),
  else L40S (48 GB; only the Q8_0 baseline needs it).  The driver keeps a
  cost ledger (wall seconds x list-price rates), persists it to the Volume,
  and ABORTS the campaign if cumulative estimated spend would exceed the
  budget (default $12, hard requirement <= $15 total).  Gate locally before
  launch:  python3 modal_credits.py --gate 13

LAUNCH (detached — survives local death):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  modal volume create cerebellum-flash   # once
  modal volume put cerebellum-flash <corpora...> corpora/   # once, small files
  nohup setsid modal run --detach cerebellum_flash_campaign.py::launch \
      > ../../cerebellum-glm47-flash/logs/modal_launch.log 2>&1 &

MONITOR:
  modal app list                                    # cerebellum-flash-campaign state
  modal app logs cerebellum-flash-campaign          # live driver log
  modal volume ls cerebellum-flash results          # artifacts
  modal volume get cerebellum-flash results/ablation_results_multidomain.json -
  python3 modal_credits.py                          # actual spend
"""

import json
import os
import re
import subprocess
import time

import modal

# Reuse the validated harness's pinned llama.cpp build + images + helpers.
from cerebellum_modal import (
    CUDA_IMAGE_REF,
    LLAMA_CPP_TAG,
    LLAMA_DIR,
    TARBALL_URL,
    _sanitize_override,
)

app = modal.App("cerebellum-flash-campaign")

hf_secret = modal.Secret.from_name("hf-token")
vol = modal.Volume.from_name("cerebellum-flash", create_if_missing=True)
V = "/vol"

HF_MODEL = "zai-org/GLM-4.7-Flash"
IMATRIX_REPO = "bartowski/zai-org_GLM-4.7-Flash-GGUF"
IMATRIX_FILE = "zai-org_GLM-4.7-Flash-imatrix.gguf"

BF16 = f"{V}/glm-4.7-flash-bf16.gguf"
IMATRIX = f"{V}/imatrix/{IMATRIX_FILE}"
CORPORA_DIR = f"{V}/corpora"
RESULTS_DIR = f"{V}/results"
DOMAINS = ["wiki", "code", "math", "dialogue"]
WIKITEST = "wiki.test.raw"

ABLATE_TYPE = "Q2_K"
BASE_TYPE = "Q4_K_M"  # ablation base, same as the 12B campaign

# Tensor groups — verified against the actual converted GGUF (tensor counts
# in comments are exact).  ECMAScript regexes consumed raw by llama-quantize
# --tensor-type-file.  blk.47 does not exist; norms/biases/router/gating-bias
# are never matched by these patterns.
GROUPS = [
    ("routed_exps_gate_up", r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$", 92),
    ("routed_exps_down", r"^blk\.\d+\.ffn_down_exps\.weight$", 46),
    ("shared_expert", r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$", 138),
    ("mla_kv_decompress", r"^blk\.\d+\.attn_[kv]_b\.weight$", 94),
    ("mla_q", r"^blk\.\d+\.attn_q_[ab]\.weight$", 94),
    ("mla_kv_compress", r"^blk\.\d+\.attn_kv_a_mqa\.weight$", 47),
    ("attn_output", r"^blk\.\d+\.attn_output\.weight$", 47),
    ("dense_ffn_l0", r"^blk\.0\.ffn_(gate|up|down)\.weight$", 3),
    ("token_embd", r"^token_embd\.weight$", 1),
    ("output_head", r"^output\.weight$", 1),
]

# ---------------------------------------------------------------------------
# Cost model (list prices, $/sec, all-in per function shape)
# ---------------------------------------------------------------------------
CPU_S = 0.0000131  # per core
MEM_S = 0.00000222  # per GiB
GPU_S = {"T4": 0.000164, "L4": 0.000222, "L40S": 0.000542}

RATES = {
    "convert": 8 * CPU_S + 64 * MEM_S,
    "quantize": 16 * CPU_S + 32 * MEM_S,
    "ppl_t4": GPU_S["T4"] + 4 * CPU_S + 16 * MEM_S,
    "ppl_l4": GPU_S["L4"] + 4 * CPU_S + 24 * MEM_S,
    "ppl_l40s": GPU_S["L40S"] + 4 * CPU_S + 48 * MEM_S,
    "driver": 1 * CPU_S + 2 * MEM_S,
}
# Conservative pre-phase estimates (driver refuses to start a phase that
# would push the ledger past budget).
PHASE_EST = {"phase0": 1.00, "phase1": 4.00, "phase2": 6.50}

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# CPU quantize image: identical strategy to the validated harness (official
# release tarball, pinned tag), rebuilt here so we can attach the Volume and
# local python sources without touching cerebellum_modal's objects.
quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "libgomp1", "libssl3")
    .env({"LD_LIBRARY_PATH": LLAMA_DIR})
    .run_commands(
        f"curl -fsSL -o /tmp/llama.tar.gz {TARBALL_URL}",
        "tar xzf /tmp/llama.tar.gz -C /opt && rm /tmp/llama.tar.gz",
        f"{LLAMA_DIR}/llama-quantize --help 2>&1 | grep -q usage",
    )
    .add_local_python_source("cerebellum_modal", "cerebellum_flash_campaign")
)

# Convert image: llama.cpp source at the same pinned tag (convert script +
# gguf-py) + CPU torch.  Local repo (b8925+28) already converts this model;
# b9603 is newer and includes Glm4MoeLiteForCausalLM.
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
    .add_local_python_source("cerebellum_modal", "cerebellum_flash_campaign")
)

gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])
    .add_local_python_source("cerebellum_modal", "cerebellum_flash_campaign")
)

driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_modal", "cerebellum_flash_campaign"
)


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=convert_image,
    cpu=8.0,
    memory=65536,
    # NOTE: no ephemeral_disk override — Modal's default container disk is
    # 512 GiB (the param only accepts values above that), plenty for the
    # ~63 GB safetensors download.
    timeout=7200,
    secrets=[hf_secret],
    volumes={V: vol},
)
def convert_and_fetch() -> dict:
    """Download GLM-4.7-Flash safetensors inside Modal, convert to BF16 GGUF
    directly onto the Volume, and fetch bartowski's imatrix.  Idempotent."""
    from huggingface_hub import hf_hub_download, snapshot_download

    t0 = time.time()
    out = {}

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
    image=quant_image,
    cpu=16.0,
    memory=32768,
    timeout=5400,
    volumes={V: vol},
)
def quantize_vol(out_rel: str, base_type: str, override_text: str = "") -> dict:
    """llama-quantize Volume BF16 -> Volume GGUF with optional tensor-type
    override file (group ablation).  Skips if output already exists (resume)."""
    out_path = f"{V}/{out_rel}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e9:
        print(f"exists, skipping: {out_path}")
        return {"out": out_rel, "bytes": os.path.getsize(out_path), "skipped": True}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    cmd = [f"{LLAMA_DIR}/llama-quantize", "--imatrix", IMATRIX]
    clean = _sanitize_override(override_text)
    if clean:
        ovr = "/tmp/tensor_types.txt"
        with open(ovr, "w") as f:
            f.write(clean + "\n")
        print(f"override:\n{clean}", flush=True)
        cmd += ["--tensor-type-file", ovr]
    cmd += [BF16, tmp, base_type, "16"]

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

    # detect k-quant fallbacks (e.g. attn_k_b rows not 256-divisible)
    fallbacks = sorted({ln.strip() for ln in output.splitlines()
                        if "fall" in ln.lower() or "requantiz" in ln.lower()})[:20]
    return {
        "out": out_rel,
        "bytes": os.path.getsize(out_path),
        "secs": round(time.time() - t0, 1),
        "fallback_lines": fallbacks,
        "converting_counts": _count_conversions(output),
    }


def _count_conversions(output: str) -> dict:
    """Histogram of 'converting to <type>' lines from llama-quantize output —
    the proof that a group override matched (the 12B no-op lesson)."""
    counts: dict = {}
    for m in re.finditer(r"converting to (\w+)", output):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def _ppl_impl(filename: str, corpora: list[str], ngl: int) -> dict:
    binary = None
    for cand in ("/app/llama-perplexity", "/llama-perplexity",
                 "/usr/local/bin/llama-perplexity"):
        if os.path.exists(cand):
            binary = cand
            break
    if binary is None:
        found = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "llama-perplexity", "-type", "f"],
            capture_output=True, text=True).stdout.split()
        if not found:
            raise RuntimeError("llama-perplexity not found in CUDA image")
        binary = found[0]

    model = f"{V}/{filename}"
    if not os.path.exists(model):
        raise RuntimeError(f"model not on volume: {model}")
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
    return {"ppl": results, "secs": timings, "llama_cpp_tag": LLAMA_CPP_TAG}


@app.function(image=gpu_image, gpu="T4", cpu=4.0, memory=16384, timeout=3600, volumes={V: vol})
def ppl_t4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=gpu_image, gpu="L4", cpu=4.0, memory=24576, timeout=3600, volumes={V: vol})
def ppl_l4(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


@app.function(image=gpu_image, gpu="L40S", cpu=4.0, memory=49152, timeout=5400, volumes={V: vol})
def ppl_l40s(filename: str, corpora: list[str], ngl: int = 99) -> dict:
    return _ppl_impl(filename, corpora, ngl)


# ---------------------------------------------------------------------------
# Driver (runs ON MODAL — survives anything local)
# ---------------------------------------------------------------------------


@app.function(
    image=driver_image,
    cpu=1.0,
    memory=2048,
    timeout=16 * 3600,
    volumes={V: vol},
)
def campaign_driver(budget_usd: float = 12.0) -> dict:
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
        # driver's own burn so far
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

    def pick_ppl(nbytes: int):
        if nbytes < 12.5e9:
            return ppl_t4, "ppl_t4"
        if nbytes < 21.0e9:
            return ppl_l4, "ppl_l4"
        return ppl_l40s, "ppl_l40s"

    def run_ppl(filename: str, nbytes: int, corpora: list[str], note: str) -> dict:
        fn, kind = pick_ppl(nbytes)
        t0 = time.time()
        try:
            r = fn.remote(filename, corpora)
        except Exception as e:
            add_cost(kind, time.time() - t0, note + " (failed)")
            log(f"  {note}: {kind} failed ({e}); retrying on L40S")
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

    log(f"=== Cerebellum GLM-4.7-Flash campaign driver (budget ${budget_usd:.2f}) ===")
    log(f"groups: {[g[0] for g in GROUPS]}")

    # ---------------- phase 0: convert + imatrix + corpora ----------------
    gate("phase0")
    vol.reload()
    missing = [c for c in [f"cerebellum_calibration_{d}.txt" for d in DOMAINS] + [WIKITEST]
               if not os.path.exists(f"{CORPORA_DIR}/{c}")]
    if missing:
        log(f"FATAL: corpora missing from volume: {missing} — "
            f"upload with `modal volume put cerebellum-flash <file> corpora/<name>`")
        raise SystemExit("missing corpora")
    if not (os.path.exists(BF16) and os.path.getsize(BF16) > 55e9):
        log("phase 0: converting on Modal (download ~63 GB + convert_hf_to_gguf)")
        r = timed("convert", convert_and_fetch, note="convert+imatrix")
        log(f"phase 0 done: {r}")
    else:
        log("phase 0: BF16 already on volume, skipping convert")
        if not os.path.exists(IMATRIX):
            timed("convert", convert_and_fetch, note="imatrix fetch")
    vol.reload()

    # ---------------- phase 1: uniform baselines ----------------
    gate("phase1")
    baselines_path = f"{RESULTS_DIR}/baselines.json"
    baselines = _load_json(baselines_path, {})
    base_specs = [("Q8_0", "baselines/glm47-flash-Q8_0.gguf"),
                  ("Q4_K_M", "baselines/glm47-flash-Q4_K_M.gguf"),
                  ("Q3_K_M", "baselines/glm47-flash-Q3_K_M.gguf")]
    # build (parallel CPU containers), but only those without PPL yet
    handles = {}
    for qtype, rel in base_specs:
        if qtype in baselines and "ppl" in baselines[qtype]:
            continue
        t0 = time.time()
        handles[qtype] = (quantize_vol.spawn(rel, qtype), t0, rel)
    for qtype, (h, t0, rel) in handles.items():
        b = h.get()
        add_cost("quantize", time.time() - t0, f"build {qtype}")
        baselines.setdefault(qtype, {})["build"] = b
        _dump_json(baselines_path, baselines)
        vol.commit()
        log(f"  built {qtype}: {b['bytes']/1e9:.2f} GB")
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
            rm(rel)  # 33 GB — reference only, free the storage
    log(f"phase 1 done: {json.dumps({k: v.get('ppl') for k, v in baselines.items()})}")

    # ---------------- phase 2: group ablation ----------------
    gate("phase2")
    abl_path = f"{RESULTS_DIR}/ablation_results_multidomain.json"
    abl = _load_json(abl_path, {})
    abl.setdefault("model", HF_MODEL)
    abl.setdefault("arch", "deepseek2 (glm4_moe_lite)")
    abl.setdefault("base_type", BASE_TYPE)
    abl.setdefault("ablate_type", ABLATE_TYPE)
    abl.setdefault("llama_cpp_tag", LLAMA_CPP_TAG)
    abl["baseline_ppl"] = {d: baselines[BASE_TYPE]["ppl"][f"cerebellum_calibration_{d}.txt"]
                           for d in DOMAINS}
    abl["baseline_bytes"] = baselines[BASE_TYPE]["build"]["bytes"]
    abl.setdefault("corpus_versions",
                   {d: f"volume:corpora/cerebellum_calibration_{d}.txt" for d in DOMAINS})
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
    for i, (name, regex, cnt) in enumerate(pending):
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
        spawn_build(i + 2)  # keep 2 builds in flight while GPU works

        # no-op guard (the 12B lesson): the override must have converted
        # tensors to the ablate type
        conv = b.get("converting_counts", {})
        crushed = sum(v for k, v in conv.items()
                      if k.lower() == ABLATE_TYPE.lower())
        if not b.get("skipped") and crushed == 0:
            log(f"  WARNING {name}: no tensors converted to {ABLATE_TYPE} — "
                f"override may not have matched! conv={conv}")
        vol.reload()
        try:
            r = run_ppl(rel, b["bytes"], cal_corpora, f"PPL ablate {name}")
            ppl = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"] for d in DOMAINS}
            tests[name] = {
                "gguf_tensor": regex,
                "tensor_count": cnt,
                "ppl": ppl,
                "delta": {d: round(ppl[d] - abl["baseline_ppl"][d], 4) for d in DOMAINS},
                "delta_pct": {d: round(100 * (ppl[d] / abl["baseline_ppl"][d] - 1), 2)
                              for d in DOMAINS},
                "candidate_bytes": b["bytes"],
                "bytes_saved_vs_base": abl["baseline_bytes"] - b["bytes"],
                "q2k_converted": crushed,
                "fallback_lines": b.get("fallback_lines", []),
            }
        except Exception as e:
            log(f"  PPL {name} FAILED: {e}")
            tests[name] = {"gguf_tensor": regex, "tensor_count": cnt,
                           "error": f"ppl_failed: {e}"}
        _dump_json(abl_path, abl)
        vol.commit()
        rm(rel)
        log(f"[{i+1}/{len(pending)}] {name} done: "
            f"{tests[name].get('delta_pct', tests[name].get('error'))}")

    # ---------------- phase 3: summary + STOP ----------------
    add_cost("driver", time.time() - t_driver, "driver wall time")
    summary = _render_summary(abl, baselines, ledger)
    with open(f"{RESULTS_DIR}/ablation_summary.md", "w") as f:
        f.write(summary)
    _dump_json(abl_path, abl)
    vol.commit()
    log("=== phase 2 complete. STOPPING per the method: allocation candidates and "
        "benchmarks require human review (see RUN_PLAN.md in the repo). ===")
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
        "# GLM-4.7-Flash group ablation — multi-domain summary",
        "",
        f"- model: {abl['model']}  (arch {abl['arch']})",
        f"- base {abl['base_type']} + group->{abl['ablate_type']}, "
        f"imatrix: bartowski, llama.cpp {abl['llama_cpp_tag']}",
        f"- baseline PPL ({abl['base_type']}): "
        + ", ".join(f"{d}={abl['baseline_ppl'][d]:.4f}" for d in DOMAINS),
        f"- estimated Modal spend: ${ledger['total_usd']:.2f}",
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
            lines.append(f"| {name} | - | - | " + " | ".join("-" for _ in DOMAINS) + " | MISSING |")
            continue
        if "error" in t:
            lines.append(f"| {name} | {t['tensor_count']} | - | "
                         + " | ".join("-" for _ in DOMAINS) + f" | ERROR: {t['error'][:40]} |")
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
        "else DEMOTABLE (incl. negative = regularizing).",
        "",
        "NEXT (human review required — no-bullshit rule): build allocation "
        "candidates from these verdicts, then gate on HumanEval+/ARC/HellaSwag/"
        "MMLU + BigCodeBench vs same-size uniform baselines. See "
        "cerebellum-glm47-flash/RUN_PLAN.md.",
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
def launch(budget_usd: float = 12.0):
    fc = campaign_driver.spawn(budget_usd=budget_usd)
    print(f"campaign driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-flash-campaign")
    print("results:  modal volume ls cerebellum-flash results")
