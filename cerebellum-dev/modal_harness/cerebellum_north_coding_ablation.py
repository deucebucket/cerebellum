"""Cerebellum North-Mini-Code-1.0 CODING ABLATION — runs ENTIRELY on Modal.

THE LOST PHASE (reconstructed 2026-06-13). The overnight campaign
(cerebellum_north_campaign.py) measured 4-domain PPL only and called North a
"no-ship". That verdict is INVALID for an MoE CODE model: PPL on code TEXT does
not see coding collapse. On Qwen3.6-27B, demoting attn_qkv→Q2_K moved PPL <1%
but dropped HumanEval pass@1 by 46 points. This driver restores phase 5/6 of the
REAL method (knowledge/REAL_PIPELINE_RECONSTRUCTED.md), measuring per-group
HumanEval pass@1 deltas directly — the thing PPL could not measure.

It mirrors scripts/coding_ablation.py Phase G (per-group), adapted to Modal:
  - North runs ONLY on llama.cpp PR #24260 binaries with the cohere2moe vocab
    fix — those live only in the campaign image, so this MUST run on Modal too.
  - The campaign image lacked llama-server; this image adds a cached server
    build layer (same PR commit + same one-line vocab patch).
  - human_eval EXECUTES untrusted generated code — Modal's disposable sandboxed
    containers are the correct, safe place to run it.

PIPELINE (artifacts on the shared `cerebellum-north` Volume — BF16 + unsloth
imatrix + Q4_K_M baseline already there from the campaign):
  baseline: build uniform Q4_K_M (skipped — exists), serve it, run HumanEval.
            THIS pass@1 is the reference all group deltas measure against.
  per group: rebuild a candidate from BF16 that demotes ONLY that group to Q2_K
            (imatrix), serve it, run HumanEval, record pass@1 delta vs baseline,
            classify CRITICAL/MEDIUM/DISPOSABLE. CPU build of candidate N+1
            pipelines while the GPU serves+evals candidate N. Candidate GGUF
            deleted after eval (KEEP only the baseline + least-damaging).
  summary:  results/coding_ablation/coding_ablation_summary.json + .md, STOP.
            Allocation + publication gates happen separately (local + gated).

HumanEval harness (verbatim convention from the recovered runner via
scripts/coding_ablation.py): /v1/completions, temp 0, max_tokens 512, parallel 4,
164 problems, OpenAI human_eval evaluate_functional_correctness, normalize_indent
+ fence strip. Override convention: blk.N.<group>.weight=q2_K per layer (output
tied → no output group; token_embd is the single token_embd.weight line).

BUDGET: retries=0, cost ledger on the Volume, one GPU at a time, watchdog
$4/hr + $28.50 doors-close enforced locally. Gate before launch:
  python3 modal_credits.py --gate 5

LAUNCH (detached — driver runs ON Modal, survives local death):
  cd cerebellum-dev/modal_harness
  nohup setsid modal run --detach \
      cerebellum_north_coding_ablation.py::launch \
      > ../../cerebellum-north-mini-code/logs/coding_ablation_launch.log 2>&1 &

MONITOR:
  modal app logs cerebellum-north-coding-ablation
  modal volume ls cerebellum-north results/coding_ablation
"""

import json
import os
import re
import subprocess
import time

import modal

app = modal.App("cerebellum-north-coding-ablation")

vol = modal.Volume.from_name("cerebellum-north", create_if_missing=True)
V = "/vol"

LLAMA_PR = "24260"
LLAMA_PR_COMMIT = "d9320477de5549e53a9452296f468d32a1d81d26"
BUILD_DIR = "/opt/llama.cpp/build/bin"          # CUDA: llama-server, llama-perplexity
CPU_BUILD_DIR = "/opt/llama.cpp/build-cpu/bin"  # CPU: llama-quantize, llama-cli

BF16 = f"{V}/north-mini-code-bf16.gguf"
IMATRIX = f"{V}/imatrix/imatrix_unsloth.gguf_file"
RESULTS_DIR = f"{V}/results/coding_ablation"
CAND_DIR = f"{V}/candidates_coding"

BASE_TYPE = "Q4_K_M"
ABLATE_TYPE = "Q2_K"
BASE_REL = "baselines/north-mini-code-Q4_K_M.gguf"   # already on volume

# HumanEval settings — verbatim from the artifacts (coding_ablation.log).
BENCH_PORT = 8084
PARALLEL = 4
MAX_TOKENS = 512
TEMPERATURE = 0.0
SERVER_CTX = 8192          # serving headroom for 4 slots * 512-tok completions
N_LAYERS = 49             # blk.0..blk.48 (measured)

# Groups — the 8 MEASURED North groups (cerebellum_north_campaign.py GROUPS).
# token_embd is TIED (also the LM head). `output` group does not exist (tied).
GROUPS = [
    ("routed_exps_gate_up", r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$"),
    ("routed_exps_down", r"^blk\.\d+\.ffn_down_exps\.weight$"),
    ("attn_q", r"^blk\.\d+\.attn_q\.weight$"),
    ("attn_k", r"^blk\.\d+\.attn_k\.weight$"),
    ("attn_v", r"^blk\.\d+\.attn_v\.weight$"),
    ("attn_output", r"^blk\.\d+\.attn_output\.weight$"),
    ("dense_ffn_l0", r"^blk\.0\.ffn_(gate|up|down)\.weight$"),
    ("token_embd", r"^token_embd\.weight$"),
]

# Cost model (list prices, $/sec) — mirrors the campaign harness.
CPU_S = 0.0000131
MEM_S = 0.00000222
GPU_S = {"L4": 0.000222, "L40S": 0.000542}
RATES = {
    "quantize": 12 * CPU_S + 32 * MEM_S,
    "eval_l4": GPU_S["L4"] + 8 * CPU_S + 24 * MEM_S,
    "eval_l40s": GPU_S["L40S"] + 8 * CPU_S + 49 * MEM_S,
    "driver": 1 * CPU_S + 2 * MEM_S,
}

# ---------------------------------------------------------------------------
# Image — campaign CUDA image + llama-server + the HumanEval harness deps.
# Reuses the EXACT PR commit + vocab patch; adds a cached llama-server layer.
# ---------------------------------------------------------------------------
llama_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "curl", "ca-certificates")
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        f"cd /opt/llama.cpp && git fetch origin pull/{LLAMA_PR}/head:pr && "
        f"(git checkout {LLAMA_PR_COMMIT} || git checkout pr) && "
        "git rev-parse HEAD | tee /opt/LLAMA_SHA",
    )
    .run_commands(
        # CPU tool build: quantize/cli (no CUDA link — runs on GPU-less builders)
        "cd /opt/llama.cpp && cmake -B build-cpu "
        "-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF "
        "-DCMAKE_BUILD_TYPE=Release && "
        "cmake --build build-cpu -j 8 --target llama-quantize llama-cli",
        f"{CPU_BUILD_DIR}/llama-quantize --help 2>&1 | grep -q tensor-type-file",
    )
    .run_commands(
        # CUDA build config (server + perplexity). Stubs at end of link line.
        "cd /opt/llama.cpp && cmake -B build "
        "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='89' "
        "-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_STANDARD_LIBRARIES='-L/usr/local/cuda/lib64/stubs -lcuda'",
    )
    .run_commands(
        "cd /opt/llama.cpp && cmake --build build -j 8 --target ggml",
    )
    .run_commands(
        # VOCAB FIX (campaign root-cause): accept cohere2moe pre-tokenizer as
        # tiny_aya, then build the server. Same one-line sed as the campaign.
        'cd /opt/llama.cpp && '
        'sed -i \'s@tokenizer_pre == "tiny_aya")@tokenizer_pre == "tiny_aya" || tokenizer_pre == "cohere2moe")@\' '
        'src/llama-vocab.cpp && grep -q \'tokenizer_pre == "cohere2moe"\' src/llama-vocab.cpp',
        "cd /opt/llama.cpp && cmake --build build -j 8 --target llama-server",
        "cd /opt/llama.cpp && cmake --build build-cpu -j 8 --target llama-cli",
        f"test -x {BUILD_DIR}/llama-server",
    )
    .pip_install("httpx", "human-eval", "numpy")
    .run_commands(
        # human_eval ships with execution disabled behind a comment guard; the
        # canonical harness un-comments exec(). Modal containers are sandboxed
        # & disposable, so executing generated code here is the safe path.
        # Robust to either commented or already-uncommented states.
        "python -c \"import human_eval.execution as e, re, pathlib; "
        "p=pathlib.Path(e.__file__); s=p.read_text(); "
        "s=re.sub(r'#\\\\s*(exec\\\\(check_program, exec_globals\\\\))', r'\\\\1', s); "
        "p.write_text(s); "
        "assert re.search(r'^\\\\s+exec\\\\(check_program', s, re.M), 'exec guard not enabled'\"",
    )
    .env({"PYTHONPATH": "/opt/llama.cpp/gguf-py"})
    .add_local_python_source("cerebellum_north_coding_ablation")
)

driver_image = (
    modal.Image.debian_slim(python_version="3.11")
    # numpy as a deserialization backstop: human_eval returns numpy scalars; the
    # eval fns now cast to pure Python, but ship numpy here too so any numpy
    # type in a payload can still be unpickled on the driver side.
    .pip_install("numpy")
    .add_local_python_source("cerebellum_north_coding_ablation")
)


def _llama_sha() -> str:
    try:
        with open("/opt/LLAMA_SHA") as f:
            return f.read().strip()
    except Exception:
        return "unknown"


def _wait_on_volume(path: str, attempts: int = 6) -> bool:
    for i in range(attempts):
        vol.reload()
        if os.path.exists(path):
            return True
        wait = 10 * (2 ** min(i, 3))
        print(f"  not visible yet ({path}), backoff {wait}s [{i+1}/{attempts}]", flush=True)
        time.sleep(wait)
    return os.path.exists(path)


# ===========================================================================
# HumanEval harness (carried from scripts/coding_ablation.py — same convention)
# ===========================================================================

def normalize_indent(completion: str) -> str:
    lines = completion.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            start = i + 1
            if start < len(lines):
                doc_line = lines[start].lstrip()
                if doc_line.startswith('"""') or doc_line.startswith("'''"):
                    quote = doc_line[:3]
                    if doc_line.count(quote) >= 2:
                        start += 1
                    else:
                        for j in range(start + 1, len(lines)):
                            if quote in lines[j]:
                                start = j + 1
                                break
            break
    body_lines = lines[start:]
    if not body_lines:
        return completion
    min_indent = None
    for line in body_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if min_indent is None or indent < min_indent:
            min_indent = indent
    if min_indent is None:
        return completion
    result = []
    for line in body_lines:
        if not line.strip():
            result.append("")
        else:
            result.append("    " + line[min_indent:])
    return "\n".join(result)


def _strip_fences(content: str) -> str:
    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content


def _run_humaneval_inproc(model_name: str) -> dict:
    """Serve the model already on BENCH_PORT, generate completions in parallel
    via /v1/completions, then execute with human_eval. Returns pass@1 + counts."""
    import httpx
    from concurrent.futures import ThreadPoolExecutor
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    problems = read_problems()
    api = f"http://127.0.0.1:{BENCH_PORT}/v1/completions"
    print(f"HumanEval: {len(problems)} problems  API={api}", flush=True)
    print(f"Settings: temp={int(TEMPERATURE)}, max_tokens={MAX_TOKENS}, "
          f"parallel={PARALLEL}  [harness=human_eval]", flush=True)

    def _completion_for(prompt: str) -> str:
        resp = httpx.post(api, json={
            "prompt": prompt, "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=240.0)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"]
        return normalize_indent(_strip_fences(text))

    def _work(item):
        task_id, problem = item
        try:
            return task_id, _completion_for(problem["prompt"]), None
        except Exception as e:  # noqa: BLE001
            return task_id, "", e

    samples = []
    start = time.time()
    completed = errors = 0
    items = list(problems.items())
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        for task_id, completion, err in ex.map(_work, items):
            samples.append({"task_id": task_id, "completion": completion})
            if err is None:
                completed += 1
            else:
                errors += 1
                print(f"  ERROR {task_id}: {err}", flush=True)
            tot = completed + errors
            if tot % 20 == 0:
                el = time.time() - start
                print(f"  [{tot}/{len(problems)}] {tot/el*60:.0f}/min", flush=True)
    print(f"Generation: {completed} ok, {errors} errors, "
          f"{time.time()-start:.0f}s", flush=True)

    samples_path = "/tmp/samples.jsonl"
    write_jsonl(samples_path, samples)
    print("Running evaluation (executing code)...", flush=True)
    res = evaluate_functional_correctness(samples_path, k=[1], n_workers=4, timeout=10.0)
    # CAST to a plain Python float: human_eval returns a numpy float64 for
    # pass@1, which the driver image (no numpy) cannot deserialize across the
    # Modal boundary. Likewise DO NOT return the samples list — keep the payload
    # small and pure-Python. Persist samples to the Volume for the wrong-answer
    # audit instead (the audit mandate: inspect completions before any score).
    p1 = float(res["pass@1"])
    print(f"\n{'='*50}\nHumanEval pass@1: {p1*100:.1f}%\n{'='*50}", flush=True)
    audit_dir = f"{RESULTS_DIR}/samples"
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = f"{audit_dir}/{model_name}_humaneval_samples.jsonl"
    try:
        write_jsonl(audit_path, samples)
        # the *_results.jsonl that evaluate_functional_correctness writes carries
        # per-problem passed/failed — keep it for the audit too.
        rj = samples_path + "_results.jsonl"
        if os.path.exists(rj):
            import shutil as _sh
            _sh.copy(rj, f"{audit_dir}/{model_name}_humaneval_results.jsonl")
        vol.commit()
    except Exception as e:  # noqa: BLE001 — audit persistence is best-effort
        print(f"  (audit persist warning: {e})", flush=True)
    return {"pass_at_1": p1, "pass_at_1_pct": round(p1 * 100, 1),
            "total": int(len(problems)), "gen_ok": int(completed),
            "gen_err": int(errors), "model": model_name}


def _serve_and_eval(gguf_path: str, model_name: str) -> dict:
    """Launch llama-server on the GGUF, wait for /health, run HumanEval, stop."""
    import httpx

    if not _wait_on_volume(gguf_path):
        raise RuntimeError(f"model not on volume: {gguf_path}")
    log = "/tmp/server.log"
    cmd = [f"{BUILD_DIR}/llama-server", "--model", gguf_path, "-ngl", "99",
           "--parallel", str(PARALLEL), "-c", str(SERVER_CTX),
           "--host", "127.0.0.1", "--port", str(BENCH_PORT),
           "--reasoning-budget", "0"]
    fh = open(log, "w")
    proc = subprocess.Popen(cmd + ["--reasoning", "off"], stdout=fh, stderr=fh)
    deadline = time.time() + 600
    up = False
    while time.time() < deadline:
        if proc.poll() is not None:  # retry once without --reasoning off
            fh.close()
            fh = open(log, "w")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
            deadline = time.time() + 600
        try:
            r = httpx.get(f"http://127.0.0.1:{BENCH_PORT}/health", timeout=5.0)
            if r.status_code == 200:
                up = True
                print(f"  [EVAL] Server up (PID {proc.pid}), running HumanEval...", flush=True)
                break
        except Exception:
            pass
        time.sleep(3)
    if not up:
        try:
            tail = "\n".join(open(log).read().splitlines()[-30:])
        except Exception:
            tail = "(no log)"
        proc.kill()
        raise RuntimeError(f"server failed to come up for {model_name}:\n{tail}")
    try:
        out = _run_humaneval_inproc(model_name)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    out["llama_sha"] = _llama_sha()
    return out


# ===========================================================================
# Remote functions
# ===========================================================================

def _sanitize_override(text: str) -> str:
    return "\n".join(ln for ln in (text or "").splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))


@app.function(image=llama_image, cpu=12.0, memory=32768, timeout=5400,
              retries=0, volumes={V: vol})
def quantize_vol(out_rel: str, base_type: str, override_text: str = "") -> dict:
    """llama-quantize Volume BF16 -> Volume GGUF with optional group override."""
    out_path = f"{V}/{out_rel}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e9:
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
    if proc.returncode != 0 or not os.path.exists(tmp):
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(f"quantize {out_rel} failed rc={proc.returncode}:\n{tail}")
    os.replace(tmp, out_path)
    vol.commit()
    counts: dict = {}
    for m in re.finditer(r"converting to (\w+)", output):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return {"out": out_rel, "bytes": os.path.getsize(out_path),
            "secs": round(time.time() - t0, 1), "converting_counts": counts,
            "llama_sha": _llama_sha()}


@app.function(image=llama_image, gpu="L4", cpu=8.0, memory=24576,
              timeout=3600, retries=0, volumes={V: vol})
def eval_l4(gguf_rel: str, model_name: str) -> dict:
    return _serve_and_eval(f"{V}/{gguf_rel}", model_name)


@app.function(image=llama_image, gpu="L40S", cpu=8.0, memory=50176,
              timeout=5400, retries=0, volumes={V: vol})
def eval_l40s(gguf_rel: str, model_name: str) -> dict:
    return _serve_and_eval(f"{V}/{gguf_rel}", model_name)


# ===========================================================================
# Driver (runs ON MODAL)
# ===========================================================================

@app.function(image=driver_image, cpu=1.0, memory=2048, timeout=int(8.5 * 3600),
              retries=0, volumes={V: vol})
def coding_ablation_driver(budget_usd: float = 6.0) -> dict:
    t_driver = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ledger_path = f"{RESULTS_DIR}/cost_ledger.json"
    ledger = _load_json(ledger_path, {"entries": [], "total_usd": 0.0})
    results_path = f"{RESULTS_DIR}/coding_ablation_results.json"
    res = _load_json(results_path, {"baseline_pct": None, "groups": {}})

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

    def projected_stop(label: str, est: float) -> bool:
        driver_usd = RATES["driver"] * (time.time() - t_driver)
        if ledger["total_usd"] + driver_usd + est > budget_usd:
            log(f"BUDGET STOP before {label}: ledger ${ledger['total_usd']:.2f} "
                f"+ driver ${driver_usd:.2f} + est ${est:.2f} > ${budget_usd:.2f}. "
                f"Partial map is still data.")
            return True
        return False

    # eval cost ~ L4 ~$0.10-0.15/candidate, L40S ~$0.25; build ~$0.10. Use
    # conservative per-candidate estimate for the gate.
    PER_CAND_EST = 0.40

    def pick_eval(nbytes: int):
        # serving a model needs weights + 4-slot ctx activations in VRAM. L4 is
        # 24 GB; the 18.6 GB base leaves ~5 GB for ctx — tight with parallel 4,
        # route base & near-base to L40S. Demoted-expert candidates (~14-15 GB)
        # fit L4 comfortably.
        if nbytes > 16.0e9:
            return eval_l40s, "eval_l40s"
        return eval_l4, "eval_l4"

    def run_eval(gguf_rel: str, nbytes: int, model_name: str) -> dict:
        fn, kind = pick_eval(nbytes)
        t0 = time.time()
        try:
            r = fn.remote(gguf_rel, model_name)
        except Exception as e:
            add_cost(kind, time.time() - t0, f"{model_name} (failed)")
            log(f"  {model_name}: {kind} failed ({e}); retry on L40S")
            t0 = time.time()
            r = eval_l40s.remote(gguf_rel, model_name)
            kind = "eval_l40s"
        add_cost(kind, time.time() - t0, model_name)
        return r

    def rm(rel: str):
        vol.reload()
        p = f"{V}/{rel}"
        if os.path.exists(p):
            os.remove(p)
            vol.commit()
            log(f"  deleted {rel}")

    log(f"=== North-Mini-Code CODING ABLATION driver (budget ${budget_usd:.2f}) ===")
    log(f"groups: {[g[0] for g in GROUPS]}")

    # -------- baseline: serve the uniform Q4_K_M, measure its HumanEval --------
    if not _wait_on_volume(BF16):
        log("FATAL: BF16 not on volume")
        raise SystemExit("no BF16")
    if not _wait_on_volume(IMATRIX):
        log("FATAL: imatrix not on volume")
        raise SystemExit("no imatrix")
    if res.get("baseline_pct") is None:
        if projected_stop("baseline eval", PER_CAND_EST):
            raise SystemExit("budget stop at baseline")
        if not _wait_on_volume(f"{V}/{BASE_REL}"):
            # rebuild the Q4_K_M base if it's gone
            log("baseline Q4_K_M not on volume — rebuilding")
            t0 = time.time()
            b = quantize_vol.remote(BASE_REL, BASE_TYPE)
            add_cost("quantize", time.time() - t0, "build baseline Q4_K_M")
        vol.reload()
        nbytes = os.path.getsize(f"{V}/{BASE_REL}")
        log(f"baseline: serving uniform Q4_K_M ({nbytes/1e9:.2f} GB), running HumanEval")
        r = run_eval(BASE_REL, nbytes, "baseline_Q4_K_M")
        res["baseline_pct"] = r["pass_at_1_pct"]
        res["baseline_detail"] = {k: r[k] for k in
                                  ("pass_at_1_pct", "total", "gen_ok", "gen_err", "llama_sha")}
        _dump_json(f"{RESULTS_DIR}/baseline_humaneval.json",
                   {k: r[k] for k in ("pass_at_1", "pass_at_1_pct", "total",
                                      "gen_ok", "gen_err", "model", "llama_sha")})
        _dump_json(results_path, res)
        vol.commit()
        log(f"*** BASELINE HumanEval pass@1 = {res['baseline_pct']:.1f}% ***")
    else:
        log(f"baseline already measured: {res['baseline_pct']:.1f}%")

    baseline = res["baseline_pct"]

    # -------- per-group coding ablation (pipelined CPU build / GPU eval) --------
    pending = [(n, rx) for (n, rx) in GROUPS if n not in res["groups"]]
    log(f"per-group: {len(pending)} pending of {len(GROUPS)}")
    build_handles: dict = {}

    def spawn_build(i: int):
        if i >= len(pending):
            return
        name, regex = pending[i]
        rel = f"{CAND_DIR.split(V+'/')[1]}/ablate_{name}.gguf"
        override = f"{regex}={ABLATE_TYPE}"
        build_handles[i] = (quantize_vol.spawn(rel, BASE_TYPE, override),
                            time.time(), rel, name, regex)
        log(f"  spawned build [{i+1}/{len(pending)}] {name}")

    spawn_build(0)
    spawn_build(1)

    for i, (name, regex) in enumerate(pending):
        if projected_stop(f"group {name}", PER_CAND_EST):
            break
        h, t0, rel, _, _ = build_handles.pop(i)
        try:
            b = h.get()
        except Exception as e:
            add_cost("quantize", time.time() - t0, f"build {name} (failed)")
            log(f"  build {name} FAILED: {e}")
            res["groups"][name] = {"error": f"quantize_failed: {e}"}
            _dump_json(results_path, res)
            vol.commit()
            spawn_build(i + 2)
            continue
        add_cost("quantize", time.time() - t0, f"build {name}")
        spawn_build(i + 2)   # keep next build in flight while GPU evals

        conv = b.get("converting_counts", {})
        crushed = sum(v for k, v in conv.items() if k.lower() == ABLATE_TYPE.lower())
        if not b.get("skipped") and crushed == 0:
            log(f"  WARNING {name}: 0 tensors converted to {ABLATE_TYPE} — "
                f"override may not have matched! conv={conv}")
        vol.reload()
        try:
            r = run_eval(rel, b["bytes"], f"ablate_{name}")
            score = r["pass_at_1_pct"]
            delta = round(score - baseline, 1)
            res["groups"][name] = {
                "gguf_tensor": regex, "q2k_converted": crushed,
                "candidate_bytes": b["bytes"],
                "pass_at_1_pct": score, "delta_pct": delta,
                "gen_ok": r["gen_ok"], "gen_err": r["gen_err"],
            }
            log(f"*** [{i+1}/{len(pending)}] {name}->Q2_K: {score:.1f}% "
                f"({delta:+.1f}%) ***")
        except Exception as e:
            log(f"  eval {name} FAILED: {e}")
            res["groups"][name] = {"gguf_tensor": regex, "q2k_converted": crushed,
                                   "error": f"eval_failed: {e}"}
        _dump_json(results_path, res)
        vol.commit()
        rm(rel)   # disk hygiene — baseline already measured, no KEEP needed

    # -------- summary + STOP --------
    add_cost("driver", time.time() - t_driver, "driver wall time")
    _write_summary(res, ledger)
    _dump_json(results_path, res)
    vol.commit()
    log("=== coding ablation complete. STOPPING. Allocation + publication gates "
        "happen separately (local/gated). ===")
    log(f"total estimated spend: ${ledger['total_usd']:.2f}")
    return {"ledger_usd": ledger["total_usd"], "baseline_pct": baseline,
            "groups_measured": len([g for g in res["groups"].values()
                                    if "pass_at_1_pct" in g])}


def _classify(delta_pct: float) -> str:
    d = abs(delta_pct)
    if d < 5.0:
        return "DISPOSABLE"
    if d < 30.0:
        return "MEDIUM"
    return "CRITICAL"


def _write_summary(res: dict, ledger: dict):
    baseline = res.get("baseline_pct")
    lines = [
        "# North-Mini-Code-1.0 CODING ABLATION — per-group HumanEval summary",
        "",
        "The REAL method for an MoE code model: demote each tensor GROUP to Q2_K "
        "over the uniform Q4_K_M base, measure HumanEval pass@1 delta. This is the "
        "measurement multi-domain PPL could NOT make.",
        "",
        f"- baseline (uniform Q4_K_M) HumanEval pass@1: "
        f"{baseline:.1f}%" if baseline is not None else "- baseline: UNMEASURED",
        f"- estimated Modal spend: ${ledger['total_usd']:.2f}",
        "- harness: human_eval pass@1, temp 0, max_tokens 512, parallel 4, 164 problems",
        "",
        "## Per-group HumanEval (demote group to Q2_K)",
        "",
        "| group | pass@1 | Δ vs baseline | classification |",
        "|---|---|---|---|",
    ]
    rows = []
    for name, _ in GROUPS:
        t = res["groups"].get(name)
        if not t:
            lines.append(f"| {name} | - | - | UNMEASURED |")
            continue
        if "error" in t:
            lines.append(f"| {name} | - | - | ERROR: {t['error'][:32]} |")
            continue
        score = t["pass_at_1_pct"]
        delta = t["delta_pct"]
        rows.append((name, score, delta))
        lines.append(f"| {name} | {score:.1f}% | {delta:+.1f}% | {_classify(delta)} |")
    if rows:
        rows.sort(key=lambda x: x[1])
        crit, disp = rows[0], rows[-1]
        lines += [
            "",
            f"CODING-CRITICAL (most damage): **{crit[0]}** "
            f"({crit[1]:.1f}%, {crit[2]:+.1f}%) → PROTECT (promote in budget)",
            f"DISPOSABLE (least damage): **{disp[0]}** "
            f"({disp[1]:.1f}%, {disp[2]:+.1f}%) → crush to make room",
            "",
            "Classification: |Δ| < 5 DISPOSABLE; < 30 MEDIUM; else CODING-CRITICAL.",
            "",
            "NEXT: allocation candidate(s) protecting the coding-critical "
            "groups/layers, then the LOCAL publication gate (HumanEval+ WORKERS=1 "
            "+ ARC vs same-size uniform Q3_K_M). See CODING_ABLATION_VERDICT.md.",
        ]
    with open(f"{RESULTS_DIR}/coding_ablation_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


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


@app.local_entrypoint()
def launch(budget_usd: float = 6.0):
    fc = coding_ablation_driver.spawn(budget_usd=budget_usd)
    print(f"coding ablation driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-north-coding-ablation")
    print("results:  modal volume ls cerebellum-north results/coding_ablation")
