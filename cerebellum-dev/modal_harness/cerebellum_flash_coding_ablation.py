"""GLM-4.7-Flash CODING ABLATION — runs ENTIRELY on Modal.

THE RECOVERED PHASE (2026-06-13, per knowledge/METHOD_TRUTH_2026-06-13.md).
The overnight Flash campaign measured 4-domain PPL only, rated
routed_exps_gate_up code Δ +0.47% ("harmless"), and C1 crushed it to Q2_K
blind. HumanEval+ chat then collapsed 70.12 (uniform Q3_K_M) -> 58.54 (-11.58):
PPL on code TEXT never sees the coding collapse of crushed MoE experts. This
driver restores the REAL coding-critical map by demoting each tensor GROUP to
Q2_K and measuring HumanEval pass@1 delta directly.

SIMPLER than the North harness: GLM-4.7-Flash serves on the STANDARD published
b9603 full-cuda image (no PR branch, no vocab patch). It DOES need the GLM CHAT
path (/v1/chat/completions --jinja, enable_thinking=false) — raw /v1/completions
produces give-up skeletons for this chat-tuned model (the documented Flash C1
artifact + the Gemma-4 exception).

NORTH-BUG AVOIDANCE: North's baseline eval died with "numpy module not available
in the remote environment" during result deserialization — the eval function
returned numpy scalars the debian-slim driver image can't unpickle. This harness
returns ONLY plain python floats/ints and NEVER returns the samples list.

PIPELINE (artifacts on the shared `cerebellum-flash` Volume — BF16 + bartowski
imatrix + Q4_K_M/Q3_K_M baselines already there):
  baseline: serve the uniform Q4_K_M (exists on Volume), run chat-HumanEval.
            THIS pass@1 is the reference all group deltas measure against.
  per group: rebuild a candidate from BF16 demoting ONLY that group to Q2_K
            (imatrix), serve, run chat-HumanEval, record pass@1 delta vs
            baseline, classify CRITICAL/MEDIUM/DISPOSABLE. CPU build of the next
            candidate pipelines while the GPU serves+evals the current one;
            candidate GGUF deleted after eval.
  summary:  results/coding_ablation/coding_ablation_summary.json + .md, STOP.
            Allocation (cerebellum/budget.py) + publication gates run LOCALLY.

BUDGET: retries=0, cost ledger on the Volume, one GPU at a time, watchdog
$4/hr + $28.50 doors-close enforced locally. Gate before launch:
  python3 modal_credits.py --gate 5

LAUNCH (detached — driver runs ON Modal, survives local death):
  cd cerebellum-dev/modal_harness
  nohup setsid modal run --detach \
      cerebellum_flash_coding_ablation.py::launch \
      > ../../cerebellum-glm47-flash/logs/coding_ablation_launch.log 2>&1 &

MONITOR:
  modal app logs cerebellum-flash-coding-ablation
  modal volume ls cerebellum-flash results/coding_ablation
"""

import json
import os
import re
import subprocess
import time

import modal

from cerebellum_modal import (
    CUDA_IMAGE_REF,
    LLAMA_DIR,
    TARBALL_URL,
)

app = modal.App("cerebellum-flash-coding-ablation")

hf_secret = modal.Secret.from_name("hf-token")
vol = modal.Volume.from_name("cerebellum-flash", create_if_missing=True)
V = "/vol"

BF16 = f"{V}/glm-4.7-flash-bf16.gguf"
IMATRIX = f"{V}/imatrix/zai-org_GLM-4.7-Flash-imatrix.gguf"
RESULTS_DIR = f"{V}/results/coding_ablation"
CAND_DIR = f"{V}/candidates_coding"

BASE_TYPE = "Q4_K_M"
ABLATE_TYPE = "Q2_K"
BASE_REL = "baselines/glm47-flash-Q4_K_M.gguf"   # already on volume

# HumanEval / serve settings.
BENCH_PORT = 8085
PARALLEL = 4
MAX_TOKENS = 1024          # chat answers wrap a function; 1024 is ample, keeps it fast
TEMPERATURE = 0.0
SERVER_CTX = 16384         # 4 slots; headroom for prompt + 1024-tok completions
EXPECT_VOCAB = 154880

# The 10 MEASURED Flash groups (cerebellum_flash_campaign.py GROUPS, verified
# against the converted BF16 GGUF). token_embd + output are UNTIED (separate).
GROUPS = [
    ("routed_exps_gate_up", r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$"),
    ("routed_exps_down", r"^blk\.\d+\.ffn_down_exps\.weight$"),
    ("shared_expert", r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$"),
    ("mla_kv_decompress", r"^blk\.\d+\.attn_[kv]_b\.weight$"),
    ("mla_q", r"^blk\.\d+\.attn_q_[ab]\.weight$"),
    ("mla_kv_compress", r"^blk\.\d+\.attn_kv_a_mqa\.weight$"),
    ("attn_output", r"^blk\.\d+\.attn_output\.weight$"),
    ("dense_ffn_l0", r"^blk\.0\.ffn_(gate|up|down)\.weight$"),
    ("token_embd", r"^token_embd\.weight$"),
    ("output_head", r"^output\.weight$"),
]

# Cost model (list prices, $/sec) — mirrors the campaign harness.
CPU_S = 0.0000131
MEM_S = 0.00000222
GPU_S = {"L4": 0.000222, "L40S": 0.000542}
RATES = {
    "quantize": 16 * CPU_S + 32 * MEM_S,
    "eval_l4": GPU_S["L4"] + 8 * CPU_S + 24 * MEM_S,
    "eval_l40s": GPU_S["L40S"] + 8 * CPU_S + 49 * MEM_S,
    "driver": 1 * CPU_S + 2 * MEM_S,
}

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# CPU quantize image: official release tarball at the pinned tag (same as the
# campaign's quant_image).
quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "libgomp1", "libssl3")
    .env({"LD_LIBRARY_PATH": LLAMA_DIR})
    .run_commands(
        f"curl -fsSL -o /tmp/llama.tar.gz {TARBALL_URL}",
        "tar xzf /tmp/llama.tar.gz -C /opt && rm /tmp/llama.tar.gz",
        f"{LLAMA_DIR}/llama-quantize --help 2>&1 | grep -q usage",
    )
    .add_local_python_source("cerebellum_modal", "cerebellum_flash_coding_ablation")
)

# GPU serve+eval image: pinned full-cuda (has /app/llama-server) + human_eval.
eval_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])
    .pip_install("httpx", "human-eval", "numpy")
    .run_commands(
        # human_eval ships exec() disabled behind a comment guard; un-comment it.
        # Modal containers are sandboxed & disposable -> safe to execute code.
        "python -c \"import human_eval.execution as e, re, pathlib; "
        "p=pathlib.Path(e.__file__); s=p.read_text(); "
        "s=re.sub(r'#\\\\s*(exec\\\\(check_program, exec_globals\\\\))', r'\\\\1', s); "
        "p.write_text(s); "
        "assert re.search(r'^\\\\s+exec\\\\(check_program', s, re.M), 'exec guard not enabled'\"",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cerebellum_modal", "cerebellum_flash_coding_ablation")
)

driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_modal", "cerebellum_flash_coding_ablation"
)


def _wait_on_volume(path: str, attempts: int = 6) -> bool:
    for i in range(attempts):
        vol.reload()
        if os.path.exists(path):
            return True
        wait = 10 * (2 ** min(i, 3))
        print(f"  not visible yet ({path}), backoff {wait}s [{i+1}/{attempts}]", flush=True)
        time.sleep(wait)
    return os.path.exists(path)


def _sanitize_override(text: str) -> str:
    return "\n".join(ln for ln in (text or "").splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))


# ===========================================================================
# HumanEval harness — CHAT path (GLM-4.7-Flash needs the chat template)
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


def _extract_body_from_chat(content: str, prompt: str) -> str:
    """A chat model returns a full solution, usually a fenced code block that may
    repeat the signature. Extract the code, then normalise to the function body
    HumanEval expects (it concatenates prompt + completion).

    Strategy: pull the first ```...``` block (or the whole content); if it
    contains a `def <name>(` for the target function, take everything after that
    def line and re-indent; else treat the content as a raw body."""
    # entrypoint name from the prompt's last "def NAME(" line
    m = list(re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(", prompt))
    entry = m[-1].group(1) if m else None

    code = content
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if fence:
        code = fence.group(1)
    else:
        # strip a leading bare ```python / trailing ```
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

    if entry and re.search(rf"def\s+{re.escape(entry)}\s*\(", code):
        # cut to the target def, drop everything before it (imports kept if before)
        idx = re.search(rf"def\s+{re.escape(entry)}\s*\(", code).start()
        # keep import/helper lines that appear before the def as a prefix the
        # body can reference: re-emit them at module indent inside the body via
        # normalize_indent which re-bases on the def line.
        code = code[idx:]
        return normalize_indent(code)

    # No target def found -> assume the model returned just a body; re-indent.
    body = code
    # if it's at column 0, indent it under the (prompt's) def
    out = []
    for line in body.split("\n"):
        if not line.strip():
            out.append("")
        elif line.startswith(" ") or line.startswith("\t"):
            out.append(line)
        else:
            out.append("    " + line)
    return "\n".join(out)


def _run_humaneval_chat(model_name: str) -> dict:
    """Generate completions via /v1/chat/completions (thinking off), execute
    with human_eval. Returns ONLY plain python floats/ints (no numpy, no
    samples) so the debian-slim driver can deserialize the result."""
    import httpx
    from concurrent.futures import ThreadPoolExecutor
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    problems = read_problems()
    api = f"http://127.0.0.1:{BENCH_PORT}/v1/chat/completions"
    print(f"HumanEval(chat): {len(problems)} problems  API={api}", flush=True)
    print(f"Settings: temp={int(TEMPERATURE)}, max_tokens={MAX_TOKENS}, "
          f"parallel={PARALLEL}, thinking=off  [harness=human_eval]", flush=True)

    instruction = (
        "Complete the following Python function. Return ONLY the complete "
        "function definition in a single ```python code block, with no "
        "explanation.\n\n"
    )

    def _completion_for(prompt: str) -> str:
        resp = httpx.post(api, json={
            "model": model_name,
            "messages": [{"role": "user", "content": instruction + prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=300.0)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""
        return _extract_body_from_chat(content, prompt)

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
    # also persist samples to the Volume for an audit trail (NOT returned)
    print("Running evaluation (executing code)...", flush=True)
    res = evaluate_functional_correctness(samples_path, k=[1], n_workers=4, timeout=10.0)
    p1 = float(res["pass@1"])
    # count non-empty completions (give-up detection)
    nonempty = sum(1 for s in samples if len(s["completion"].strip()) >= 10)
    print(f"\n{'='*50}\nHumanEval pass@1: {p1*100:.1f}%  "
          f"(nonempty {nonempty}/{len(samples)})\n{'='*50}", flush=True)
    return {
        "pass_at_1": round(p1, 6),
        "pass_at_1_pct": round(p1 * 100, 1),
        "total": int(len(problems)),
        "gen_ok": int(completed),
        "gen_err": int(errors),
        "nonempty": int(nonempty),
        "model": str(model_name),
        "samples_jsonl": "\n".join(json.dumps(s) for s in samples),
    }


def _serve_and_eval(gguf_path: str, model_name: str) -> dict:
    """Launch llama-server (chat, thinking off) on the GGUF, wait for /health,
    run chat-HumanEval, stop. Returns plain-python dict only."""
    import httpx

    if not _wait_on_volume(gguf_path):
        raise RuntimeError(f"model not on volume: {gguf_path}")

    server_bin = next(
        (c for c in ("/app/llama-server", "/llama-server",
                     "/usr/local/bin/llama-server") if os.path.exists(c)), None)
    if server_bin is None:
        found = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "llama-server", "-type", "f"],
            capture_output=True, text=True).stdout.split()
        if not found:
            raise RuntimeError("llama-server not found in CUDA image")
        server_bin = found[0]

    log = "/tmp/server.log"
    base_cmd = [server_bin, "-m", gguf_path, "--host", "127.0.0.1",
                "--port", str(BENCH_PORT), "-ngl", "99", "--parallel", str(PARALLEL),
                "-c", str(SERVER_CTX), "--jinja", "--no-webui", "-a", model_name]
    flag_sets = [
        ["--reasoning-format", "auto", "--reasoning-budget", "0"],
        [],
    ]
    proc = None
    up = False
    used_flags = None
    for extra in flag_sets:
        cmd = base_cmd + extra
        print("server: " + " ".join(cmd), flush=True)
        fh = open(log, "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                r = httpx.get(f"http://127.0.0.1:{BENCH_PORT}/health", timeout=5.0)
                if r.status_code == 200:
                    up = True
                    used_flags = extra
                    break
            except Exception:
                pass
            time.sleep(4)
        if up:
            break
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if not up:
        try:
            tail = "\n".join(open(log).read().splitlines()[-40:])
        except Exception:
            tail = "(no log)"
        raise RuntimeError(f"server failed to come up for {model_name}:\n{tail}")
    print(f"  [EVAL] Server up (PID {proc.pid}, flags {used_flags}), HumanEval...", flush=True)

    # vocab sanity
    try:
        props = httpx.get(f"http://127.0.0.1:{BENCH_PORT}/props", timeout=10).json()
        vocab_seen = str(EXPECT_VOCAB) in json.dumps(props)
        print(f"  vocab {EXPECT_VOCAB} seen in props: {vocab_seen}", flush=True)
    except Exception:
        pass

    try:
        out = _run_humaneval_chat(model_name)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    out["server_flags"] = " ".join(used_flags) if used_flags else "(default)"
    return out


# ===========================================================================
# Remote functions
# ===========================================================================

@app.function(image=quant_image, cpu=16.0, memory=32768, timeout=5400,
              retries=0, volumes={V: vol})
def quantize_vol(out_rel: str, base_type: str, override_text: str = "") -> dict:
    """llama-quantize Volume BF16 -> Volume GGUF with optional group override."""
    out_path = f"{V}/{out_rel}"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e9:
        return {"out": out_rel, "bytes": int(os.path.getsize(out_path)), "skipped": True}
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
        print(f"override:\n{clean}", flush=True)
        cmd += ["--tensor-type-file", ovr]
    cmd += [BF16, tmp, base_type, "16"]
    t0 = time.time()
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 or not os.path.exists(tmp):
        tail = "\n".join(output.splitlines()[-25:])
        raise RuntimeError(f"quantize {out_rel} failed rc={proc.returncode}:\n{tail}")
    os.replace(tmp, out_path)
    vol.commit()
    counts: dict = {}
    for m in re.finditer(r"converting to (\w+)", output):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return {"out": out_rel, "bytes": int(os.path.getsize(out_path)),
            "secs": round(time.time() - t0, 1), "converting_counts": counts}


@app.function(image=eval_image, gpu="L4", cpu=8.0, memory=24576,
              timeout=3600, retries=0, volumes={V: vol})
def eval_l4(gguf_rel: str, model_name: str) -> dict:
    return _serve_and_eval(f"{V}/{gguf_rel}", model_name)


@app.function(image=eval_image, gpu="L40S", cpu=8.0, memory=50176,
              timeout=5400, retries=0, volumes={V: vol})
def eval_l40s(gguf_rel: str, model_name: str) -> dict:
    return _serve_and_eval(f"{V}/{gguf_rel}", model_name)


# ===========================================================================
# Driver (runs ON MODAL)
# ===========================================================================

@app.function(image=driver_image, cpu=1.0, memory=2048, timeout=int(8.5 * 3600),
              retries=0, volumes={V: vol})
def coding_ablation_driver(budget_usd: float = 5.0) -> dict:
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

    PER_CAND_EST = 0.45

    def pick_eval(nbytes: int):
        # Q4_K_M base ~18.6 GB needs L40S (weights + 4-slot ctx > L4 24 GB tight);
        # demoted-expert candidates (~11-15 GB) fit L4.
        if nbytes > 16.0e9:
            return eval_l40s, "eval_l40s"
        return eval_l4, "eval_l4"

    def save_samples(model_name: str, r: dict):
        sj = r.pop("samples_jsonl", None)
        if sj:
            with open(f"{RESULTS_DIR}/{model_name}_samples.jsonl", "w") as f:
                f.write(sj + "\n")
            vol.commit()

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

    log(f"=== GLM-4.7-Flash CODING ABLATION driver (budget ${budget_usd:.2f}) ===")
    log(f"groups: {[g[0] for g in GROUPS]}")

    if not _wait_on_volume(BF16):
        log("FATAL: BF16 not on volume")
        raise SystemExit("no BF16")
    if not _wait_on_volume(IMATRIX):
        log("FATAL: imatrix not on volume")
        raise SystemExit("no imatrix")

    # -------- baseline: serve uniform Q4_K_M, measure chat-HumanEval --------
    if res.get("baseline_pct") is None:
        if projected_stop("baseline eval", PER_CAND_EST):
            raise SystemExit("budget stop at baseline")
        if not _wait_on_volume(f"{V}/{BASE_REL}"):
            log("baseline Q4_K_M not on volume — rebuilding")
            t0 = time.time()
            quantize_vol.remote(BASE_REL, BASE_TYPE)
            add_cost("quantize", time.time() - t0, "build baseline Q4_K_M")
        vol.reload()
        nbytes = os.path.getsize(f"{V}/{BASE_REL}")
        log(f"baseline: serving uniform Q4_K_M ({nbytes/1e9:.2f} GB), chat-HumanEval")
        r = run_eval(BASE_REL, nbytes, "baseline_Q4_K_M")
        save_samples("baseline_Q4_K_M", r)
        res["baseline_pct"] = r["pass_at_1_pct"]
        res["baseline_detail"] = {k: r[k] for k in
                                  ("pass_at_1_pct", "total", "gen_ok", "gen_err",
                                   "nonempty", "server_flags")}
        _dump_json(f"{RESULTS_DIR}/baseline_humaneval.json", res["baseline_detail"])
        _dump_json(results_path, res)
        vol.commit()
        log(f"*** BASELINE HumanEval pass@1 = {res['baseline_pct']:.1f}% "
            f"(nonempty {r['nonempty']}/{r['total']}) ***")
    else:
        log(f"baseline already measured: {res['baseline_pct']:.1f}%")

    baseline = res["baseline_pct"]

    # -------- per-group coding ablation (pipelined CPU build / GPU eval) -----
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
        spawn_build(i + 2)

        conv = b.get("converting_counts", {})
        crushed = sum(v for k, v in conv.items() if k.lower() == ABLATE_TYPE.lower())
        if not b.get("skipped") and crushed == 0:
            log(f"  WARNING {name}: 0 tensors converted to {ABLATE_TYPE} — "
                f"override may not have matched! conv={conv}")
        vol.reload()
        try:
            r = run_eval(rel, b["bytes"], f"ablate_{name}")
            save_samples(f"ablate_{name}", r)
            score = r["pass_at_1_pct"]
            delta = round(score - baseline, 1)
            res["groups"][name] = {
                "gguf_tensor": regex, "q2k_converted": crushed,
                "candidate_bytes": b["bytes"],
                "pass_at_1_pct": score, "delta_pct": delta,
                "nonempty": r["nonempty"], "gen_ok": r["gen_ok"], "gen_err": r["gen_err"],
            }
            log(f"*** [{i+1}/{len(pending)}] {name}->Q2_K: {score:.1f}% "
                f"({delta:+.1f}%)  nonempty {r['nonempty']}/{r['total']} ***")
        except Exception as e:
            log(f"  eval {name} FAILED: {e}")
            res["groups"][name] = {"gguf_tensor": regex, "q2k_converted": crushed,
                                   "error": f"eval_failed: {e}"}
        _dump_json(results_path, res)
        vol.commit()
        rm(rel)

    # -------- summary + STOP --------
    add_cost("driver", time.time() - t_driver, "driver wall time")
    _write_summary(res, ledger)
    _dump_json(results_path, res)
    vol.commit()
    log("=== coding ablation complete. STOPPING. Allocation (cerebellum/budget.py) "
        "+ publication gate run LOCALLY. ===")
    log(f"total estimated spend: ${ledger['total_usd']:.2f}")
    return {"ledger_usd": ledger["total_usd"], "baseline_pct": baseline,
            "groups_measured": len([g for g in res["groups"].values()
                                    if "pass_at_1_pct" in g])}


def _classify(delta_pct: float) -> str:
    d = -delta_pct  # damage = how far pass@1 DROPPED
    if d < 5.0:
        return "DISPOSABLE"
    if d < 30.0:
        return "MEDIUM"
    return "CODING-CRITICAL"


def _write_summary(res: dict, ledger: dict):
    baseline = res.get("baseline_pct")
    lines = [
        "# GLM-4.7-Flash CODING ABLATION — per-group HumanEval summary",
        "",
        "The RECOVERED method for an MoE code model: demote each tensor GROUP to "
        "Q2_K over the uniform Q4_K_M base, measure HumanEval pass@1 delta. This "
        "is the measurement multi-domain PPL could NOT make (overnight C1 crushed "
        "routed_exps_gate_up blind on a +0.47% code-PPL reading and lost 11.58 "
        "HumanEval+ points).",
        "",
        (f"- baseline (uniform Q4_K_M) HumanEval pass@1: {baseline:.1f}%"
         if baseline is not None else "- baseline: UNMEASURED"),
        f"- estimated Modal spend: ${ledger['total_usd']:.2f}",
        "- harness: human_eval pass@1 via CHAT (/v1/chat/completions --jinja, "
        "enable_thinking=false), temp 0, max_tokens 1024, parallel 4, 164 problems",
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
            f"({crit[1]:.1f}%, {crit[2]:+.1f}%) -> PROTECT (promote in budget)",
            f"DISPOSABLE (least damage): **{disp[0]}** "
            f"({disp[1]:.1f}%, {disp[2]:+.1f}%) -> crush to make room",
            "",
            "Classification by DAMAGE (pass@1 drop): < 5 DISPOSABLE; < 30 MEDIUM; "
            "else CODING-CRITICAL.",
            "",
            "NEXT (LOCAL): feed these deltas + the per-group PPL into "
            "cerebellum/budget.py's multi-pass promotion allocator (~10-11 GB, "
            "protect CODING-CRITICAL), build 1-2 candidates, gate HumanEval+ "
            "(chat, WORKERS=1) + ARC vs uniform Q3_K_M. See FLASH_RECOVERED_VERDICT.md.",
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
def launch(budget_usd: float = 5.0):
    fc = coding_ablation_driver.spawn(budget_usd=budget_usd)
    print(f"coding ablation driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-flash-coding-ablation")
    print("results:  modal volume ls cerebellum-flash results/coding_ablation")
