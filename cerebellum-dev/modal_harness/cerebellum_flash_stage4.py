"""Stage 4 — GLM-4.7-Flash benchmark gates on Modal (RUN_PLAN stage 4).

Reuses the campaign app (`cerebellum-flash-campaign`), its Volume, images and
quantize_vol.  The driver runs ON MODAL (detached) so it survives local death.

WHAT IT DOES
  1. Rebuild on the Volume (CPU, parallel spawns; quantize_vol skips if the
     file already exists):
       - C1 conservative: Q4_K_M base + results/phase3/c1_conservative_overrides.txt
         (read from the Volume; falls back to the verbatim phase-3 constant)
       - uniform Q3_K_M baseline (no overrides)
     Both were deleted after their phase-3 PPL — rebuilt exactly as phase 2/3
     built them (same BF16 source, same bartowski imatrix, llama.cpp b9603).
  2. bench_suite (GPU L40S, timeout 3.5 h, ONE model per call — never two GPU
     containers at once, watchdog rule):
       /app/llama-server -ngl 99 --parallel 4 -c 24576 --jinja
       (+ --reasoning-format auto --reasoning-budget 0 when accepted)
     then, in protocol order with the WORKERS rules:
       HumanEval+ (benchmark_evalplus.py, BENCH_WORKERS=1; auto-falls back to
         scripts/benchmark_evalplus_chat.py if a raw-completions probe shows
         GLM's template misbehaving — decision is journaled),
       audit_evalplus_completions.py (mechanical, in-container),
       ARC / HellaSwag / MMLU-Redux at BENCH_WORKERS=4.
     All result JSONs + detailed JSONLs land in Volume results/stage4/<model>/.
  3. stage4_summary.md: C1 vs uniform Q3_K_M table, PPL context, gate verdict
     per the no-ship rule — ALWAYS provisional, pending the human wrong-answer
     audit.  Cost ledger included.
  4. BigCodeBench (Gate 3): SKIPPED here — scripts/benchmark_bigcodebench.py
     needs the `bigcodebench` package plus its sandboxed eval environment
     (heavy setup) and its generate_completion fabricates "    pass" on retry
     exhaustion (the exact BE-15 bug class).  The summary says so loudly:
     Gate 3 remains to run before any ship.

BUDGET: gate locally first —  python3 modal_credits.py --gate 15
  L40S all-in ~ $2.71/h.  Expected: rebuilds ~$0.6-1.0, 2 benches ~ $4-6 ea.
  The driver keeps its own ledger (results/stage4/stage4_ledger.json) and
  refuses to start a bench that would bust the stage budget.

LAUNCH (detached — survives local death):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  nohup setsid modal run --detach cerebellum_flash_stage4.py::launch_stage4 \
      > ../../cerebellum-glm47-flash/logs/modal_stage4_launch.log 2>&1 &

MONITOR / SYNC HOME:
  modal app logs cerebellum-flash-campaign
  modal volume ls cerebellum-flash results/stage4
  modal volume get --force cerebellum-flash results/ \
      ../../cerebellum-glm47-flash/modal_results/results/
"""

import json
import os
import subprocess
import sys
import time

import modal

from cerebellum_modal import CUDA_IMAGE_REF, LLAMA_CPP_TAG
from cerebellum_flash_campaign import (
    DOMAINS,
    RATES,
    V,
    app,
    hf_secret,
    quantize_vol,
    vol,
)

# ---------------------------------------------------------------------------
# Names / paths
# ---------------------------------------------------------------------------
ST4 = f"{V}/results/stage4"
PORT = 8084

C1_NAME = "glm47_flash_cerebellum_c1"
Q3_NAME = "glm47_flash_uniform_q3km"
C1_REL = "candidates/glm47-flash-cerebellum-c1_conservative.gguf"
Q3_REL = "baselines/glm47-flash-Q3_K_M.gguf"
C1_OVERRIDE_VOL = f"{V}/results/phase3/c1_conservative_overrides.txt"

# Verbatim fallback = cerebellum_flash_phase3.C1_OVERRIDE (phase-3 build input)
C1_OVERRIDE_FALLBACK = "\n".join([
    r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$=Q2_K",
    r"^blk\.\d+\.attn_kv_a_mqa\.weight$=Q2_K",
    r"^blk\.0\.ffn_(gate|up|down)\.weight$=Q2_K",
    r"^blk\.\d+\.ffn_down_exps\.weight$=Q3_K",
    r"^blk\.\d+\.attn_output\.weight$=Q3_K",
])

# Phase-3 measured bytes — sanity anchors for the rebuilds
C1_EXPECT_BYTES = 11321136832
Q3_EXPECT_BYTES = 14377306432
EXPECT_VOCAB = 154880

# all-in $/s for the bench shape (L40S + 8 cpu + 48 GiB)
BENCH_RATE = 0.000542 + 8 * 0.0000131 + 48 * 0.00000222
ST4_RATES = dict(RATES, bench_l40s=BENCH_RATE)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_LOCAL = os.path.join(_REPO, "scripts")

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# GPU bench image: pinned llama.cpp full-cuda (has /app/llama-server) + the
# bench scripts' python deps + the repo's scripts/ dir baked in.
bench_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])
    .pip_install(
        "httpx",
        "evalplus==0.3.1",
        "pyarrow",
        "huggingface_hub[hf_transfer]>=0.30",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(_SCRIPTS_LOCAL, remote_path="/bench/scripts")
    .add_local_python_source(
        "cerebellum_modal", "cerebellum_flash_campaign", "cerebellum_flash_stage4"
    )
)

st4_driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_modal", "cerebellum_flash_campaign", "cerebellum_flash_stage4"
)


# ---------------------------------------------------------------------------
# Small helpers (shared by bench + driver)
# ---------------------------------------------------------------------------


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _wait_for_vol_file(path: str, min_bytes: float, tries: int = 40, sleep_s: int = 15):
    """Reader-side existence check with backoff + vol.reload() — the
    volume-race lesson: never declare a file missing on the first look."""
    for i in range(tries):
        vol.reload()
        if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
            s1 = os.path.getsize(path)
            time.sleep(3)
            vol.reload()
            if os.path.exists(path) and os.path.getsize(path) == s1:
                return s1
        print(f"  waiting for {path} (try {i + 1}/{tries})", flush=True)
        time.sleep(sleep_s)
    raise RuntimeError(f"volume file never appeared: {path}")


# ---------------------------------------------------------------------------
# GPU bench function — one model per call (never two concurrently)
# ---------------------------------------------------------------------------


@app.function(
    image=bench_image,
    gpu="L40S",
    cpu=8.0,
    memory=49152,
    timeout=12600,  # 3.5 h
    volumes={V: vol},
    secrets=[hf_secret],
)
def bench_suite(model_rel: str, model_name: str) -> dict:
    """Wrapper guaranteeing llama-server is dead and its volume log handle is
    closed before this function returns OR raises — Modal reuses warm
    containers, and a lingering server process holding llama_server.log open
    on the Volume makes the next call's vol.reload() fail with "there are
    open files preventing the operation" (the 2026-06-12 q3km failure)."""
    state: dict = {"proc": None, "slog": None}
    try:
        return _bench_body(model_rel, model_name, state)
    finally:
        _shutdown_server(state)
        try:
            vol.commit()
        except Exception:
            pass


def _shutdown_server(state: dict):
    """Kill the llama-server child (terminate -> wait -> kill -> wait) and
    flush/close our handle on its volume-resident log file."""
    proc = state.get("proc")
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=15)
            except Exception:
                pass
        state["proc"] = None
    slog = state.get("slog")
    if slog is not None:
        try:
            slog.flush()
            os.fsync(slog.fileno())
        except Exception:
            pass
        try:
            slog.close()
        except Exception:
            pass
        state["slog"] = None


def _bench_body(model_rel: str, model_name: str, state: dict) -> dict:
    import httpx

    t_all = time.time()
    resdir = f"{ST4}/{model_name}"
    os.makedirs(resdir, exist_ok=True)
    report: dict = {"model_rel": model_rel, "model_name": model_name,
                    "llama_cpp_tag": LLAMA_CPP_TAG, "steps": {}}

    def note(msg: str):
        line = f"[{time.strftime('%F %T')}] [{model_name}] {msg}"
        print(line, flush=True)
        with open(f"{resdir}/bench_container.log", "a") as f:
            f.write(line + "\n")

    def save():
        _dump_json(f"{resdir}/bench_report.json", report)
        vol.commit()

    # ---- model from volume (race-safe), copy to local disk for fast mmap ----
    vol_path = f"{V}/{model_rel}"
    nbytes = _wait_for_vol_file(vol_path, 5e9)
    note(f"model on volume: {nbytes / 1e9:.2f} GB")
    local_model = "/tmp/model.gguf"
    t0 = time.time()
    subprocess.run(["cp", vol_path, local_model], check=True)
    note(f"copied to local disk in {time.time() - t0:.0f}s")

    # ---- llama-server ----
    server_bin = next(
        (c for c in ("/app/llama-server", "/llama-server", "/usr/local/bin/llama-server")
         if os.path.exists(c)), None)
    if server_bin is None:
        found = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "llama-server", "-type", "f"],
            capture_output=True, text=True).stdout.split()
        if not found:
            raise RuntimeError("llama-server not found in CUDA image")
        server_bin = found[0]

    base_cmd = [server_bin, "-m", local_model, "--host", "127.0.0.1",
                "--port", str(PORT), "-ngl", "99", "--parallel", "4",
                "-c", "24576", "--jinja", "--no-webui", "-a", model_name]
    flag_sets = [
        ["--reasoning-format", "auto", "--reasoning-budget", "0"],
        [],  # fallback if the reasoning flags are rejected
    ]
    server_log = f"{resdir}/llama_server.log"
    proc = None
    for extra in flag_sets:
        cmd = base_cmd + extra
        note("server: " + " ".join(cmd))
        _shutdown_server(state)  # clear any previous attempt's proc/handle
        state["slog"] = open(server_log, "ab")
        proc = state["proc"] = subprocess.Popen(
            cmd, stdout=state["slog"], stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(5)
        if healthy:
            report["server_flags"] = extra
            break
        note(f"server failed with flags {extra} (rc={proc.poll()}); see llama_server.log")
        _shutdown_server(state)
        vol.commit()
    else:
        raise RuntimeError("llama-server never became healthy (both flag sets)")
    note("server healthy")

    # ---- props / vocab sanity (EOS vocab 154880 watch) ----
    try:
        props = httpx.get(f"http://127.0.0.1:{PORT}/props", timeout=10).json()
        report["props_total_size"] = props.get("total_size")
        with open(f"{resdir}/server_props.json", "w") as f:
            json.dump(props, f, indent=2)
        ptxt = json.dumps(props)
        report["vocab_154880_seen"] = str(EXPECT_VOCAB) in ptxt
        note(f"props saved; vocab {EXPECT_VOCAB} seen in props: {report['vocab_154880_seen']}")
    except Exception as e:
        note(f"props fetch failed (non-fatal): {e}")

    # ---- smoke: thinking actually disabled? ----
    try:
        r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json={
            "model": model_name, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user",
                          "content": "What is 2+2? Answer with just the number."}],
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=120).json()
        msg = r.get("choices", [{}])[0].get("message", {})
        report["smoke_chat"] = {"content": msg.get("content"),
                                "reasoning_content": msg.get("reasoning_content")}
        note(f"smoke chat: {report['smoke_chat']}")
    except Exception as e:
        note(f"smoke chat failed (non-fatal): {e}")
    save()

    # ---- script runner ----
    scripts = "/bench/scripts"
    py = sys.executable

    def run_logged(name: str, cmd: list, env_extra: dict, logfile: str) -> int:
        # env_extra may legitimately override defaults (e.g. the probe passes
        # its own RESULTS_DIR) — build with a dict literal, where duplicate
        # keys override, NOT dict(**kw) which raises "dict() got multiple
        # values for keyword argument" on any overlap (killed the 2026-06-12
        # c1 bench at the very first probe call).
        env = {**os.environ, "BENCH_PORT": str(PORT),
               "BENCH_MODEL": model_name, "RESULTS_DIR": resdir,
               "PYTHONUNBUFFERED": "1", **env_extra}
        t0 = time.time()
        note(f"RUN {name}: {' '.join(cmd)}  env+={env_extra}")
        with open(logfile, "a") as lf:
            p = subprocess.Popen(cmd, cwd=scripts, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True)
            n = 0
            for line in p.stdout:
                lf.write(line)
                n += 1
                # keep modal logs readable: progress lines every 50th only
                if not line.startswith("  [") or n % 50 == 0:
                    print(f"[{model_name}/{name}] {line.rstrip()}", flush=True)
            p.wait()
        secs = time.time() - t0
        report["steps"][name] = {"rc": p.returncode, "secs": round(secs, 1)}
        note(f"DONE {name}: rc={p.returncode} in {secs:.0f}s")
        save()
        return p.returncode

    # =========== 1. HumanEval+ (protocol: first, WORKERS=1) ===========
    # Probe raw /v1/completions first: GLM's chat-tuned template can make raw
    # completions garbage.  8 problems into a probe dir, then decide.
    probe_dir = f"{resdir}/probe"
    os.makedirs(probe_dir, exist_ok=True)
    probe_env = {"BENCH_WORKERS": "1", "RESULTS_DIR": probe_dir}
    rc = run_logged("evalplus_probe",
                    [py, "benchmark_evalplus.py", "8"],
                    probe_env, f"{resdir}/evalplus_probe.log")
    use_chat = False
    probe_samples = f"{probe_dir}/{model_name}_evalplus_samples.jsonl"
    nonempty = 0
    probe_base_pass = None
    if os.path.exists(probe_samples):
        entries = [json.loads(ln) for ln in open(probe_samples) if ln.strip()]
        nonempty = sum(1 for e in entries if len(e.get("completion", "").strip()) >= 10)
        pr = _load_json(f"{probe_dir}/{model_name}_evalplus_results.json", {})
        probe_base_pass = pr.get("pass_at_1_base")
    if rc != 0 or nonempty < 6 or not probe_base_pass:
        use_chat = True
    report["evalplus_probe"] = {"rc": rc, "nonempty_of_8": nonempty,
                                "probe_base_pass": probe_base_pass,
                                "decision": "chat" if use_chat else "raw"}
    probe_desc = f"rc={rc} nonempty={nonempty}/8 base_pass={probe_base_pass}"
    if use_chat:
        note(f"EVALPLUS PATH DECISION: CHAT (raw completions misbehaved: {probe_desc})")
    else:
        note(f"EVALPLUS PATH DECISION: RAW ({probe_desc})")
    save()

    if use_chat:
        ep_script = "benchmark_evalplus_chat.py"
        ep_env = {"BENCH_WORKERS": "1", "BENCH_ENABLE_THINKING": "0",
                  "BENCH_THINKING_BUDGET": "0"}
        samples = f"{resdir}/{model_name}_evalplus_chat_samples.jsonl"
    else:
        ep_script = "benchmark_evalplus.py"
        ep_env = {"BENCH_WORKERS": "1", "BENCH_MAX_TOKENS": "4096"}
        samples = f"{resdir}/{model_name}_evalplus_samples.jsonl"
    rc = run_logged("evalplus", [py, ep_script], ep_env, f"{resdir}/evalplus.log")
    if rc != 0 and os.path.exists(samples):
        # generation may be complete but in-process evaluate() crashed —
        # try the evalplus CLI evaluator on the saved samples
        note("evalplus script rc!=0 with samples present — trying CLI evaluator")
        run_logged("evalplus_cli_eval",
                   [py, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                    "--samples", samples],
                   {}, f"{resdir}/evalplus_cli_eval.log")

    # mechanical in-container audit (human wrong-answer audit happens at review)
    if os.path.exists(samples):
        run_logged("audit_evalplus",
                   [py, "audit_evalplus_completions.py", samples],
                   {}, f"{resdir}/audit_evalplus.txt")
    else:
        note("AUDIT SKIPPED: no samples file found — evalplus generation failed")
    save()

    # =========== 2-4. ARC / HellaSwag / MMLU-Redux (WORKERS=4) ===========
    for name, script in [("arc", "benchmark_arc.py"),
                         ("hellaswag", "benchmark_hellaswag.py"),
                         ("mmlu_redux", "benchmark_mmlu_redux.py")]:
        run_logged(name, [py, script], {"BENCH_WORKERS": "4"},
                   f"{resdir}/{name}.log")

    # ---- collect summaries ----
    scores = {}
    for bench, fname in [
        ("evalplus", f"{model_name}_evalplus_results.json"),
        ("evalplus_chat", f"{model_name}_evalplus_chat_results.json"),
        ("arc", f"{model_name}_arc_results.json"),
        ("hellaswag", f"{model_name}_hellaswag_results.json"),
        ("mmlu_redux", f"{model_name}_mmlu_redux_results.json"),
    ]:
        p = f"{resdir}/{fname}"
        if os.path.exists(p):
            scores[bench] = _load_json(p, {})
    report["scores"] = scores
    report["total_secs"] = round(time.time() - t_all, 1)

    _shutdown_server(state)
    save()
    note(f"bench suite complete in {report['total_secs'] / 3600:.2f}h: "
         f"{json.dumps({k: _topline(k, v) for k, v in scores.items()})}")
    return report


def _topline(bench: str, r: dict):
    for k in ("pass_at_1_plus", "accuracy", "score", "pass_at_1_base"):
        if r.get(k) is not None:
            return {k: r[k]}
    return {}


# ---------------------------------------------------------------------------
# Driver (runs ON MODAL, detached)
# ---------------------------------------------------------------------------


@app.function(
    image=st4_driver_image,
    cpu=1.0,
    memory=2048,
    timeout=int(8.5 * 3600),  # under the 9 h watchdog
    volumes={V: vol},
)
def stage4_driver(budget_usd: float = 9.5) -> dict:
    t_driver = time.time()
    os.makedirs(ST4, exist_ok=True)
    ledger_path = f"{ST4}/stage4_ledger.json"
    ledger = _load_json(ledger_path, {"entries": [], "total_usd": 0.0})

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(f"{ST4}/stage4_progress.log", "a") as f:
            f.write(line + "\n")
        vol.commit()

    def add_cost(kind: str, secs: float, note: str = ""):
        usd = ST4_RATES[kind] * secs
        ledger["entries"].append({"kind": kind, "secs": round(secs, 1),
                                  "usd": round(usd, 4), "note": note})
        ledger["total_usd"] = round(sum(e["usd"] for e in ledger["entries"]), 4)
        _dump_json(ledger_path, ledger)
        vol.commit()
        return usd

    def gate(est: float, what: str):
        driver_usd = ST4_RATES["driver"] * (time.time() - t_driver)
        projected = ledger["total_usd"] + driver_usd + est
        if projected > budget_usd:
            log(f"BUDGET ABORT before {what}: ledger ${ledger['total_usd']:.2f} "
                f"+ driver ${driver_usd:.2f} + est ${est:.2f} > ${budget_usd:.2f}")
            _dump_json(f"{ST4}/ABORTED.json", {"before": what, "ledger": ledger})
            vol.commit()
            raise SystemExit(f"budget abort before {what}")
        log(f"gate OK for {what}: ledger ${ledger['total_usd']:.2f} "
            f"+ est ${est:.2f} <= ${budget_usd:.2f}")

    log(f"=== stage 4 driver start (budget ${budget_usd:.2f}) ===")

    # ---- C1 override: read from the Volume (race-safe), fallback verbatim ----
    c1_override = C1_OVERRIDE_FALLBACK
    try:
        _wait_for_vol_file(C1_OVERRIDE_VOL, 50, tries=4, sleep_s=10)
        with open(C1_OVERRIDE_VOL) as f:
            vol_text = f.read().strip()
        if vol_text:
            if vol_text != C1_OVERRIDE_FALLBACK:
                log("NOTE: volume c1 override differs from fallback constant — "
                    "using the VOLUME version (phase-3 source of truth)")
            c1_override = vol_text
        log(f"C1 override ({C1_OVERRIDE_VOL}):\n{c1_override}")
    except Exception as e:
        log(f"WARNING: could not read {C1_OVERRIDE_VOL} ({e}) — "
            f"using verbatim phase-3 fallback constant")

    # ---- 1. rebuild candidates (CPU, parallel; skip-if-exists) ----
    gate(1.2, "rebuilds")
    builds = {}
    t0 = time.time()
    handles = {
        C1_REL: quantize_vol.spawn(C1_REL, "Q4_K_M", c1_override),
        Q3_REL: quantize_vol.spawn(Q3_REL, "Q3_K_M", ""),
    }
    log("spawned rebuilds: C1 (Q4_K_M base + phase3 overrides) and uniform Q3_K_M")
    for rel, h in handles.items():
        b = h.get()
        add_cost("quantize", time.time() - t0, f"rebuild {rel}")
        builds[rel] = b
        conv = b.get("converting_counts", {})
        if rel == C1_REL and not b.get("skipped"):
            q2 = sum(v for k, v in conv.items() if k.lower() == "q2_k")
            q3 = sum(v for k, v in conv.items() if k.lower() == "q3_k")
            if q2 == 0 or q3 == 0:
                log(f"WARNING C1: override may not have applied! conv={conv}")
        expect = C1_EXPECT_BYTES if rel == C1_REL else Q3_EXPECT_BYTES
        match = b["bytes"] == expect
        log(f"built {rel}: {b['bytes'] / 1e9:.2f} GB "
            f"(phase-3 byte match: {match}; skipped={b.get('skipped', False)})")
        if not match:
            log(f"WARNING {rel}: bytes {b['bytes']} != phase-3 {expect} — "
                f"flagging for the human audit pass")
    _dump_json(f"{ST4}/rebuilds.json", builds)
    vol.commit()

    # ---- 2. bench C1 first, then Q3_K_M — strictly sequential (one GPU) ----
    reports = {}
    bench_state_path = f"{ST4}/bench_state.json"
    bench_state = _load_json(bench_state_path, {})

    for rel, name, est in [(C1_REL, C1_NAME, 3.5), (Q3_REL, Q3_NAME, None)]:
        if bench_state.get(name, {}).get("done"):
            log(f"skip bench {name} (already done)")
            reports[name] = bench_state[name].get("report", {})
            continue
        # second bench estimate = actual cost of the first (plus 20%)
        if est is None:
            prev = [e for e in ledger["entries"] if e["kind"] == "bench_l40s"]
            est = (prev[-1]["usd"] * 1.2) if prev else 3.5
        gate(est, f"bench {name}")
        t0 = time.time()
        try:
            r = bench_suite.remote(rel, name)
            add_cost("bench_l40s", time.time() - t0, f"bench {name}")
            reports[name] = r
            bench_state[name] = {"done": True, "report": r}
        except Exception as e:
            add_cost("bench_l40s", time.time() - t0, f"bench {name} FAILED")
            log(f"bench {name} FAILED: {e}")
            bench_state[name] = {"done": False, "error": str(e)}
            reports[name] = {"error": str(e)}
        _dump_json(bench_state_path, bench_state)
        vol.commit()
        log(f"bench {name} finished (cum ${ledger['total_usd']:.2f})")

    # ---- 3. summary ----
    add_cost("driver", time.time() - t_driver, "driver wall time")
    vol.reload()
    baselines = _load_json(f"{V}/results/baselines.json", {})
    summary = _render_stage4_summary(reports, builds, baselines, ledger)
    with open(f"{ST4}/stage4_summary.md", "w") as f:
        f.write(summary)
    _dump_json(f"{ST4}/stage4_results.json",
               {"reports": reports, "builds": builds,
                "ledger_usd": ledger["total_usd"]})
    vol.commit()
    log(f"=== stage 4 complete. est spend ${ledger['total_usd']:.2f} — "
        f"summary at results/stage4/stage4_summary.md ===")
    print(summary, flush=True)
    return {"ledger_usd": ledger["total_usd"]}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

BENCH_KEYS = [
    ("HumanEval+ (pass@1, plus)", ["evalplus", "evalplus_chat"], "pass_at_1_plus"),
    ("HumanEval (pass@1, base)", ["evalplus", "evalplus_chat"], "pass_at_1_base"),
    ("ARC-Challenge", ["arc"], None),
    ("HellaSwag", ["hellaswag"], None),
    ("MMLU-Redux", ["mmlu_redux"], None),
]


def _score_of(report: dict, benches: list, key) -> float | None:
    scores = (report or {}).get("scores", {})
    for b in benches:
        r = scores.get(b)
        if not r:
            continue
        if key and r.get(key) is not None:
            return float(r[key])
        for k in ("accuracy", "score", "pct_correct"):
            if r.get(k) is not None:
                v = float(r[k])
                return v * 100 if v <= 1.0 else v
    return None


def _render_stage4_summary(reports, builds, baselines, ledger) -> str:
    c1 = reports.get(C1_NAME, {})
    q3 = reports.get(Q3_NAME, {})
    c1_gb = builds.get(C1_REL, {}).get("bytes", C1_EXPECT_BYTES) / 1e9
    q3_gb = builds.get(Q3_REL, {}).get("bytes", Q3_EXPECT_BYTES) / 1e9

    lines = [
        "# GLM-4.7-Flash stage 4 — benchmark gates: C1 cerebellum vs uniform Q3_K_M",
        "",
        f"Generated {time.strftime('%F %T')} on Modal (llama.cpp {LLAMA_CPP_TAG}, "
        f"L40S, server `-ngl 99 --parallel 4 -c 24576 --jinja`).",
        "",
        "**All verdicts below are PROVISIONAL — pending the human wrong-answer "
        "audit** (sample `correct == false` entries in the detailed JSONLs and "
        "confirm real model errors, not parser/clipping artifacts). The "
        "in-container `audit_evalplus_completions.py` run is mechanical only.",
        "",
        "## Head-to-head",
        "",
        f"| benchmark | C1 cerebellum ({c1_gb:.2f} GB) | uniform Q3_K_M "
        f"({q3_gb:.2f} GB) | delta (C1 - Q3) | gate |",
        "|---|---|---|---|---|",
    ]
    verdicts = []
    for label, benches, key in BENCH_KEYS:
        a = _score_of(c1, benches, key)
        b = _score_of(q3, benches, key)
        if a is None or b is None:
            lines.append(f"| {label} | {a if a is not None else 'MISSING'} | "
                         f"{b if b is not None else 'MISSING'} | - | NO DATA |")
            verdicts.append(None)
            continue
        d = a - b
        # gate: C1 (smaller) must beat or match the bigger uniform baseline;
        # allow 0.5 pt noise band, flag anything inside it for the audit
        if d >= 0.5:
            g = "PASS (provisional)"
        elif d >= -0.5:
            g = "TIE-ish — flag for audit"
        else:
            g = "FAIL (provisional)"
        verdicts.append(d)
        lines.append(f"| {label} | {a:.2f} | {b:.2f} | {d:+.2f} | {g} |")

    known = [v for v in verdicts if v is not None]
    if not known:
        overall = "NO VERDICT — benchmark data missing; see errors below"
    elif all(v >= -0.5 for v in known) and any(v >= 0.5 for v in known):
        overall = ("PROVISIONAL PASS — C1 (smaller) beats or matches the bigger "
                   "uniform Q3_K_M on every benchmark. NOT final: pending "
                   "wrong-answer audit + BigCodeBench Gate 3.")
    elif all(v >= -0.5 for v in known):
        overall = ("PROVISIONAL TIE — no benchmark lost by more than the noise "
                   "band, none clearly won. Human review decides. Pending "
                   "wrong-answer audit + Gate 3.")
    else:
        overall = ("PROVISIONAL FAIL — C1 loses at least one benchmark to the "
                   "bigger uniform Q3_K_M. No-ship unless the wrong-answer "
                   "audit overturns the losing rows.")
    lines += ["", f"**Gate verdict: {overall}**", ""]

    # EvalPlus path + suspicious flags
    lines += ["## Run notes / flags for the human audit", ""]
    for name, rep in ((C1_NAME, c1), (Q3_NAME, q3)):
        if not rep or "error" in rep:
            lines.append(f"- {name}: BENCH ERROR — {rep.get('error', 'no report')}")
            continue
        probe = rep.get("evalplus_probe", {})
        lines.append(f"- {name}: HumanEval+ ran via **{probe.get('decision', '?')}** "
                     f"completions (probe: {probe.get('nonempty_of_8', '?')}/8 "
                     f"non-empty, base pass {probe.get('probe_base_pass')}). "
                     f"Server flags: {rep.get('server_flags')}. "
                     f"Smoke chat: {rep.get('smoke_chat', {}).get('content')!r}")
        bad_steps = {k: v for k, v in rep.get("steps", {}).items()
                     if v.get("rc") not in (0, None)}
        if bad_steps:
            lines.append(f"  - SUSPICIOUS: non-zero step rcs: {bad_steps}")
    lines += [
        "",
        "Detailed JSONLs, audit output, server logs: Volume "
        f"`results/stage4/{C1_NAME}/` and `results/stage4/{Q3_NAME}/`.",
        "",
        "## PPL context (phase 1/3, calibration corpora)",
        "",
        "| build | size GB | " + " | ".join(DOMAINS) + " |",
        "|---|---|" + "---|" * len(DOMAINS),
    ]

    def bppl(q, d):
        try:
            return f"{baselines[q]['ppl'][f'cerebellum_calibration_{d}.txt']:.4f}"
        except Exception:
            return "-"

    for q in ("Q4_K_M", "Q3_K_M"):
        if q in baselines:
            gb = baselines[q].get("build", {}).get("bytes", 0) / 1e9
            lines.append(f"| uniform {q} | {gb:.2f} | "
                         + " | ".join(bppl(q, d) for d in DOMAINS) + " |")
    lines.append("| C1 cerebellum (phase 3) | 11.32 | 9.4454 | 3.7114 | 2.9310 "
                 "| 3.6417 |")
    lines += [
        "",
        "Q4_K_M published benchmark numbers: NOT AVAILABLE for this model in "
        "this repo — only PPL was measured (phase 1). The required head-to-head "
        "is the same-size-class comparison above (C1 11.32 GB vs uniform "
        "Q3_K_M 14.38 GB — C1 is 3 GB SMALLER, so matching is winning).",
        "",
        "## Gate 3 — BigCodeBench: **SKIPPED — REMAINS TO RUN BEFORE ANY SHIP**",
        "",
        "The repo has `scripts/benchmark_bigcodebench.py` (endpoint-based "
        "generation) but its evaluation needs the `bigcodebench` package plus "
        "its sandboxed eval environment (heavy task-dependency set) — not "
        "improvised here. ALSO NOTE: that script's `generate_completion` "
        "returns a fabricated `    pass` on retry exhaustion (the BE-15 bug "
        "class) — fix before using it for a published Gate 3 number.",
        "",
        "## Cost ledger (stage 4, list-price estimates)",
        "",
        "| kind | secs | usd | note |",
        "|---|---|---|---|",
    ]
    for e in ledger.get("entries", []):
        lines.append(f"| {e['kind']} | {e['secs']} | {e['usd']:.4f} | {e['note']} |")
    lines.append(f"| **total** | | **${ledger.get('total_usd', 0):.2f}** | |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Local entrypoint — spawn the driver and exit (use with `modal run --detach`)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def launch_stage4(budget_usd: float = 9.5):
    fc = stage4_driver.spawn(budget_usd=budget_usd)
    print(f"stage4 driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-flash-campaign")
    print("results:  modal volume ls cerebellum-flash results/stage4")
