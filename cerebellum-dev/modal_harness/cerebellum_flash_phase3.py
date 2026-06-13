"""Phase 3 — GLM-4.7-Flash Cerebellum allocation candidates (RUN_PLAN stage 3).

Reuses the campaign app's remote functions (quantize_vol / ppl_t4 / ppl_l4 /
ppl_l40s), images, and the `cerebellum-flash` Volume.  The driver runs LOCALLY
(modal run, not detached) and persists every result incrementally to the local
results dir AND the Volume — resumable: existing volume GGUFs are skipped by
quantize_vol, candidates with recorded PPL are skipped here.

Order of operations:
  0. Q8_0 anomaly recheck FIRST (rebuild once, wiki-calibration PPL once on
     L40S, delete).  Phase 1 measured Q8_0 wiki 14.32 vs Q4_K_M 10.43 —
     abnormal; build metadata showed nothing wrong, so reproduce-or-bust.
  1. Build C1/C2/C3 group-verdict candidates from the Volume BF16 + bartowski
     imatrix (CPU, parallel spawns), exactly like the phase-2 builds.
  2. 4-domain calibration PPL per candidate (size-routed GPU, escalating
     fallback T4 -> L4 -> L40S), record actual bytes, DELETE candidate.
  3. Write results/phase3/phase3_summary.md + JSON, push to Volume.

Budget: gate locally first — python3 modal_credits.py --gate 2

Launch:
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  modal run cerebellum_flash_phase3.py::phase3
"""

import json
import os
import subprocess
import time

from cerebellum_flash_campaign import (
    DOMAINS,
    RATES,
    app,
    ppl_l4,
    ppl_l40s,
    ppl_t4,
    quantize_vol,
)

VOLUME = "cerebellum-flash"
LOCAL_P3 = (
    "/var/home/deucebucket/ai-drive/cerebellum/cerebellum-glm47-flash/"
    "modal_results/results/phase3"
)
LOCAL_BASELINES = (
    "/var/home/deucebucket/ai-drive/cerebellum/cerebellum-glm47-flash/"
    "modal_results/results/baselines.json"
)
CAL = [f"cerebellum_calibration_{d}.txt" for d in DOMAINS]
WIKI_CAL = "cerebellum_calibration_wiki.txt"

# ---------------------------------------------------------------------------
# Group regexes (verified in phase 2 — identical to cerebellum_flash_campaign.GROUPS)
# ---------------------------------------------------------------------------
RX_GATE_UP = r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$"
RX_DOWN = r"^blk\.\d+\.ffn_down_exps\.weight$"
RX_SHEXP = r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$"
RX_KV_B = r"^blk\.\d+\.attn_[kv]_b\.weight$"
RX_Q_AB = r"^blk\.\d+\.attn_q_[ab]\.weight$"
RX_KV_A = r"^blk\.\d+\.attn_kv_a_mqa\.weight$"
RX_ATTN_OUT = r"^blk\.\d+\.attn_output\.weight$"
RX_DENSE_L0 = r"^blk\.0\.ffn_(gate|up|down)\.weight$"
RX_EMBD = r"^token_embd\.weight$"
RX_OUTPUT = r"^output\.weight$"

# Override files: uppercase types, NO comment lines (llama-quantize comment bug).
C1_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=Q2_K",
    f"{RX_KV_A}=Q2_K",
    f"{RX_DENSE_L0}=Q2_K",
    f"{RX_DOWN}=Q3_K",
    f"{RX_ATTN_OUT}=Q3_K",
])

C2_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=Q2_K",
    f"{RX_DOWN}=Q2_K",
    f"{RX_KV_A}=Q2_K",
    f"{RX_DENSE_L0}=Q2_K",
    f"{RX_ATTN_OUT}=Q2_K",
    f"{RX_SHEXP}=Q5_K",
    f"{RX_Q_AB}=Q5_K",
    f"{RX_KV_B}=Q5_K",
    f"{RX_EMBD}=Q6_K",
    f"{RX_OUTPUT}=Q6_K",
])

C3_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=IQ2_XS",
    f"{RX_DOWN}=Q2_K",
    f"{RX_KV_A}=Q2_K",
    f"{RX_DENSE_L0}=Q2_K",
    f"{RX_ATTN_OUT}=Q2_K",
    f"{RX_SHEXP}=Q5_K",
    f"{RX_Q_AB}=Q5_K",
    f"{RX_KV_B}=Q5_K",
    f"{RX_EMBD}=Q6_K",
    f"{RX_OUTPUT}=Q6_K",
])

CANDIDATES = [
    # (name, base_type, override, description)
    ("c1_conservative", "Q4_K_M", C1_OVERRIDE,
     "DEMOTABLE->Q2_K, TOLERANT->Q3_K, PROTECT at Q4_K_M base"),
    ("c2_aggressive", "Q4_K_M", C2_OVERRIDE,
     "DEMOTABLE+TOLERANT->Q2_K, PROTECT promoted (embd/head Q6_K, "
     "shexp/mla_q/kv_b Q5_K)"),
    ("c3_floor_probe", "Q4_K_M", C3_OVERRIDE,
     "like C2 but routed gate_up -> IQ2_XS (skip if IQ rejected)"),
]

Q8_REL = "baselines/glm47-flash-Q8_0.gguf"


def _log(msg: str):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOCAL_P3, exist_ok=True)
    with open(f"{LOCAL_P3}/phase3_driver.log", "a") as f:
        f.write(line + "\n")


def _load(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _vol_rm(rel: str):
    r = subprocess.run(["modal", "volume", "rm", VOLUME, rel],
                       capture_output=True, text=True)
    _log(f"volume rm {rel}: rc={r.returncode} {r.stderr.strip()[:120]}")


def _add_cost(state: dict, kind: str, secs: float, note: str):
    usd = RATES[kind] * secs
    state.setdefault("costs", []).append(
        {"kind": kind, "secs": round(secs, 1), "usd": round(usd, 4), "note": note})
    state["est_total_usd"] = round(sum(c["usd"] for c in state["costs"]), 4)
    return usd


def _run_ppl(state, rel, nbytes, corpora, note):
    """Size-routed PPL with escalating GPU fallback. Returns (result, gpu_kind)."""
    if nbytes < 12.5e9:
        tiers = [(ppl_t4, "ppl_t4"), (ppl_l4, "ppl_l4"), (ppl_l40s, "ppl_l40s")]
    elif nbytes < 21.0e9:
        tiers = [(ppl_l4, "ppl_l4"), (ppl_l40s, "ppl_l40s")]
    else:
        tiers = [(ppl_l40s, "ppl_l40s")]
    last = None
    for fn, kind in tiers:
        t0 = time.time()
        try:
            r = fn.remote(rel, corpora)
            usd = _add_cost(state, kind, time.time() - t0, note)
            _log(f"  {note} on {kind}: {r['ppl']}  ({time.time()-t0:.0f}s ~${usd:.3f})")
            return r, kind
        except Exception as e:
            _add_cost(state, kind, time.time() - t0, f"{note} FAILED on {kind}")
            _log(f"  {note} FAILED on {kind}: {e} — escalating")
            last = e
    raise last


@app.local_entrypoint()
def phase3():
    os.makedirs(LOCAL_P3, exist_ok=True)
    state_path = f"{LOCAL_P3}/phase3_results.json"
    state = _load(state_path, {"costs": [], "est_total_usd": 0.0,
                               "candidates": {}, "q8_recheck": {}})
    baselines = _load(LOCAL_BASELINES, {})

    _log("=== phase 3 start: Q8 recheck + allocation candidates C1/C2/C3 ===")

    # ----- spawn ALL CPU builds up front (parallel containers) -----
    handles = {}
    if "ppl_wiki" not in state["q8_recheck"]:
        handles["__q8__"] = (quantize_vol.spawn(Q8_REL, "Q8_0", ""), time.time())
        _log("spawned Q8_0 rebuild (anomaly recheck)")
    for name, base, override, _desc in CANDIDATES:
        if state["candidates"].get(name, {}).get("ppl"):
            _log(f"skip {name} (PPL already recorded)")
            continue
        rel = f"candidates/glm47-flash-cerebellum-{name}.gguf"
        handles[name] = (quantize_vol.spawn(rel, base, override), time.time())
        _log(f"spawned build {name} -> {rel}")

    # ----- Q8 anomaly recheck first (instructed: cheap, do first) -----
    if "__q8__" in handles:
        h, t0 = handles.pop("__q8__")
        try:
            b = h.get()
            _add_cost(state, "quantize", time.time() - t0, "rebuild Q8_0")
            q8 = state["q8_recheck"]
            q8["build"] = b
            q8["bytes_match_phase1"] = (b["bytes"] == 31842799296)
            _dump(state_path, state)
            _log(f"Q8_0 rebuilt: {b['bytes']/1e9:.2f} GB "
                 f"(phase-1 byte match: {q8['bytes_match_phase1']}), "
                 f"conv={b.get('converting_counts')}, skipped={b.get('skipped', False)}")
            r, kind = _run_ppl(state, Q8_REL, b["bytes"], [WIKI_CAL], "PPL Q8_0 recheck wiki")
            q8["ppl_wiki"] = r["ppl"][WIKI_CAL]
            q8["gpu"] = kind
            q8["phase1_ppl_wiki"] = 14.3202
            q8["q4_k_m_ppl_wiki"] = 10.4263
            q8["reproduced"] = abs(q8["ppl_wiki"] - 14.3202) < 0.5
            _dump(state_path, state)
            _log(f"Q8 recheck: wiki PPL {q8['ppl_wiki']:.4f} "
                 f"(phase1 14.3202, Q4_K_M 10.4263) reproduced={q8['reproduced']}")
        except Exception as e:
            state["q8_recheck"]["error"] = str(e)
            _dump(state_path, state)
            _log(f"Q8 recheck FAILED: {e}")
        finally:
            _vol_rm(Q8_REL)

    # ----- candidates: collect builds, PPL, delete -----
    for name, base, override, desc in CANDIDATES:
        if name not in handles:
            continue
        rel = f"candidates/glm47-flash-cerebellum-{name}.gguf"
        h, t0 = handles.pop(name)
        c = state["candidates"].setdefault(name, {})
        c["base_type"] = base
        c["override"] = override
        c["description"] = desc
        try:
            b = h.get()
        except Exception as e:
            _add_cost(state, "quantize", time.time() - t0, f"build {name} FAILED")
            c["error"] = f"quantize_failed: {e}"
            _dump(state_path, state)
            _log(f"build {name} FAILED (skipping candidate): {e}")
            continue
        _add_cost(state, "quantize", time.time() - t0, f"build {name}")
        c["build"] = b
        c["bytes"] = b["bytes"]
        conv = b.get("converting_counts", {})
        c["converting_counts"] = conv
        c["fallback_lines"] = b.get("fallback_lines", [])
        # no-op guards
        if name == "c3_floor_probe" and not b.get("skipped"):
            iq = sum(v for k, v in conv.items() if k.lower().startswith("iq2"))
            c["iq2_xs_converted"] = iq
            if iq == 0:
                _log(f"WARNING {name}: no IQ2_XS conversions recorded — "
                     f"override may have silently not applied; conv={conv}")
        crushed = sum(v for k, v in conv.items() if k.lower() == "q2_k")
        if not b.get("skipped") and crushed == 0:
            _log(f"WARNING {name}: zero Q2_K conversions! conv={conv}")
        _dump(state_path, state)
        _log(f"built {name}: {b['bytes']/1e9:.2f} GB, conv={conv}, "
             f"fallbacks={len(c['fallback_lines'])}")
        try:
            r, kind = _run_ppl(state, rel, b["bytes"], CAL, f"PPL {name}")
            c["ppl"] = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"] for d in DOMAINS}
            c["ppl_secs"] = r["secs"]
            c["gpu"] = kind
        except Exception as e:
            c["error"] = f"ppl_failed: {e}"
            _log(f"PPL {name} FAILED: {e}")
        _dump(state_path, state)
        _vol_rm(rel)

    # ----- summary -----
    summary = _render_summary(state, baselines)
    with open(f"{LOCAL_P3}/phase3_summary.md", "w") as f:
        f.write(summary)
    for i, (name, _b, override, _d) in enumerate(CANDIDATES):
        with open(f"{LOCAL_P3}/{name}_overrides.txt", "w") as f:
            f.write(override + "\n")
    _dump(state_path, state)
    _log(f"phase 3 complete. est spend this phase: ${state['est_total_usd']:.2f}")

    # push phase3 results to the Volume (source of truth mirrors local)
    r = subprocess.run(["modal", "volume", "put", "--force", VOLUME,
                        LOCAL_P3, "results/phase3"],
                       capture_output=True, text=True)
    _log(f"volume put results/phase3: rc={r.returncode} {r.stderr.strip()[:200]}")
    print(summary, flush=True)


def _render_summary(state: dict, baselines: dict) -> str:
    def bppl(q, d):
        try:
            return baselines[q]["ppl"][f"cerebellum_calibration_{d}.txt"]
        except Exception:
            return float("nan")

    lines = [
        "# GLM-4.7-Flash phase 3 — allocation candidates (group-verdict builds)",
        "",
        "Built from Volume BF16 + bartowski imatrix, llama.cpp b9603, base Q4_K_M",
        "with tensor-type overrides from the phase-2 group verdicts.",
        "",
        "## Candidates",
        "",
        "| candidate | size GB | GPU | " + " | ".join(DOMAINS) + " |",
        "|---|---|---|" + "---|" * len(DOMAINS),
    ]
    for q in ("Q4_K_M", "Q3_K_M"):
        if q in baselines and "ppl" in baselines[q]:
            lines.append(
                f"| uniform {q} (baseline) | {baselines[q]['build']['bytes']/1e9:.2f} | - | "
                + " | ".join(f"{bppl(q, d):.4f}" for d in DOMAINS) + " |")
    for name, _b, _o, _d in [(c[0], c[1], c[2], c[3]) for c in CANDIDATES]:
        c = state["candidates"].get(name, {})
        if "ppl" in c:
            lines.append(
                f"| {name} | {c['bytes']/1e9:.2f} | {c.get('gpu','-')} | "
                + " | ".join(f"{c['ppl'][d]:.4f}" for d in DOMAINS) + " |")
        else:
            lines.append(f"| {name} | - | - | "
                         + " | ".join("-" for _ in DOMAINS)
                         + f" | ERROR: {c.get('error','missing')[:60]}")
    lines += ["", "## ΔPPL% vs uniform Q4_K_M (18.13 GB)", ""]
    lines += ["| candidate | " + " | ".join(f"Δ%{d}" for d in DOMAINS) + " |",
              "|---|" + "---|" * len(DOMAINS)]
    for name, *_ in CANDIDATES:
        c = state["candidates"].get(name, {})
        if "ppl" not in c:
            continue
        lines.append(f"| {name} | " + " | ".join(
            f"{100*(c['ppl'][d]/bppl('Q4_K_M', d)-1):+.2f}" for d in DOMAINS) + " |")
    lines += ["", "## ΔPPL% vs uniform Q3_K_M (14.38 GB)", ""]
    lines += ["| candidate | " + " | ".join(f"Δ%{d}" for d in DOMAINS) + " |",
              "|---|" + "---|" * len(DOMAINS)]
    for name, *_ in CANDIDATES:
        c = state["candidates"].get(name, {})
        if "ppl" not in c:
            continue
        lines.append(f"| {name} | " + " | ".join(
            f"{100*(c['ppl'][d]/bppl('Q3_K_M', d)-1):+.2f}" for d in DOMAINS) + " |")

    q8 = state.get("q8_recheck", {})
    lines += ["", "## Q8_0 anomaly recheck", ""]
    if "ppl_wiki" in q8:
        lines += [
            f"- rebuilt Q8_0 ({q8['build']['bytes']/1e9:.2f} GB, byte-identical to "
            f"phase 1: {q8.get('bytes_match_phase1')}), wiki-calibration PPL on "
            f"{q8.get('gpu')}: **{q8['ppl_wiki']:.4f}** "
            f"(phase 1: 14.3202; Q4_K_M: 10.4263)",
            f"- reproduced: {q8.get('reproduced')}",
        ]
    else:
        lines += [f"- recheck error: {q8.get('error', 'not run')}"]
    lines += [
        "",
        f"- estimated phase-3 Modal spend: ${state.get('est_total_usd', 0):.2f}",
        "",
        "NEXT: stage 4 benchmark gates (HumanEval+/ARC/HellaSwag/MMLU + "
        "BigCodeBench) vs same-size uniform baselines — separately reviewed, "
        "NOT run in this phase.",
    ]
    return "\n".join(lines) + "\n"
