"""Phase 3 — North-Mini-Code-1.0 Cerebellum allocation candidates (RUN_PLAN stage 3).

Reuses the North campaign app's remote functions (quantize_vol / ppl_l4 /
ppl_l40s), the custom PR #24260 image, and the `cerebellum-north` Volume.
Unlike the Flash phase 3 (local driver), THIS driver runs ON MODAL
(detached-first: it survives local session death).  Resumable: existing
volume GGUFs are skipped by quantize_vol, candidates with recorded PPL are
skipped here; state lives on the Volume at results/phase3/.

CANDIDATE DESIGN — ADAPTED FROM THE MEASURED MAP (deviation from the OG 26B
literal mapping, logged in cerebellum-north-mini-code/OPS_LOG.md):
  The 8-group map came back with NO DEMOTABLE groups.  Verdicts:
    PROTECT : routed_exps_gate_up (+9.6% math), routed_exps_down (+8.1% wiki),
              attn_v (+5.1% math), token_embd (+10.0% wiki, TIED head)
    TOLERANT: attn_q, attn_k, attn_output, dense_ffn_l0
  The literal OG recipe degenerates here: "DEMOTABLE->Q2_K" matches nothing,
  and "PROTECT promoted to Q5/Q6" would promote the two routed-expert groups
  (~8 GB of the savings pool) and make the model BIGGER than Q4_K_M.
  The expert groups are the only real size levers, so candidates grade them
  by measured relative sensitivity instead (gate_up > down on code/math;
  verdicts are vs a Q2_K crush — Q3_K is the moderate step the verdict says
  nothing against), and the small fragile groups get the promotions:
    C1 conservative (~14.2 GB, rival to uniform Q3_K_M 14.76):
       both expert groups -> Q3_K, token_embd -> Q6_K, rest Q4_K_M base.
    C2 aggressive (~12.6 GB, between Q3_K_M and Q2_K 11.36):
       gate_up -> Q3_K, down -> Q2_K, TOLERANT attn + dense_l0 -> Q3_K,
       attn_v -> Q5_K, token_embd -> Q6_K.
    C3 floor probe (~9-10 GB, rival to uniform Q2_K):
       like C2 but gate_up -> IQ2_XS and down -> Q2_K (imatrix present;
       no-op guard: hard warning if zero IQ2 conversions recorded).

4-domain calibration PPL per candidate (L4 — image is sm89-only — with L40S
fallback), candidates DELETED from the Volume after PPL.  STOP after
phase3_summary.md: benchmark gates (HumanEval+/BigCodeBench) run LOCALLY per
RUN_PLAN — code model, never benched on Modal.

Budget: gate locally first — python3 modal_credits.py --gate 2.5

LAUNCH (detached):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  nohup setsid modal run --detach cerebellum_north_phase3.py::launch_p3 \
      > ../../cerebellum-north-mini-code/logs/modal_phase3_launch.log 2>&1 &

MONITOR:
  modal app list                       # cerebellum-north-campaign (reused app)
  modal volume ls cerebellum-north results/phase3
"""

import json
import os
import time

from cerebellum_north_campaign import (
    DOMAINS,
    RATES,
    RESULTS_DIR,
    V,
    app,
    driver_image,
    ppl_l4,
    ppl_l40s,
    quantize_vol,
    vol,
)

P3_DIR = f"{RESULTS_DIR}/phase3"
CAL = [f"cerebellum_calibration_{d}.txt" for d in DOMAINS]

# ---------------------------------------------------------------------------
# Group regexes (identical to cerebellum_north_campaign.GROUPS)
# ---------------------------------------------------------------------------
RX_GATE_UP = r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$"
RX_DOWN = r"^blk\.\d+\.ffn_down_exps\.weight$"
RX_ATTN_Q = r"^blk\.\d+\.attn_q\.weight$"
RX_ATTN_K = r"^blk\.\d+\.attn_k\.weight$"
RX_ATTN_V = r"^blk\.\d+\.attn_v\.weight$"
RX_ATTN_OUT = r"^blk\.\d+\.attn_output\.weight$"
RX_DENSE_L0 = r"^blk\.0\.ffn_(gate|up|down)\.weight$"
RX_EMBD = r"^token_embd\.weight$"  # TIED — also the LM head

# Override files: uppercase types, NO comment lines (llama-quantize comment bug).
C1_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=Q3_K",
    f"{RX_DOWN}=Q3_K",
    f"{RX_EMBD}=Q6_K",
])

C2_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=Q3_K",
    f"{RX_DOWN}=Q2_K",
    f"{RX_ATTN_Q}=Q3_K",
    f"{RX_ATTN_K}=Q3_K",
    f"{RX_ATTN_OUT}=Q3_K",
    f"{RX_DENSE_L0}=Q3_K",
    f"{RX_ATTN_V}=Q5_K",
    f"{RX_EMBD}=Q6_K",
])

C3_OVERRIDE = "\n".join([
    f"{RX_GATE_UP}=IQ2_XS",
    f"{RX_DOWN}=Q2_K",
    f"{RX_ATTN_Q}=Q3_K",
    f"{RX_ATTN_K}=Q3_K",
    f"{RX_ATTN_OUT}=Q3_K",
    f"{RX_DENSE_L0}=Q3_K",
    f"{RX_ATTN_V}=Q5_K",
    f"{RX_EMBD}=Q6_K",
])

CANDIDATES = [
    # (name, base_type, override, description)
    ("c1_conservative", "Q4_K_M", C1_OVERRIDE,
     "experts gate_up+down -> Q3_K, token_embd -> Q6_K, rest Q4_K_M base "
     "(~14.2 GB; rival to uniform Q3_K_M)"),
    ("c2_aggressive", "Q4_K_M", C2_OVERRIDE,
     "gate_up -> Q3_K, down -> Q2_K, TOLERANT attn/dense -> Q3_K, "
     "attn_v -> Q5_K, embd -> Q6_K (~12.6 GB)"),
    ("c3_floor_probe", "Q4_K_M", C3_OVERRIDE,
     "like C2 but gate_up -> IQ2_XS (~9-10 GB; rival to uniform Q2_K; "
     "extreme-small probe)"),
]

BUDGET_USD = 2.50


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


p3_image = driver_image.add_local_python_source("cerebellum_north_phase3")


@app.function(
    image=p3_image,
    cpu=1.0,
    memory=2048,
    timeout=4 * 3600,
    retries=0,
    volumes={V: vol},
)
def phase3_driver(budget_usd: float = BUDGET_USD) -> dict:
    t_driver = time.time()
    os.makedirs(P3_DIR, exist_ok=True)
    state_path = f"{P3_DIR}/phase3_results.json"
    state = _load_json(state_path, {"costs": [], "est_total_usd": 0.0,
                                    "candidates": {}})
    baselines = _load_json(f"{RESULTS_DIR}/baselines.json", {})

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(f"{P3_DIR}/phase3_driver.log", "a") as f:
            f.write(line + "\n")

    def save():
        _dump_json(state_path, state)
        vol.commit()

    def add_cost(kind: str, secs: float, note: str):
        usd = RATES[kind] * secs
        state["costs"].append({"kind": kind, "secs": round(secs, 1),
                               "usd": round(usd, 4), "note": note})
        state["est_total_usd"] = round(sum(c["usd"] for c in state["costs"]), 4)
        save()
        return usd

    def rm(rel: str):
        vol.reload()
        p = f"{V}/{rel}"
        if os.path.exists(p):
            os.remove(p)
            vol.commit()
            log(f"  deleted {rel} from volume")

    def run_ppl(rel: str, nbytes: int, note: str) -> tuple[dict, str]:
        """L4 first (<21 GB; image is sm89-only), L40S fallback. ONE GPU at
        a time — sequential by construction."""
        tiers = ([(ppl_l4, "ppl_l4"), (ppl_l40s, "ppl_l40s")]
                 if nbytes < 21.0e9 else [(ppl_l40s, "ppl_l40s")])
        last = None
        for fn, kind in tiers:
            t0 = time.time()
            try:
                r = fn.remote(rel, CAL)
                usd = add_cost(kind, time.time() - t0, note)
                log(f"  {note} on {kind}: {r['ppl']} "
                    f"({time.time()-t0:.0f}s ~${usd:.3f})")
                return r, kind
            except Exception as e:
                add_cost(kind, time.time() - t0, f"{note} FAILED on {kind}")
                log(f"  {note} FAILED on {kind}: {e} — escalating")
                last = e
        raise last

    log(f"=== North phase 3: allocation candidates C1/C2/C3 "
        f"(budget ${budget_usd:.2f}) ===")
    log("map verdicts: PROTECT={exps_gate_up, exps_down, attn_v, token_embd}; "
        "TOLERANT={attn_q, attn_k, attn_output, dense_ffn_l0}; no DEMOTABLE — "
        "candidates adapted (see module docstring)")

    # ----- builds: 2 CPU containers in flight (velocity-cap discipline) -----
    todo = [c for c in CANDIDATES
            if not state["candidates"].get(c[0], {}).get("ppl")]
    for name, *_ in [c for c in CANDIDATES if c[0] not in [t[0] for t in todo]]:
        log(f"skip {name} (PPL already recorded)")
    handles: dict = {}

    def rel_of(name: str) -> str:
        return f"candidates/north-mini-code-cerebellum-{name}.gguf"

    def spawn(i: int):
        if i < len(todo):
            name, base, override, _d = todo[i]
            handles[i] = (quantize_vol.spawn(rel_of(name), base, override),
                          time.time())
            log(f"spawned build [{i+1}/{len(todo)}] {name}")

    spawn(0)
    spawn(1)

    for i, (name, base, override, desc) in enumerate(todo):
        driver_usd = RATES["driver"] * (time.time() - t_driver)
        if state["est_total_usd"] + driver_usd + 0.45 > budget_usd:
            log(f"BUDGET STOP before {name}: est ${state['est_total_usd']:.2f} "
                f"+ driver ${driver_usd:.2f} + ~$0.45 next > ${budget_usd:.2f}")
            break
        rel = rel_of(name)
        h, t0 = handles.pop(i)
        c = state["candidates"].setdefault(name, {})
        c.update({"base_type": base, "override": override, "description": desc})
        try:
            b = h.get()
        except Exception as e:
            add_cost("quantize", time.time() - t0, f"build {name} FAILED")
            c["error"] = f"quantize_failed: {e}"
            save()
            log(f"build {name} FAILED (skipping candidate): {e}")
            spawn(i + 2)
            continue
        add_cost("quantize", time.time() - t0, f"build {name}")
        spawn(i + 2)
        c["build"] = b
        c["bytes"] = b["bytes"]
        conv = b.get("converting_counts", {})
        c["converting_counts"] = conv
        c["fallback_lines"] = b.get("fallback_lines", [])
        # no-op guards (the 12B lesson)
        if name == "c3_floor_probe" and not b.get("skipped"):
            iq = sum(v for k, v in conv.items() if k.lower().startswith("iq2"))
            c["iq2_xs_converted"] = iq
            if iq == 0:
                log(f"WARNING {name}: zero IQ2_XS conversions — override may "
                    f"have silently not applied; conv={conv}")
        if not b.get("skipped") and not conv:
            log(f"WARNING {name}: no conversion lines recorded! conv={conv}")
        save()
        log(f"built {name}: {b['bytes']/1e9:.2f} GB, conv={conv}, "
            f"fallbacks={len(c['fallback_lines'])}")
        vol.reload()
        try:
            r, kind = run_ppl(rel, b["bytes"], f"PPL {name}")
            c["ppl"] = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"]
                        for d in DOMAINS}
            c["ppl_secs"] = r["secs"]
            c["gpu"] = kind
        except Exception as e:
            c["error"] = f"ppl_failed: {e}"
            log(f"PPL {name} FAILED: {e}")
        save()
        rm(rel)

    # ----- summary -----
    add_cost("driver", time.time() - t_driver, "phase3 driver wall time")
    summary = _render_summary(state, baselines)
    with open(f"{P3_DIR}/phase3_summary.md", "w") as f:
        f.write(summary)
    for name, _b, override, _d in CANDIDATES:
        with open(f"{P3_DIR}/{name}_overrides.txt", "w") as f:
            f.write(override + "\n")
    save()
    log(f"phase 3 complete. est spend this phase: "
        f"${state['est_total_usd']:.2f}. STOPPING — benchmark gates "
        f"(HumanEval+/BigCodeBench) run LOCALLY per RUN_PLAN.")
    print(summary, flush=True)
    return {"est_total_usd": state["est_total_usd"],
            "candidates": {k: v.get("ppl") for k, v in state["candidates"].items()}}


def _render_summary(state: dict, baselines: dict) -> str:
    def bppl(q, d):
        try:
            return baselines[q]["ppl"][f"cerebellum_calibration_{d}.txt"]
        except Exception:
            return float("nan")

    lines = [
        "# North-Mini-Code-1.0 phase 3 — allocation candidates (group-verdict builds)",
        "",
        "Built from Volume BF16 + unsloth imatrix (pre-merge PR #24260 "
        "provenance), base Q4_K_M with tensor-type overrides.",
        "",
        "DESIGN NOTE: the measured map has NO DEMOTABLE groups (both routed "
        "expert groups are PROTECT). The literal OG verdict mapping degenerates, "
        "so candidates grade the expert groups (the only size levers) by "
        "relative sensitivity and promote the small fragile groups "
        "(attn_v, tied token_embd). See module docstring + OPS_LOG.",
        "",
        "## Candidates",
        "",
        "| candidate | size GB | GPU | " + " | ".join(DOMAINS) + " |",
        "|---|---|---|" + "---|" * len(DOMAINS),
    ]
    for q in ("Q4_K_M", "Q3_K_M", "Q2_K"):
        if q in baselines and "ppl" in baselines[q]:
            lines.append(
                f"| uniform {q} (baseline) | "
                f"{baselines[q]['build']['bytes']/1e9:.2f} | - | "
                + " | ".join(f"{bppl(q, d):.4f}" for d in DOMAINS) + " |")
    for name, _b, _o, _d in CANDIDATES:
        c = state["candidates"].get(name, {})
        if "ppl" in c:
            lines.append(
                f"| {name} | {c['bytes']/1e9:.2f} | {c.get('gpu', '-')} | "
                + " | ".join(f"{c['ppl'][d]:.4f}" for d in DOMAINS) + " |")
        else:
            lines.append(f"| {name} | - | - | "
                         + " | ".join("-" for _ in DOMAINS)
                         + f" | ERROR: {c.get('error', 'missing')[:60]}")
    for ref in ("Q4_K_M", "Q3_K_M", "Q2_K"):
        lines += ["", f"## ΔPPL% vs uniform {ref}", "",
                  "| candidate | " + " | ".join(f"Δ%{d}" for d in DOMAINS) + " |",
                  "|---|" + "---|" * len(DOMAINS)]
        for name, *_ in CANDIDATES:
            c = state["candidates"].get(name, {})
            if "ppl" not in c:
                continue
            lines.append(f"| {name} | " + " | ".join(
                f"{100*(c['ppl'][d]/bppl(ref, d)-1):+.2f}" for d in DOMAINS) + " |")
    lines += [
        "",
        f"- estimated phase-3 Modal spend: ${state.get('est_total_usd', 0):.2f}",
        "",
        "NEXT: benchmark gates LOCALLY per RUN_PLAN stage 4 (HumanEval+ and "
        "BigCodeBench vs same-size uniform baselines) — human-reviewed, NOT "
        "run on Modal. Re-imatrix after PR #24260 merges before finals.",
    ]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def launch_p3(budget_usd: float = BUDGET_USD):
    fc = phase3_driver.spawn(budget_usd=budget_usd)
    print(f"phase3 driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-north-campaign")
    print("results:  modal volume ls cerebellum-north results/phase3")
