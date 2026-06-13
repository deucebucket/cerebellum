"""XXS stage — GLM-4.7-Flash extreme-small candidates on Modal (2026-06-12).

GOAL
----
Build and PPL-screen 3 IQ2-class candidates targeting ~8-9 GB, well below C1
(11.32 GB).  NO benchmarks here — bench gates run locally after download.
Results land in results/xxs/ on the `cerebellum-flash` Volume.

CANDIDATE RATIONALE
-------------------
All three start from the map verdicts (ablation_summary.md):
  DEMOTABLE: routed_exps_gate_up, mla_kv_compress, dense_ffn_l0
  TOLERANT:  routed_exps_down, attn_output
  PROTECT:   shared_expert, mla_kv_decompress, mla_q, token_embd, output_head

C1 reference (11.32 GB): Q4_K_M base + DEMOTABLE→Q2_K, TOLERANT→Q3_K.
The XXS candidates push below C1 by using IQ2-class *base* types and/or
deeper expert demotion, while PROTECT groups are lifted above the base.

  XXS-A (~8.8 GB) — IQ2_M base, PROTECT lifted to Q4_K / Q5_K
    IQ2_M (~2.67 bpw) cuts the expert mass substantially vs Q2_K (~2.56 bpw
    average in Q4_K_M base). PROTECT lifted to Q4_K (KV decompress/Q) and Q5_K
    (shared expert, embeddings, output head) so sacred tensors stay safe.
    Rationale: the smallest step below C1 that still respects every PROTECT
    verdict; good chance of surviving PPL within C2 territory.

  XXS-B (~8.3 GB) — IQ2_XS base, PROTECT lifted to Q4_K / Q5_K, down→IQ2_M
    IQ2_XS (~2.31 bpw) goes one notch deeper. routed_exps_down (TOLERANT,
    +4.16% wiki max) gets IQ2_M — slightly above the base so the down path
    has a small cushion vs the gate/up path. PROTECT lifts match XXS-A.
    Rationale: tests whether the IQ2_XS base + cushioned down break the PPL
    budget or stay tolerable.

  XXS-C (~7.9 GB) — IQ2_XXS base, PROTECT lifted to Q4_K / Q5_K, down→IQ2_XS
    IQ2_XXS (~2.06 bpw) is the floor. Deepest probe — may PPL-collapse, but
    that is the data point for the floor exploration line. PROTECT lifts keep
    shared_expert / token_embd / output_head at Q5_K regardless.
    Rationale: maps the extreme floor; if PPL survives within ~15% of C1 wiki
    this becomes a genuine "8 GB card" candidate.

PROTECT lift logic (all three candidates share this):
  ^blk\\.\\d+\\.ffn_(gate|up|down)_shexp\\.weight$=Q5_K   (shared expert)
  ^blk\\.\\d+\\.attn_[kv]_b\\.weight$=Q4_K                (MLA KV decompress)
  ^blk\\.\\d+\\.attn_q_[ab]\\.weight$=Q4_K                (MLA Q)
  ^token_embd\\.weight$=Q5_K                               (embedding)
  ^output\\.weight$=Q5_K                                   (output head)

SEQUENCING
----------
This script must NOT run while the north campaign is active (shared GPU
account).  Use launch_xxs_when_north_done.sh to gate on north v1 completion.

BUDGET: hard $5.00 for the XXS stage.
  build ~$0.10 each (CPU: 12 cores, 32 GiB, ~5-7 min)
  PPL ~$0.20-0.35 each (T4: all candidates < 12.5 GB)
  driver + overhead ~$0.10
  3 builds + 3 PPLs ≈ $0.30 + $0.75 + $0.10 = ~$1.15  well inside $5.

LAUNCH (detached — do NOT launch manually; use the gated launcher):
  cd /var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/modal_harness
  nohup setsid modal run --detach cerebellum_flash_xxs.py::launch_xxs \
      > ../../cerebellum-glm47-flash/logs/modal_xxs_launch.log 2>&1 &

MONITOR:
  modal app logs cerebellum-flash-campaign
  modal volume ls cerebellum-flash results/xxs
  modal volume get --force cerebellum-flash results/xxs/ \
      ../../cerebellum-glm47-flash/modal_results/results/xxs/
"""

import json
import os
import re
import time

import modal

from cerebellum_modal import CUDA_IMAGE_REF, LLAMA_CPP_TAG
from cerebellum_flash_campaign import (
    DOMAINS,
    RATES,
    V,
    app,
    quantize_vol,
    vol,
)

# ---------------------------------------------------------------------------
# Names / paths
# ---------------------------------------------------------------------------
XXS_DIR = f"{V}/results/xxs"
WIKITEST = "wiki.test.raw"
CORPORA_DIR = f"{V}/corpora"

# C1 phase-3 measured PPLs (reference anchors for the summary table)
C1_PPL = {
    "wiki": 9.4454,
    "code": 3.7114,
    "math": 2.9310,
    "dialogue": 3.6417,
}
C1_GB = 11.32

# Q3_K_M uniform baseline PPLs (from baselines.json)
Q3KM_PPL = {
    "wiki": 10.7152,
    "code": 4.0065,
    "math": 3.9060,
    "dialogue": 3.8210,
}
Q3KM_GB = 14.38

# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

# PROTECT lifts shared by all three candidates.
# Placed FIRST in the override file so they take priority over the base type.
_PROTECT_LIFTS = "\n".join([
    # shared expert: PROTECT, raise above IQ2 base
    r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$=Q5_K",  # shared expert
    # MLA KV decompress: PROTECT, raise above IQ2 base
    r"^blk\.\d+\.attn_[kv]_b\.weight$=Q4_K",              # MLA KV decompress
    # MLA Q: PROTECT, raise above IQ2 base
    r"^blk\.\d+\.attn_q_[ab]\.weight$=Q4_K",               # MLA Q
    # embedding + output head: PROTECT, always
    r"^token_embd\.weight$=Q5_K",                           # embedding
    r"^output\.weight$=Q5_K",                               # output head
])

CANDIDATES = [
    {
        "name": "xxs_a",
        "label": "XXS-A",
        "base": "IQ2_M",
        "est_gb": 8.8,
        # PROTECT lifts + no further demotions needed (base already covers DEMOTABLE/TOLERANT)
        "override": _PROTECT_LIFTS,
        "rel": "candidates/xxs/glm47-flash-xxs-a.gguf",
        "rationale": "IQ2_M base (~2.67 bpw) + PROTECT lifted Q4_K/Q5_K; "
                     "smallest safe step below C1, respects all PROTECT verdicts",
    },
    {
        "name": "xxs_b",
        "label": "XXS-B",
        "base": "IQ2_XS",
        "est_gb": 8.3,
        # IQ2_XS base; routed_exps_down (TOLERANT) gets IQ2_M cushion above base;
        # PROTECT lifts as above
        "override": "\n".join([
            _PROTECT_LIFTS,
            # down experts: TOLERANT (+4.16% wiki), bumped one notch above base
            r"^blk\.\d+\.ffn_down_exps\.weight$=IQ2_M",
        ]),
        "rel": "candidates/xxs/glm47-flash-xxs-b.gguf",
        "rationale": "IQ2_XS base (~2.31 bpw) + PROTECT lifted Q4_K/Q5_K + "
                     "down_exps→IQ2_M cushion (TOLERANT +4% wiki); "
                     "tests one notch deeper than XXS-A",
    },
    {
        "name": "xxs_c",
        "label": "XXS-C",
        "base": "IQ2_XXS",
        "est_gb": 7.9,
        # IQ2_XXS base (~2.06 bpw); down_exps get IQ2_XS cushion; PROTECT lifts as above
        "override": "\n".join([
            _PROTECT_LIFTS,
            # down experts: TOLERANT, one notch above IQ2_XXS base
            r"^blk\.\d+\.ffn_down_exps\.weight$=IQ2_XS",
        ]),
        "rel": "candidates/xxs/glm47-flash-xxs-c.gguf",
        "rationale": "IQ2_XXS base (~2.06 bpw) + PROTECT lifted Q4_K/Q5_K + "
                     "down_exps→IQ2_XS; floor probe — may PPL-collapse, "
                     "but maps the hard lower bound for 8 GB card viability",
    },
]

# ---------------------------------------------------------------------------
# Cost model (inherits RATES from cerebellum_flash_campaign)
# ---------------------------------------------------------------------------
# T4-all-in rate reused for PPL (all XXS candidates < 12.5 GB → T4)
_PPL_RATE = RATES["ppl_t4"]
_QUANT_RATE = RATES["quantize"]
_DRIVER_RATE = RATES["driver"]

XXS_BUDGET_USD = 5.00

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
ppl_image = (
    modal.Image.from_registry(CUDA_IMAGE_REF, add_python="3.11")
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source(
        "cerebellum_modal", "cerebellum_flash_campaign", "cerebellum_flash_xxs"
    )
)

xxs_driver_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "cerebellum_modal", "cerebellum_flash_campaign", "cerebellum_flash_xxs"
)

# ---------------------------------------------------------------------------
# Helpers
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


def _wait_for_vol_file(path: str, min_bytes: float, tries: int = 40, sleep_s: int = 15):
    """Volume write-race fix (Flash lesson): reload + backoff before declaring missing."""
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
# PPL function (T4 — all XXS candidates fit comfortably)
# ---------------------------------------------------------------------------


@app.function(
    image=ppl_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=3600,
    retries=0,
    volumes={V: vol},
)
def ppl_xxs(filename: str, corpora: list) -> dict:
    """Run llama-perplexity on a Volume GGUF against the given corpora list.

    No llama-server involved here — no open-file-on-volume risk from the
    2026-06-12 bench bug.  Each llama-perplexity call is a blocking
    subprocess.run that exits cleanly before the next vol.reload().
    """
    import subprocess

    binary = next(
        (c for c in ("/app/llama-perplexity", "/llama-perplexity",
                     "/usr/local/bin/llama-perplexity")
         if os.path.exists(c)), None)
    if binary is None:
        found = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "llama-perplexity", "-type", "f"],
            capture_output=True, text=True).stdout.split()
        if not found:
            raise RuntimeError("llama-perplexity not found in CUDA image")
        binary = found[0]

    model_path = f"{V}/{filename}"
    _wait_for_vol_file(model_path, 5e9)

    results: dict = {}
    timings: dict = {}
    for corpus_name in corpora:
        corpus_path = f"{CORPORA_DIR}/{corpus_name}"
        vol.reload()
        if not os.path.exists(corpus_path):
            raise RuntimeError(f"corpus not on volume: {corpus_path}")
        cmd = [binary, "--model", model_path, "--file", corpus_path,
               "-ngl", "99", "--ctx-size", "2048", "--chunks", "-1"]
        t0 = time.time()
        print("+ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
        output = proc.stdout + proc.stderr
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", output)
        if not m:
            print("\n".join(output.splitlines()[-30:]), flush=True)
            raise RuntimeError(
                f"PPL parse failed for {filename} on {corpus_name} "
                f"(rc={proc.returncode})"
            )
        results[corpus_name] = float(m.group(1))
        timings[corpus_name] = round(time.time() - t0, 1)
        print(f"  {corpus_name}: PPL {results[corpus_name]:.4f} "
              f"in {timings[corpus_name]:.0f}s", flush=True)
    return {"ppl": results, "secs": timings, "llama_cpp_tag": LLAMA_CPP_TAG}


# ---------------------------------------------------------------------------
# Driver (runs ON MODAL — survives local death)
# ---------------------------------------------------------------------------


@app.function(
    image=xxs_driver_image,
    cpu=1.0,
    memory=2048,
    timeout=int(6 * 3600),
    retries=0,
    volumes={V: vol},
)
def xxs_driver(budget_usd: float = XXS_BUDGET_USD) -> dict:
    t_driver = time.time()
    os.makedirs(XXS_DIR, exist_ok=True)
    ledger_path = f"{XXS_DIR}/xxs_ledger.json"
    ledger = _load_json(ledger_path, {"entries": [], "total_usd": 0.0})

    def log(msg: str):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(f"{XXS_DIR}/xxs_progress.log", "a") as f:
            f.write(line + "\n")
        vol.commit()

    def add_cost(kind: str, secs: float, note: str = ""):
        rate = {
            "quantize": _QUANT_RATE,
            "ppl_t4": _PPL_RATE,
            "driver": _DRIVER_RATE,
        }.get(kind, _DRIVER_RATE)
        usd = rate * secs
        ledger["entries"].append({
            "kind": kind, "secs": round(secs, 1),
            "usd": round(usd, 4), "note": note,
        })
        ledger["total_usd"] = round(sum(e["usd"] for e in ledger["entries"]), 4)
        _dump_json(ledger_path, ledger)
        vol.commit()
        return usd

    def gate(est: float, what: str):
        driver_usd = _DRIVER_RATE * (time.time() - t_driver)
        projected = ledger["total_usd"] + driver_usd + est
        if projected > budget_usd:
            log(f"BUDGET ABORT before {what}: "
                f"ledger ${ledger['total_usd']:.2f} + driver ${driver_usd:.2f} "
                f"+ est ${est:.2f} > ${budget_usd:.2f}")
            _dump_json(f"{XXS_DIR}/ABORTED.json", {"before": what, "ledger": ledger})
            vol.commit()
            raise SystemExit(f"budget abort before {what}")
        log(f"gate OK for {what}: ledger ${ledger['total_usd']:.2f} "
            f"+ est ${est:.2f} <= ${budget_usd:.2f}")

    log(f"=== XXS driver start (budget ${budget_usd:.2f}, "
        f"llama.cpp {LLAMA_CPP_TAG}) ===")
    log(f"candidates: {[c['name'] for c in CANDIDATES]}")

    # Calibration corpora (same 4-domain set used for all prior phases)
    cal_corpora = [f"cerebellum_calibration_{d}.txt" for d in DOMAINS]
    all_corpora = cal_corpora + [WIKITEST]
    vol.reload()
    missing = [c for c in all_corpora if not os.path.exists(f"{CORPORA_DIR}/{c}")]
    if missing:
        log(f"FATAL: corpora missing from volume: {missing}")
        raise SystemExit("missing corpora")

    # Idempotency: load prior results if any
    results_path = f"{XXS_DIR}/xxs_results.json"
    results = _load_json(results_path, {})

    # ---- 1. Builds — spawn all three CPU quantize jobs in parallel ----
    # Est. $0.10 each x 3 = $0.30
    gate(0.45, "builds")
    os.makedirs(f"{V}/candidates/xxs", exist_ok=True)

    build_handles: dict = {}
    for cand in CANDIDATES:
        name = cand["name"]
        if results.get(name, {}).get("build"):
            log(f"build {name}: skipped (already on volume)")
            continue
        build_handles[name] = (
            quantize_vol.spawn(cand["rel"], cand["base"], cand["override"]),
            time.time(),
        )
        log(f"spawned build {name} ({cand['label']}) base={cand['base']} "
            f"est_gb={cand['est_gb']}")

    for name, (handle, t0) in build_handles.items():
        try:
            b = handle.get()
            add_cost("quantize", time.time() - t0, f"build {name}")
            results.setdefault(name, {})["build"] = b
            gb = b["bytes"] / 1e9
            log(f"build {name} done: {gb:.2f} GB "
                f"(skipped={b.get('skipped', False)})")
            # Sanity: converting_counts should show IQ2/Q4/Q5 conversions
            conv = b.get("converting_counts", {})
            if not b.get("skipped") and not conv:
                log(f"WARNING {name}: no converting_counts — "
                    f"override may not have applied (check fallback_lines)")
        except Exception as e:
            add_cost("quantize", time.time() - t0, f"build {name} FAILED")
            log(f"build {name} FAILED: {e}")
            results.setdefault(name, {})["build_error"] = str(e)
        _dump_json(results_path, results)
        vol.commit()

    # ---- 2. PPL screen — sequential, one T4 at a time ----
    # Est. ~$0.25-0.35 per candidate (T4, 5 corpora, ~90-120s each)
    for cand in CANDIDATES:
        name = cand["name"]
        build = results.get(name, {}).get("build")
        if not build:
            log(f"PPL {name}: skipping (no successful build)")
            continue
        if results.get(name, {}).get("ppl"):
            log(f"PPL {name}: skipping (already measured)")
            continue

        nbytes = build["bytes"]
        est_ppl = max(0.30, nbytes / 1e9 * 0.035)  # ~$0.035 per GB on T4
        gate(est_ppl, f"ppl {name}")

        t0 = time.time()
        try:
            r = ppl_xxs.remote(cand["rel"], all_corpora)
            add_cost("ppl_t4", time.time() - t0, f"ppl {name}")
            results[name]["ppl"] = r["ppl"]
            results[name]["ppl_secs"] = r["secs"]
            ppl_d = {d: r["ppl"][f"cerebellum_calibration_{d}.txt"] for d in DOMAINS}
            log(f"PPL {name}: {ppl_d}  wiki.test={r['ppl'].get(WIKITEST, '?')}")
        except Exception as e:
            add_cost("ppl_t4", time.time() - t0, f"ppl {name} FAILED")
            log(f"PPL {name} FAILED: {e}")
            results[name]["ppl_error"] = str(e)
        _dump_json(results_path, results)
        vol.commit()

    # ---- 3. Summary ----
    add_cost("driver", time.time() - t_driver, "driver wall time")
    vol.reload()
    summary = _render_xxs_summary(results, ledger)
    with open(f"{XXS_DIR}/xxs_summary.md", "w") as f:
        f.write(summary)
    _dump_json(results_path, results)
    vol.commit()
    log(f"=== XXS driver complete. est spend ${ledger['total_usd']:.2f} — "
        f"summary at results/xxs/xxs_summary.md ===")
    print(summary, flush=True)
    return {"ledger_usd": ledger["total_usd"]}


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _render_xxs_summary(results: dict, ledger: dict) -> str:
    lines = [
        "# GLM-4.7-Flash XXS stage — IQ2-class extreme-small PPL screen",
        "",
        f"Generated {time.strftime('%F %T')} on Modal (llama.cpp {LLAMA_CPP_TAG}, T4).",
        "",
        "No benchmarks here — bench gates (HumanEval+/ARC/HellaSwag/MMLU + "
        "BigCodeBench) run locally after download per the no-ship rule.",
        "",
        "## PPL comparison vs C1 and uniform Q3_K_M",
        "",
    ]

    # Build header from domains
    domain_headers = " | ".join(d for d in DOMAINS) + " | wikitext"
    lines += [
        f"| build | size GB | {domain_headers} | status |",
        "|---|---|" + "---|" * (len(DOMAINS) + 2),
    ]

    def _ppl_row(label: str, gb: float, ppl_map: dict, status: str) -> str:
        d_vals = " | ".join(
            f"{ppl_map[f'cerebellum_calibration_{d}.txt']:.4f}"
            if f"cerebellum_calibration_{d}.txt" in ppl_map else "-"
            for d in DOMAINS
        )
        wiki_val = f"{ppl_map.get(WIKITEST, float('nan')):.4f}" if WIKITEXT in ppl_map else "-"
        return f"| {label} | {gb:.2f} | {d_vals} | {wiki_val} | {status} |"

    # Reference rows
    WIKITEXT = WIKITEST
    lines.append(_ppl_row("uniform Q3_K_M (baseline)", Q3KM_GB, {
        "cerebellum_calibration_wiki.txt": Q3KM_PPL["wiki"],
        "cerebellum_calibration_code.txt": Q3KM_PPL["code"],
        "cerebellum_calibration_math.txt": Q3KM_PPL["math"],
        "cerebellum_calibration_dialogue.txt": Q3KM_PPL["dialogue"],
        WIKITEST: 9.8012,
    }, "reference"))
    lines.append(_ppl_row("C1 cerebellum (11.32 GB)", C1_GB, {
        "cerebellum_calibration_wiki.txt": C1_PPL["wiki"],
        "cerebellum_calibration_code.txt": C1_PPL["code"],
        "cerebellum_calibration_math.txt": C1_PPL["math"],
        "cerebellum_calibration_dialogue.txt": C1_PPL["dialogue"],
    }, "reference (phase 3)"))

    # XXS candidate rows
    for cand in CANDIDATES:
        name = cand["name"]
        rec = results.get(name, {})
        build = rec.get("build")
        ppl_data = rec.get("ppl")
        if build and ppl_data:
            gb = build["bytes"] / 1e9
            lines.append(_ppl_row(
                f"{cand['label']} ({cand['base']} base)",
                gb, ppl_data, "measured"))
        elif rec.get("ppl_error"):
            gb = build["bytes"] / 1e9 if build else cand["est_gb"]
            lines.append(
                f"| {cand['label']} ({cand['base']} base) | {gb:.2f} | "
                + " | ".join("-" for _ in DOMAINS)
                + f" | - | PPL FAILED: {rec['ppl_error'][:60]} |")
        elif rec.get("build_error"):
            lines.append(
                f"| {cand['label']} ({cand['base']} base) | est {cand['est_gb']:.1f} | "
                + " | ".join("-" for _ in DOMAINS)
                + f" | - | BUILD FAILED: {rec['build_error'][:60]} |")
        else:
            lines.append(
                f"| {cand['label']} ({cand['base']} base) | est {cand['est_gb']:.1f} | "
                + " | ".join("-" for _ in DOMAINS)
                + " | - | NOT RUN |")

    # ΔPPL% vs C1
    lines += [
        "",
        "## ΔPPL% vs C1 cerebellum (11.32 GB, reference)",
        "",
        "| build | Δ%wiki | Δ%code | Δ%math | Δ%dialogue |",
        "|---|---|---|---|---|",
    ]
    for cand in CANDIDATES:
        name = cand["name"]
        rec = results.get(name, {})
        ppl_data = rec.get("ppl")
        if not ppl_data:
            lines.append(f"| {cand['label']} | - | - | - | - |")
            continue
        row_parts = []
        for d in DOMAINS:
            key = f"cerebellum_calibration_{d}.txt"
            if key in ppl_data and C1_PPL.get(d):
                delta_pct = 100.0 * (ppl_data[key] / C1_PPL[d] - 1.0)
                row_parts.append(f"{delta_pct:+.2f}")
            else:
                row_parts.append("-")
        lines.append(f"| {cand['label']} | " + " | ".join(row_parts) + " |")

    # Candidate override summary
    lines += [
        "",
        "## Candidate overrides",
        "",
    ]
    for cand in CANDIDATES:
        rec = results.get(cand["name"], {})
        build = rec.get("build", {})
        conv = build.get("converting_counts", {})
        gb_actual = f"{build['bytes']/1e9:.2f} GB" if build else f"est {cand['est_gb']:.1f} GB"
        lines += [
            f"### {cand['label']} — {cand['base']} base ({gb_actual})",
            "",
            f"{cand['rationale']}",
            "",
            "Override file (protect lifts first, then any per-candidate demotions):",
            "```",
            cand["override"],
            "```",
        ]
        if conv:
            lines.append(f"Observed converting_counts: {conv}")
        fallbacks = build.get("fallback_lines", [])
        if fallbacks:
            lines.append(f"Fallback lines ({len(fallbacks)}): {fallbacks[:3]}")
        lines.append("")

    # Cost ledger
    lines += [
        "## Cost ledger (XXS stage, list-price estimates)",
        "",
        "| kind | secs | usd | note |",
        "|---|---|---|---|",
    ]
    for e in ledger.get("entries", []):
        lines.append(f"| {e['kind']} | {e['secs']} | ${e['usd']:.4f} | {e['note']} |")
    lines.append(f"| **total** | | **${ledger.get('total_usd', 0):.2f}** | |")

    lines += [
        "",
        "## Next steps",
        "",
        "1. Download best-PPL candidate(s): "
        "`modal volume get --force cerebellum-flash results/xxs/ .../modal_results/results/xxs/`",
        "2. Run local bench gates (HumanEval+/ARC/HellaSwag/MMLU + BigCodeBench) "
        "vs same-size uniform Q2_K baseline — no ship without passing.",
        "3. Human wrong-answer audit before recording any score.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def launch_xxs(budget_usd: float = XXS_BUDGET_USD):
    fc = xxs_driver.spawn(budget_usd=budget_usd)
    print(f"XXS driver spawned: {fc.object_id}")
    print("monitor:  modal app logs cerebellum-flash-campaign")
    print("results:  modal volume ls cerebellum-flash results/xxs")
    print("sync:     modal volume get --force cerebellum-flash results/xxs/ "
          "../../cerebellum-glm47-flash/modal_results/results/xxs/")
