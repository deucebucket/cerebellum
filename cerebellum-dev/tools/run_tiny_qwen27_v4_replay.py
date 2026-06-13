#!/usr/bin/env python3
"""Run a Qwen 3.6 27B v4-shaped sparse probe replay on tiny Qwen3-0.6B.

This is private research automation. It intentionally mirrors the old sparse
probe -> ablation_results.json -> osmosis.cerebellum allocator path.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/v4-sparse-replay")
SOURCE = Path("/var/home/deucebucket/games/models/Qwen3-0.6B-f16.gguf")
IMATRIX = Path("/var/home/deucebucket/games/cerebellum-runs/recreate-qwen36-v4/qwen3-0.6b-og-method-20260607/imatrix/cerebellum_imatrix_ncall8.dat")
CORPUS = Path("/var/home/deucebucket/games/osmosis-quants/wiki.test.raw")
QUANTIZE = Path("/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-quantize")
PERPLEXITY = Path("/var/home/deucebucket/ai-drive/llama.cpp/build/bin/llama-perplexity")
BASELINE_PPL = 19.7289

PROBES = [
    ("layer_0.mlp.up_proj", "blk.0.ffn_up.weight", "early_ffn_up_low_kl"),
    ("layer_0.mlp.down_proj", "blk.0.ffn_down.weight", "early_ffn_down_control"),
    ("layer_1.mlp.gate_proj", "blk.1.ffn_gate.weight", "early_gate_low_kl"),
    ("layer_2.mlp.gate_proj", "blk.2.ffn_gate.weight", "early_gate_low_kl"),
    ("layer_5.mlp.gate_proj", "blk.5.ffn_gate.weight", "early_mid_gate_low_kl"),
    ("layer_7.self_attn.q_proj", "blk.7.attn_q.weight", "early_attention_role"),
    ("layer_7.self_attn.k_proj", "blk.7.attn_k.weight", "early_attention_role"),
    ("layer_7.self_attn.v_proj", "blk.7.attn_v.weight", "early_attention_role"),
    ("layer_7.self_attn.o_proj", "blk.7.attn_output.weight", "early_attention_role"),
    ("layer_10.mlp.up_proj", "blk.10.ffn_up.weight", "mid_ffn_up"),
    ("layer_10.mlp.gate_proj", "blk.10.ffn_gate.weight", "mid_ffn_gate"),
    ("layer_10.mlp.down_proj", "blk.10.ffn_down.weight", "mid_ffn_down"),
    ("layer_13.mlp.down_proj", "blk.13.ffn_down.weight", "mid_ffn_down_control"),
    ("layer_18.self_attn.q_proj", "blk.18.attn_q.weight", "late_attention_role"),
    ("layer_18.self_attn.k_proj", "blk.18.attn_k.weight", "late_attention_role"),
    ("layer_18.self_attn.v_proj", "blk.18.attn_v.weight", "late_attention_role"),
    ("layer_18.self_attn.o_proj", "blk.18.attn_output.weight", "late_attention_role"),
    ("layer_20.mlp.down_proj", "blk.20.ffn_down.weight", "late_ffn_down"),
    ("layer_25.mlp.gate_proj", "blk.25.ffn_gate.weight", "tail_gate"),
    ("layer_25.mlp.down_proj", "blk.25.ffn_down.weight", "tail_ffn_down"),
    ("layer_27.self_attn.q_proj", "blk.27.attn_q.weight", "tail_attention_q"),
    ("layer_27.self_attn.v_proj", "blk.27.attn_v.weight", "tail_attention_v"),
    ("layer_27.mlp.down_proj", "blk.27.ffn_down.weight", "tail_ffn_down"),
]


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.returncode


def distrobox(cmd: list[str]) -> list[str]:
    return ["distrobox", "enter", "ai", "--", *map(str, cmd)]


def parse_ppl(log: Path) -> tuple[float | None, float | None]:
    text = log.read_text(errors="ignore")
    matches = re.findall(r"Final estimate:\s+PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if not matches:
        matches = re.findall(r"PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if not matches:
        return None, None
    ppl, err = matches[-1]
    return float(ppl), float(err)


def write_plan() -> None:
    plan = {
        "tensors": [
            {
                "name": hf_name,
                "gguf_tensor": gguf,
                "reason": reason,
            }
            for hf_name, gguf, reason in PROBES
        ],
        "baseline": BASELINE_PPL,
        "notes": "Tiny Qwen3-0.6B sparse replay of Qwen3.6-27B v4 probe/allocator path.",
    }
    (ROOT / "ablation_plan_tiny.json").write_text(json.dumps(plan, indent=2) + "\n")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / "artifacts"
    logs = ROOT / "logs"
    type_dir = ROOT / "tensor_types"
    artifacts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    type_dir.mkdir(exist_ok=True)
    write_plan()

    results_path = ROOT / "ablation_results_tiny.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {"baseline_ppl": BASELINE_PPL, "tests": {}, "schema": "cerebellum.qwen27_v4_sparse_replay_tiny.v1"}

    for idx, (hf_name, gguf_tensor, reason) in enumerate(PROBES, 1):
        if hf_name in results["tests"] and results["tests"][hf_name].get("ppl") is not None:
            print(f"[{idx}/{len(PROBES)}] skip {gguf_tensor}: already measured")
            continue

        safe = gguf_tensor.replace(".", "_")
        type_file = type_dir / f"{safe}_q2.txt"
        out = artifacts / f"{safe}_q2.gguf"
        qlog = logs / f"{safe}_quant.log"
        plog = logs / f"{safe}_ppl.log"
        type_file.write_text(f"{gguf_tensor}=q2_K\n")

        print(f"[{idx}/{len(PROBES)}] quant {gguf_tensor} -> q2_K", flush=True)
        qcmd = distrobox([
            QUANTIZE,
            "--allow-requantize",
            "--imatrix",
            IMATRIX,
            "--tensor-type-file",
            type_file,
            SOURCE,
            out,
            "Q4_K_M",
        ])
        qrc = run(qcmd, qlog)
        if qrc != 0:
            results["tests"][hf_name] = {
                "gguf_tensor": gguf_tensor,
                "reason": reason,
                "ppl": None,
                "error": f"quantize failed rc={qrc}",
            }
            results_path.write_text(json.dumps(results, indent=2) + "\n")
            print(f"  quant failed rc={qrc}", flush=True)
            continue

        print(f"[{idx}/{len(PROBES)}] ppl {gguf_tensor}", flush=True)
        pcmd = distrobox([
            PERPLEXITY,
            "--model",
            out,
            "--ctx-size",
            "2048",
            "-f",
            CORPUS,
            "-ngl",
            "99",
            "--chunks",
            "32",
        ])
        start = time.time()
        prc = run(pcmd, plog)
        ppl, err = parse_ppl(plog)
        elapsed = time.time() - start
        results["tests"][hf_name] = {
            "gguf_tensor": gguf_tensor,
            "reason": reason,
            "ppl": ppl,
            "ppl_error": err,
            "delta": None if ppl is None else ppl - BASELINE_PPL,
            "quant_log": str(qlog),
            "ppl_log": str(plog),
            "candidate": str(out),
            "ppl_returncode": prc,
            "elapsed_seconds": elapsed,
        }
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"  ppl={ppl} err={err} delta={None if ppl is None else ppl - BASELINE_PPL:+.4f}", flush=True)
        if out.exists():
            out.unlink()

    print(f"Wrote {results_path}")
    shutil.copy2(results_path, ROOT / "ablation_results.json")
    shutil.copy2(ROOT / "ablation_plan_tiny.json", ROOT / "ablation_plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
