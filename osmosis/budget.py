"""Budget-constrained bit allocator for Osmosis.

Given a multi-depth sensitivity report and a target file size in GB, finds the
optimal per-tensor quantization type that maximizes quality within the budget.
Outputs a tensor-type file for llama-quantize.

Uses KL divergence curves from sensitivity_multi.py to make informed staggering
decisions — each tensor gets the cheapest quant level that keeps damage below
threshold, then remaining budget upgrades the most sensitive tensors further.

Usage:
    python -m osmosis.budget \
        --sensitivity osmosis-qwen3.5-9b/sensitivity_multi.json \
        --source-gguf Qwen3.5-9B-Q5_K_M.gguf \
        --budget-gb 4.0 \
        --output tensor_types.txt
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


QUANT_BPW = {
    "q2_K": 2.5625,
    "q3_K": 3.4375,
    "q4_K": 4.5,
    "q5_K": 5.5,
    "q6_K": 6.5625,
    "q8_0": 8.5,
    "f16":  16.0,
    "f32":  32.0,
}

QUANT_TIERS = ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_0", "f16"]

# Map quant tiers to the bit depths tested in sensitivity_multi
TIER_TO_DEPTH = {
    "q2_K": 2, "q3_K": 3, "q4_K": 4, "q5_K": 4,
    "q6_K": 6, "q8_0": 8, "f16": 8,
}


def estimate_tensor_size(param_count, qtype):
    bpw = QUANT_BPW.get(qtype, 4.5)
    return int(param_count * bpw / 8)


def parse_gguf_tensors(gguf_path, quantize_bin=None):
    if quantize_bin is None:
        for candidate in [
            Path.home() / "ai-drive/llama.cpp/build/bin/llama-quantize",
            Path.home() / "ai-drive/llama.cpp/build-cpu/bin/llama-quantize",
            Path.home() / "ai-drive/llama-prismml/build/bin/llama-quantize",
        ]:
            if candidate.exists():
                quantize_bin = str(candidate)
                break

    if not quantize_bin:
        raise FileNotFoundError("llama-quantize not found")

    result = subprocess.run(
        [quantize_bin, "--dry-run", gguf_path, "/dev/null", "q4_K_M"],
        capture_output=True, text=True, timeout=60,
    )

    tensors = {}
    output = result.stderr + result.stdout
    for line in output.split("\n"):
        match = re.match(
            r"\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[\s*([\d,\s]+)\],\s+type\s+=\s+(\S+),\s+size\s+=\s+([\d.]+)\s+MiB",
            line,
        )
        if match:
            name = match.group(1)
            dims = [int(d.strip()) for d in match.group(2).split(",")]
            dims = [d for d in dims if d > 1]
            current_type = match.group(3)
            size_mib = float(match.group(4))
            param_count = 1
            for d in dims:
                param_count *= d
            tensors[name] = {
                "name": name,
                "dims": dims,
                "param_count": param_count,
                "current_type": current_type,
                "current_size_bytes": int(size_mib * 1024 * 1024),
            }

    return tensors


# HF sensitivity name → GGUF tensor name mapping
HF_TO_GGUF_COMPONENT = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.in_proj_z": "ssm_in",
    "linear_attn.out_proj": "ssm_out",
    "linear_attn.conv1d": "ssm_conv1d",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt",
    "linear_attn.norm": "ssm_norm",
    "linear_attn.attn_gate": "attn_gate",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "mlp.down_proj": "ffn_down",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
}

GGUF_TO_HF = {v: k for k, v in HF_TO_GGUF_COMPONENT.items()}


def build_multi_sensitivity_map(report_path):
    """Load multi-depth sensitivity report. Returns name → {depths: {bit: kl}, kl_q2: float}."""
    with open(report_path) as f:
        report = json.load(f)

    groups = report.get("groups", [])
    sens = {}
    for g in groups:
        depths = g.get("depths", {})
        kl_by_depth = {}
        for bit_str, metrics in depths.items():
            kl_by_depth[int(bit_str)] = metrics["kl_mean"]

        sens[g["name"]] = {
            "depths": kl_by_depth,
            "kl_q2": kl_by_depth.get(2, 0),
            "group_type": g.get("group_type", ""),
            "param_count": g.get("param_count", 0),
        }
    return sens, report


def build_legacy_sensitivity_map(report_path):
    """Load old single-depth sensitivity report (cosine_sim based)."""
    with open(report_path) as f:
        report = json.load(f)
    groups = report.get("groups", report if isinstance(report, list) else [])
    sens = {}
    for g in groups:
        sens[g["name"]] = g.get("cosine_sim", g.get("kl_divergence", 0))
    return sens


def match_tensor_to_sensitivity(tensor_name, sensitivity_map):
    """Find multi-depth sensitivity data for a GGUF tensor name."""
    match = re.match(r"blk\.(\d+)\.(\S+?)(?:\.weight)?$", tensor_name)
    if not match:
        return None

    layer_num = match.group(1)
    gguf_comp = match.group(2)

    hf_comp = GGUF_TO_HF.get(gguf_comp)
    if hf_comp:
        sens_key = f"layer_{layer_num}.{hf_comp}"
        if sens_key in sensitivity_map:
            return sensitivity_map[sens_key]

    return None


def pick_minimum_tier(kl_depths, kl_threshold=0.01):
    """Pick the cheapest quant tier that keeps KL below threshold."""
    for tier_idx, tier in enumerate(QUANT_TIERS):
        depth = TIER_TO_DEPTH[tier]
        kl = kl_depths.get(depth, 0)
        if kl <= kl_threshold:
            return tier_idx
    return len(QUANT_TIERS) - 1  # f16 if nothing else works


def allocate_budget(tensors, sensitivity_map, budget_bytes, kl_threshold=0.01, verbose=False):
    """Multi-depth budget allocation using KL curves.

    Strategy:
    1. Fix immovable tensors (norms, tiny scalars) at their current type
    2. For each movable tensor, find the minimum tier that keeps KL < threshold
    3. If total exceeds budget, downgrade least-sensitive tensors
    4. If budget remains, upgrade most-sensitive tensors further
    """
    fixed = {}
    movable = {}

    for name, info in tensors.items():
        pc = info["param_count"]

        if pc < 100000 or "norm" in name or "ssm_a" == name.split(".")[-1] or "ssm_dt" in name:
            fixed[name] = info["current_type"]
            continue

        if name in ("output.weight", "token_embd.weight"):
            fixed[name] = "q4_K"
            continue

        sens_data = match_tensor_to_sensitivity(name, sensitivity_map)
        if sens_data is None:
            fixed[name] = info["current_type"]
            continue

        kl_depths = sens_data["depths"]
        min_tier = pick_minimum_tier(kl_depths, kl_threshold)

        movable[name] = {
            "param_count": pc,
            "kl_depths": kl_depths,
            "kl_q2": sens_data["kl_q2"],
            "group_type": sens_data["group_type"],
            "ideal_tier_idx": min_tier,
            "tier_idx": min_tier,
        }

    # Calculate fixed cost
    fixed_bytes = sum(
        estimate_tensor_size(tensors[n]["param_count"], qt)
        for n, qt in fixed.items()
    )

    remaining = budget_bytes - fixed_bytes
    if remaining <= 0:
        print(f"ERROR: Fixed tensors alone ({fixed_bytes/1e9:.2f} GB) exceed budget")
        return None

    # Calculate ideal cost (every tensor at its KL-minimum tier)
    ideal_cost = sum(
        estimate_tensor_size(m["param_count"], QUANT_TIERS[m["ideal_tier_idx"]])
        for m in movable.values()
    )

    if verbose:
        print(f"Fixed cost:   {fixed_bytes / 1e9:.2f} GB ({len(fixed)} tensors)")
        print(f"Movable:      {len(movable)} tensors")
        print(f"Ideal cost:   {(fixed_bytes + ideal_cost) / 1e9:.2f} GB (KL threshold={kl_threshold})")
        print(f"Budget:       {budget_bytes / 1e9:.2f} GB")

        ideal_dist = Counter(QUANT_TIERS[m["ideal_tier_idx"]] for m in movable.values())
        print(f"Ideal tiers:  {dict(sorted(ideal_dist.items()))}")

    total_ideal = fixed_bytes + ideal_cost

    if total_ideal <= budget_bytes:
        # Under budget — upgrade the most sensitive tensors with remaining room
        if verbose:
            print(f"\nUnder budget by {(budget_bytes - total_ideal)/1e9:.2f} GB — upgrading sensitive tensors")

        current_cost = total_ideal
        upgrades = 0

        upgrade_queue = []
        for name, m in movable.items():
            for tier_idx in range(m["ideal_tier_idx"] + 1, len(QUANT_TIERS)):
                cost_delta = (
                    estimate_tensor_size(m["param_count"], QUANT_TIERS[tier_idx])
                    - estimate_tensor_size(m["param_count"], QUANT_TIERS[tier_idx - 1])
                )
                priority = m["kl_q2"] / max(cost_delta, 1)
                upgrade_queue.append({
                    "name": name,
                    "from_tier": tier_idx - 1,
                    "to_tier": tier_idx,
                    "cost_delta": cost_delta,
                    "priority": priority,
                })

        upgrade_queue.sort(key=lambda x: x["priority"], reverse=True)

        for u in upgrade_queue:
            if movable[u["name"]]["tier_idx"] != u["from_tier"]:
                continue
            if current_cost + u["cost_delta"] <= budget_bytes:
                movable[u["name"]]["tier_idx"] = u["to_tier"]
                current_cost += u["cost_delta"]
                upgrades += 1

        if verbose:
            print(f"  {upgrades} bonus upgrades applied")

    else:
        # Over budget — downgrade least-sensitive tensors from their ideal
        if verbose:
            print(f"\nOver budget by {(total_ideal - budget_bytes)/1e9:.2f} GB — downgrading least-sensitive")

        current_cost = total_ideal
        downgrades = 0

        downgrade_queue = []
        for name, m in movable.items():
            for tier_idx in range(m["ideal_tier_idx"], 0, -1):
                cost_savings = (
                    estimate_tensor_size(m["param_count"], QUANT_TIERS[tier_idx])
                    - estimate_tensor_size(m["param_count"], QUANT_TIERS[tier_idx - 1])
                )
                pain = m["kl_q2"]
                downgrade_queue.append({
                    "name": name,
                    "from_tier": tier_idx,
                    "to_tier": tier_idx - 1,
                    "cost_savings": cost_savings,
                    "pain": pain,
                    "priority": cost_savings / max(pain, 1e-10),
                })

        # Best downgrades: most savings, least pain
        downgrade_queue.sort(key=lambda x: x["priority"], reverse=True)

        for d in downgrade_queue:
            if current_cost <= budget_bytes:
                break
            if movable[d["name"]]["tier_idx"] != d["from_tier"]:
                continue
            movable[d["name"]]["tier_idx"] = d["to_tier"]
            current_cost -= d["cost_savings"]
            downgrades += 1

        if verbose:
            print(f"  {downgrades} downgrades applied")

        if current_cost > budget_bytes:
            print(f"WARNING: Still over budget after all downgrades ({current_cost/1e9:.2f} GB > {budget_bytes/1e9:.2f} GB)")

    # Build final assignments
    assignments = {}
    for name, qtype in fixed.items():
        assignments[name] = qtype
    for name, m in movable.items():
        assignments[name] = QUANT_TIERS[m["tier_idx"]]

    current_cost = sum(
        estimate_tensor_size(tensors[n]["param_count"], qt)
        for n, qt in assignments.items()
    )

    type_counts = Counter(assignments.values())

    if verbose:
        print(f"\nFinal allocation ({current_cost / 1e9:.2f} GB / {budget_bytes / 1e9:.2f} GB budget):")
        for qtype in QUANT_TIERS + ["f32"]:
            if qtype in type_counts:
                print(f"  {qtype:6s}: {type_counts[qtype]:4d} tensors")
        print(f"  Budget utilization: {current_cost / budget_bytes * 100:.1f}%")

        # Show what layer 31 got (the sacred cow)
        l31 = {n: qt for n, qt in assignments.items() if n.startswith("blk.31.")}
        if l31:
            print(f"\n  Layer 31 (most sensitive):")
            for n, qt in sorted(l31.items()):
                print(f"    {n}: {qt}")

    return assignments


def write_tensor_types(assignments, output_path, tensors):
    lines = []
    for name, qtype in sorted(assignments.items()):
        if qtype == "f32":
            continue
        lines.append(f"{name}={qtype}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} tensor type assignments to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Osmosis budget allocator — fit a model to your VRAM budget"
    )
    parser.add_argument("--sensitivity", required=True, help="Sensitivity report JSON (multi-depth or legacy)")
    parser.add_argument("--source-gguf", required=True, help="Source GGUF file to analyze")
    parser.add_argument("--budget-gb", type=float, required=True, help="Target file size in GB")
    parser.add_argument("--output", required=True, help="Output tensor types file")
    parser.add_argument("--kl-threshold", type=float, default=0.01,
                        help="Max KL divergence per tensor (default: 0.01)")
    parser.add_argument("--quantize-bin", default=None, help="Path to llama-quantize")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    print(f"Budget: {args.budget_gb:.1f} GB (KL threshold: {args.kl_threshold})")
    print(f"Source: {args.source_gguf}")

    sensitivity_map, report = build_multi_sensitivity_map(args.sensitivity)
    print(f"Loaded {len(sensitivity_map)} multi-depth sensitivity curves")

    tensors = parse_gguf_tensors(args.source_gguf, args.quantize_bin)
    print(f"Found {len(tensors)} tensors in GGUF")

    total_params = sum(t["param_count"] for t in tensors.values())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    budget_bytes = int(args.budget_gb * 1e9)
    assignments = allocate_budget(
        tensors, sensitivity_map, budget_bytes,
        kl_threshold=args.kl_threshold, verbose=True,
    )

    if assignments is None:
        sys.exit(1)

    write_tensor_types(assignments, args.output, tensors)

    print(f"\nTo build the GGUF:")
    print(f"  llama-quantize --allow-requantize \\")
    print(f"    --tensor-type-file {args.output} \\")
    print(f"    {args.source_gguf} \\")
    print(f"    output.gguf q2_K")


if __name__ == "__main__":
    main()
