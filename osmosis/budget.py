"""Budget-constrained bit allocator for Osmosis.

Given a multi-depth sensitivity report and a target file size in GB, enhances
llama-quantize's built-in allocation with targeted tensor promotions based on
actual sensitivity data. Outputs a tensor-type file for llama-quantize.

Strategy: llama-quantize's base quant types (Q2_K, Q3_K_M, etc.) already have
hardcoded per-tensor heuristics. We start from the best-fitting base type and
use sensitivity data to surgically promote the tensors that benefit most.

Usage:
    python -m osmosis.budget \
        --sensitivity osmosis-qwen36-27b/sensitivity_multi.json \
        --source-gguf qwen3.6-27b-f16.gguf \
        --budget-gb 12.0 \
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
}

PROMOTION_ORDER = ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_0"]

TIER_TO_DEPTH = {
    "q2_K": 2, "q3_K": 3, "q4_K": 4, "q5_K": 4,
    "q6_K": 6, "q8_0": 8,
}

BASE_QUANT_SIZES = [
    ("Q2_K", "q2_K"),
    ("Q3_K_S", "q3_K"),
    ("Q3_K_M", "q3_K"),
    ("Q4_K_S", "q4_K"),
    ("Q4_K_M", "q4_K"),
    ("Q5_K_S", "q5_K"),
    ("Q5_K_M", "q5_K"),
    ("Q6_K", "q6_K"),
    ("Q8_0", "q8_0"),
]

HF_TO_GGUF_COMPONENT = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "linear_attn.conv1d": "ssm_conv1d",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt",
    "linear_attn.norm": "ssm_norm",
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


def find_quantize_bin(quantize_bin=None):
    if quantize_bin:
        return quantize_bin
    for candidate in [
        Path("/tmp/llama-cpu-build/bin/llama-quantize"),
        Path.home() / "ai-drive/llama.cpp/build/bin/llama-quantize",
        Path.home() / "ai-drive/llama.cpp/build-cpu/bin/llama-quantize",
        Path.home() / "ai-drive/llama-prismml/build/bin/llama-quantize",
    ]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("llama-quantize not found")


def dry_run_quantize(gguf_path, base_type, quantize_bin, imatrix=None):
    """Run llama-quantize --dry-run and parse its tensor allocation."""
    cmd = [quantize_bin, "--dry-run"]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [gguf_path, "/dev/null", base_type]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        stderr_snippet = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "no output"
        raise RuntimeError(f"llama-quantize --dry-run failed (exit {result.returncode}): {stderr_snippet}")

    tensors = {}
    total_size = 0
    output = result.stderr + result.stdout
    for line in output.split("\n"):
        if not line.startswith("["):
            continue

        # Format when converting: "size = X MiB -> Y MiB (target_type)"
        m_convert = re.match(
            r"\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[\s*([\d,\s]+)\],\s+type\s+=\s+(\S+),\s+"
            r"size\s+=\s+[\d.]+\s+MiB\s+->\s+([\d.]+)\s+MiB\s+\((\S+)\)",
            line,
        )
        # Format when keeping as-is: "size = X MiB"
        m_keep = re.match(
            r"\[\s*\d+/\s*\d+\]\s+(\S+)\s+- \[\s*([\d,\s]+)\],\s+type\s+=\s+(\S+),\s+"
            r"size\s+=\s+([\d.]+)\s+MiB$",
            line,
        )

        if m_convert:
            name = m_convert.group(1)
            dims = [int(d.strip()) for d in m_convert.group(2).split(",")]
            dims = [d for d in dims if d > 1]
            size_mib = float(m_convert.group(4))
            assigned_type = m_convert.group(5)
        elif m_keep:
            name = m_keep.group(1)
            dims = [int(d.strip()) for d in m_keep.group(2).split(",")]
            dims = [d for d in dims if d > 1]
            size_mib = float(m_keep.group(4))
            assigned_type = m_keep.group(3).rstrip(",")
        else:
            continue

        param_count = 1
        for d in dims:
            param_count *= d
        tensors[name] = {
            "name": name,
            "dims": dims,
            "param_count": param_count,
            "assigned_type": assigned_type,
            "size_bytes": int(size_mib * 1024 * 1024),
        }
        total_size += int(size_mib * 1024 * 1024)

    return tensors, total_size


def estimate_tensor_size(param_count, qtype):
    bpw = QUANT_BPW.get(qtype, 4.5)
    return int(param_count * bpw / 8)


def build_size_table(source_gguf, quantize_bin, imatrix=None, base_types=None):
    """Run dry_run_quantize at multiple quant levels and collect real sizes.

    Returns (size_table, dry_run_cache):
      size_table: {tensor_name: {normalized_qtype: real_size_bytes}}
      dry_run_cache: {base_label: (tensors_dict, total_size)}
    """
    if base_types is None:
        base_types = [label for label, _ in BASE_QUANT_SIZES]

    size_table = {}
    dry_run_cache = {}
    for base_label in base_types:
        tensors, total_size = dry_run_quantize(source_gguf, base_label, quantize_bin, imatrix)
        dry_run_cache[base_label] = (tensors, total_size)
        for name, tinfo in tensors.items():
            assigned = normalize_qtype(tinfo["assigned_type"])
            if name not in size_table:
                size_table[name] = {}
            if assigned not in size_table[name]:
                size_table[name][assigned] = tinfo["size_bytes"]
    return size_table, dry_run_cache


def lookup_tensor_size(size_table, tensor_name, param_count, qtype):
    """Look up real size from the table, fall back to BPW estimate."""
    normed = normalize_qtype(qtype)
    real = size_table.get(tensor_name, {}).get(normed)
    if real is not None:
        return real
    return estimate_tensor_size(param_count, normed)


def build_sensitivity_map(report_path):
    with open(report_path) as f:
        report = json.load(f)

    sens = {}
    for g in report.get("groups", []):
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
    return sens


def match_tensor_to_sensitivity(tensor_name, sensitivity_map):
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


def normalize_qtype(qtype):
    """Normalize quant type names: q2_k -> q2_K, Q2_K -> q2_K, etc."""
    s = qtype.strip().rstrip(",").lower()
    return re.sub(r'_([a-z])$', lambda m: '_' + m.group(1).upper(), s)


def next_tier(current_type):
    """Return the next higher K-quant tier, or None if already at max."""
    normed = normalize_qtype(current_type)
    try:
        idx = PROMOTION_ORDER.index(normed)
    except ValueError:
        return None
    if idx + 1 < len(PROMOTION_ORDER):
        return PROMOTION_ORDER[idx + 1]
    return None


def promotion_benefit(sens_data, current_type, target_type):
    """Compute quality benefit of promoting a tensor from current to target type."""
    current_depth = TIER_TO_DEPTH.get(normalize_qtype(current_type))
    target_depth = TIER_TO_DEPTH.get(normalize_qtype(target_type))
    if current_depth is None or target_depth is None:
        return 0

    kl_current = sens_data["depths"].get(current_depth, 0)
    kl_target = sens_data["depths"].get(target_depth, 0)
    return (kl_current - kl_target) * sens_data["param_count"]


def allocate_budget(source_gguf, sensitivity_map, budget_bytes, quantize_bin,
                    imatrix=None, verbose=False):
    """Enhance llama-quantize's allocation with targeted promotions.

    1. Find the best base quant type that fits the budget
    2. Get llama-quantize's default allocation via dry-run
    3. Spend remaining budget promoting the most sensitive tensors
    """
    # Build size table from dry-runs at all base types (includes imatrix effects)
    all_base_labels = [label for label, _ in BASE_QUANT_SIZES]
    size_table, dry_run_cache = build_size_table(
        source_gguf, quantize_bin, imatrix=imatrix, base_types=all_base_labels,
    )

    # Find the best base type from cached results
    best_base = None
    best_tensors = None
    best_size = 0

    if verbose:
        print(f"\nScanning base quant types...")

    for base_label, base_tier in BASE_QUANT_SIZES:
        tensors, total_size = dry_run_cache[base_label]
        fits = total_size <= budget_bytes
        if verbose:
            marker = " ✓" if fits else ""
            print(f"  {base_label:8s}: {total_size / 1e9:.2f} GB{marker}")
        if fits and total_size > best_size:
            best_base = base_label
            best_tensors = tensors
            best_size = total_size

    if best_base is None:
        print(f"ERROR: No base quant type fits within {budget_bytes/1e9:.1f} GB budget")
        return None, None

    remaining_budget = budget_bytes - best_size
    if verbose:
        print(f"\nBest base: {best_base} ({best_size / 1e9:.2f} GB)")
        print(f"Remaining budget: {remaining_budget / 1e6:.1f} MB")

        base_dist = Counter(t["assigned_type"].rstrip(",") for t in best_tensors.values())
        print(f"Base allocation: {dict(sorted(base_dist.items()))}")

    # Build promotion candidates: tensors that can be upgraded one tier
    candidates = []
    for name, tinfo in best_tensors.items():
        current = tinfo["assigned_type"].rstrip(",")
        target = next_tier(current)
        if target is None:
            continue

        sens_data = match_tensor_to_sensitivity(name, sensitivity_map)
        if sens_data is None:
            continue

        benefit = promotion_benefit(sens_data, current, target)
        target_size = lookup_tensor_size(size_table, name, tinfo["param_count"], target)
        cost = target_size - tinfo["size_bytes"]
        if cost <= 0:
            continue

        candidates.append({
            "name": name,
            "current": current,
            "target": target,
            "benefit": benefit,
            "cost": cost,
            "efficiency": benefit / cost,
            "param_count": tinfo["param_count"],
        })

    candidates.sort(key=lambda x: x["efficiency"], reverse=True)

    # Greedily promote the most efficient candidates
    overrides = {}
    spent = 0
    for c in candidates:
        if spent + c["cost"] > remaining_budget:
            continue
        overrides[c["name"]] = c["target"]
        spent += c["cost"]

    if verbose:
        print(f"\nPromotions: {len(overrides)} tensors upgraded, {spent / 1e6:.1f} MB spent")
        if overrides:
            override_dist = Counter(overrides.values())
            print(f"Promotion targets: {dict(sorted(override_dist.items()))}")

            # Show top promotions
            promoted = [c for c in candidates if c["name"] in overrides][:10]
            print(f"\nTop promotions by efficiency:")
            for p in promoted:
                print(f"  {p['name']:45s} {p['current']:5s} → {p['target']:5s}  "
                      f"benefit={p['benefit']:.0f}  cost={p['cost']/1e6:.1f}MB  "
                      f"pc={p['param_count']/1e6:.1f}M")

    return best_base, overrides


def write_tensor_types(overrides, output_path):
    if not overrides:
        print("No overrides needed — base quant type is optimal for this budget")
        return

    lines = []
    for name, qtype in sorted(overrides.items()):
        lines.append(f"{name}={qtype}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} tensor type overrides to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Osmosis budget allocator — fit a model to your VRAM budget"
    )
    parser.add_argument("--sensitivity", required=True, help="Sensitivity report JSON")
    parser.add_argument("--source-gguf", required=True, help="F16 source GGUF")
    parser.add_argument("--budget-gb", type=float, required=True, help="Target file size in GB")
    parser.add_argument("--output", required=True, help="Output tensor types file")
    parser.add_argument("--imatrix", default=None, help="Imatrix file for accurate size estimation")
    parser.add_argument("--quantize-bin", default=None, help="Path to llama-quantize")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    quantize_bin = find_quantize_bin(args.quantize_bin)
    print(f"Budget: {args.budget_gb:.1f} GB")
    print(f"Source: {args.source_gguf}")
    if args.imatrix:
        print(f"Imatrix: {args.imatrix}")

    sensitivity_map = build_sensitivity_map(args.sensitivity)
    print(f"Loaded {len(sensitivity_map)} sensitivity curves")

    budget_bytes = int(args.budget_gb * 1e9)
    base_type, overrides = allocate_budget(
        args.source_gguf, sensitivity_map, budget_bytes,
        quantize_bin, imatrix=args.imatrix, verbose=True,
    )

    if base_type is None:
        sys.exit(1)

    write_tensor_types(overrides, args.output)

    imatrix_flag = "--imatrix <imatrix.dat> "
    print(f"\nTo build the GGUF:")
    if overrides:
        print(f"  llama-quantize {imatrix_flag}\\")
        print(f"    --tensor-type-file {args.output} \\")
        print(f"    {args.source_gguf} \\")
        print(f"    output.gguf {base_type}")
    else:
        print(f"  llama-quantize {imatrix_flag}\\")
        print(f"    {args.source_gguf} \\")
        print(f"    output.gguf {base_type}")


if __name__ == "__main__":
    main()
