"""Cerebellum — ablation-informed tensor placement and precision optimizer.

Uses ground-truth perplexity ablation data (not proxy estimates) to decide:
1. Which tensors should be at what precision (tensor-type-file for llama-quantize)
2. Which layers should be on GPU vs CPU (optimal -ngl split)
3. Future: per-tensor backend override file for llama.cpp

The ablation experiment crushes one tensor at a time from a Q4_K_M baseline
to Q2_K and measures the real PPL delta. Negative delta = tensor is BETTER
at lower precision. Positive delta = tensor needs protection.

Usage:
    python -m osmosis.cerebellum \
        --ablation osmosis-qwen36-27b/ablation_results.json \
        --plan osmosis-qwen36-27b/ablation_plan.json \
        --source-gguf qwen3.6-27b-f16.gguf \
        --budget-gb 12.0 \
        --output tensor_types.txt
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from osmosis.budget import (
    QUANT_BPW,
    PROMOTION_ORDER,
    HF_TO_GGUF_COMPONENT,
    dry_run_quantize,
    find_quantize_bin,
    normalize_qtype,
    next_tier,
    estimate_tensor_size,
)

GGUF_TO_HF = {v: k for k, v in HF_TO_GGUF_COMPONENT.items()}

DEMOTION_ORDER = list(reversed(PROMOTION_ORDER))


def prev_tier(current_type):
    normed = normalize_qtype(current_type)
    try:
        idx = PROMOTION_ORDER.index(normed)
    except ValueError:
        return None
    if idx > 0:
        return PROMOTION_ORDER[idx - 1]
    return None


def load_ablation_data(results_path, plan_path=None):
    with open(results_path) as f:
        results = json.load(f)

    baseline_ppl = results["baseline_ppl"]
    tests = results.get("tests", {})

    ablation_map = {}
    for hf_name, data in tests.items():
        gguf_tensor = data["gguf_tensor"]
        ppl = data["ppl"]
        delta = ppl - baseline_ppl
        ablation_map[gguf_tensor] = {
            "hf_name": hf_name,
            "ppl": ppl,
            "delta": delta,
            "abs_delta": abs(delta),
        }

    plan_info = {}
    if plan_path and Path(plan_path).exists():
        with open(plan_path) as f:
            plan = json.load(f)
        for t in plan.get("tensors", []):
            name = t["name"]
            plan_info[name] = t

    return baseline_ppl, ablation_map, plan_info


def gguf_tensor_to_layer(tensor_name):
    m = re.match(r"blk\.(\d+)\.", tensor_name)
    return int(m.group(1)) if m else None


def classify_tensors(ablation_map, noise_threshold=0.02):
    """Classify tensors into categories based on ablation deltas.

    Returns dict of gguf_tensor_name -> classification:
      - "demote": negative delta beyond noise — actively better at lower precision
      - "sacred": positive delta beyond noise — needs protection
      - "neutral": within noise threshold — don't care
    """
    classifications = {}
    for tensor_name, data in ablation_map.items():
        delta = data["delta"]
        if delta < -noise_threshold:
            classifications[tensor_name] = "demote"
        elif delta > noise_threshold:
            classifications[tensor_name] = "sacred"
        else:
            classifications[tensor_name] = "neutral"
    return classifications


def extrapolate_layer_sensitivity(ablation_map, n_layers=64):
    """Build per-layer sensitivity scores from sparse ablation samples.

    Uses tested tensors to estimate overall layer sensitivity.
    Layers with no data get interpolated from neighbors.
    """
    layer_deltas = defaultdict(list)
    for tensor_name, data in ablation_map.items():
        layer = gguf_tensor_to_layer(tensor_name)
        if layer is not None:
            layer_deltas[layer].append(data["delta"])

    layer_scores = {}
    for layer in range(n_layers):
        if layer in layer_deltas:
            deltas = layer_deltas[layer]
            layer_scores[layer] = sum(deltas) / len(deltas)
        else:
            layer_scores[layer] = None

    # interpolate missing layers from nearest neighbors
    for layer in range(n_layers):
        if layer_scores[layer] is not None:
            continue
        lower = upper = None
        for l in range(layer - 1, -1, -1):
            if layer_scores[l] is not None:
                lower = (l, layer_scores[l])
                break
        for l in range(layer + 1, n_layers):
            if layer_scores[l] is not None:
                upper = (l, layer_scores[l])
                break
        if lower and upper:
            frac = (layer - lower[0]) / (upper[0] - lower[0])
            layer_scores[layer] = lower[1] + frac * (upper[1] - lower[1])
        elif lower:
            layer_scores[layer] = lower[1]
        elif upper:
            layer_scores[layer] = upper[1]
        else:
            layer_scores[layer] = 0.0

    return layer_scores


def optimal_gpu_split(layer_scores, n_layers=64):
    """Find optimal -ngl value by putting the most sensitive layers on GPU.

    llama.cpp offloads the LAST n_gpu_layers to GPU. So we want the split
    point where all high-sensitivity layers are on GPU.

    Returns (n_gpu_layers, gpu_sensitivity_sum, cpu_sensitivity_sum).
    """
    results = []
    for ngl in range(0, n_layers + 1):
        cpu_start = 0
        cpu_end = n_layers - ngl
        gpu_start = cpu_end
        gpu_end = n_layers

        cpu_sens = sum(max(0, layer_scores[l]) for l in range(cpu_start, cpu_end))
        gpu_sens = sum(max(0, layer_scores[l]) for l in range(gpu_start, gpu_end))
        cpu_risk = sum(max(0, layer_scores[l]) for l in range(cpu_start, cpu_end))

        results.append({
            "ngl": ngl,
            "cpu_layers": f"{cpu_start}-{cpu_end-1}" if cpu_end > cpu_start else "none",
            "gpu_layers": f"{gpu_start}-{gpu_end-1}" if gpu_end > gpu_start else "none",
            "cpu_risk": cpu_risk,
            "gpu_protected": gpu_sens,
        })

    return results


def ablation_aware_allocate(source_gguf, ablation_map, budget_bytes,
                            quantize_bin, base_type="Q4_K_M", verbose=False):
    """Precision allocator that uses ground-truth ablation data.

    Strategy:
    1. Start from a base quant type (default Q4_K_M)
    2. DEMOTE tensors with negative ablation deltas (save bits)
    3. PROMOTE sacred tensors with freed budget
    4. For tensors without ablation data, use position-based heuristics
       derived from ablation patterns
    """
    classifications = classify_tensors(ablation_map)
    tensors, base_size = dry_run_quantize(source_gguf, base_type, quantize_bin)

    if verbose:
        print(f"\nBase type: {base_type} ({base_size / 1e9:.2f} GB)")
        print(f"Budget: {budget_bytes / 1e9:.2f} GB")
        print(f"Ablation data: {len(ablation_map)} tensors tested")
        n_demote = sum(1 for c in classifications.values() if c == "demote")
        n_sacred = sum(1 for c in classifications.values() if c == "sacred")
        n_neutral = sum(1 for c in classifications.values() if c == "neutral")
        print(f"Classifications: {n_demote} demote, {n_sacred} sacred, {n_neutral} neutral")

    overrides = {}
    saved_bytes = 0

    # Phase 1: Demote tensors that are proven better at lower precision
    for tensor_name, tinfo in tensors.items():
        if tensor_name not in ablation_map:
            continue
        classification = classifications.get(tensor_name)
        if classification != "demote":
            continue

        current = normalize_qtype(tinfo["assigned_type"].rstrip(","))
        target = prev_tier(current)
        if target is None:
            continue

        old_size = tinfo["size_bytes"]
        new_size = estimate_tensor_size(tinfo["param_count"], target)
        savings = old_size - new_size
        if savings > 0:
            overrides[tensor_name] = target
            saved_bytes += savings
            if verbose:
                delta = ablation_map[tensor_name]["delta"]
                print(f"  DEMOTE {tensor_name}: {current} → {target} "
                      f"(delta={delta:+.4f}, saves {savings/1e6:.1f}MB)")

    # Phase 2: Extrapolate demotions to untested tensors in safe layers
    layer_scores = extrapolate_layer_sensitivity(ablation_map)
    for tensor_name, tinfo in tensors.items():
        if tensor_name in ablation_map or tensor_name in overrides:
            continue

        layer = gguf_tensor_to_layer(tensor_name)
        if layer is None:
            continue

        score = layer_scores.get(layer, 0)
        if score >= -0.02:
            continue

        current = normalize_qtype(tinfo["assigned_type"].rstrip(","))
        target = prev_tier(current)
        if target is None:
            continue

        old_size = tinfo["size_bytes"]
        new_size = estimate_tensor_size(tinfo["param_count"], target)
        savings = old_size - new_size
        if savings > 0:
            overrides[tensor_name] = target
            saved_bytes += savings

    if verbose:
        n_extrapolated = sum(1 for t in overrides if t not in ablation_map)
        print(f"\n  Extrapolated {n_extrapolated} additional demotions from layer scores")
        print(f"  Total saved: {saved_bytes / 1e6:.1f} MB")

    # Phase 3: Promote sacred tensors (spend saved budget)
    # Multi-pass: keep promoting until budget exhausted or no more candidates
    promotion_budget = saved_bytes + (budget_bytes - base_size)
    if verbose:
        print(f"\n  Promotion budget: {promotion_budget / 1e6:.1f} MB")

    spent = 0
    current_types = {name: normalize_qtype(t["assigned_type"].rstrip(","))
                     for name, t in tensors.items()}
    for name in overrides:
        current_types[name] = overrides[name]

    promotion_pass = 0
    while True:
        promotion_pass += 1
        promo_candidates = []
        for tensor_name, tinfo in tensors.items():
            classification = classifications.get(tensor_name)
            current = current_types[tensor_name]
            target = next_tier(current)
            if target is None:
                continue

            cost = (estimate_tensor_size(tinfo["param_count"], target)
                    - estimate_tensor_size(tinfo["param_count"], current))
            if cost <= 0:
                continue

            if classification == "sacred":
                benefit = ablation_map[tensor_name]["delta"] * tinfo["param_count"]
            else:
                layer = gguf_tensor_to_layer(tensor_name)
                score = layer_scores.get(layer, 0) if layer is not None else 0
                if score <= 0:
                    continue
                benefit = score * tinfo["param_count"]

            promo_candidates.append({
                "name": tensor_name,
                "current": current,
                "target": target,
                "benefit": benefit,
                "cost": cost,
                "efficiency": benefit / cost,
                "source": "ablation" if classification == "sacred" else "extrapolated",
            })

        if not promo_candidates:
            break

        promo_candidates.sort(key=lambda x: x["efficiency"], reverse=True)

        promoted_this_pass = 0
        for c in promo_candidates:
            if spent + c["cost"] > promotion_budget:
                continue
            overrides[c["name"]] = c["target"]
            current_types[c["name"]] = c["target"]
            spent += c["cost"]
            promoted_this_pass += 1

        if promoted_this_pass == 0:
            break

    if verbose:
        n_promoted = len([n for n in overrides if overrides[n] != "q2_K"
                          or n not in {k for k, v in classifications.items() if v == "demote"}])
        print(f"  Promoted across {promotion_pass} passes, spent {spent / 1e6:.1f} MB")

        final_size = base_size - saved_bytes + spent
        print(f"\n  Final estimated size: {final_size / 1e9:.2f} GB")

    return base_type, overrides


def print_layer_heatmap(layer_scores, n_layers=64):
    """Print a visual heatmap of per-layer sensitivity."""
    print("\nLayer sensitivity heatmap (negative=safe to crush, positive=needs protection):")
    print("  Layer  Score   Bar")
    print("  " + "-" * 60)

    max_abs = max(abs(v) for v in layer_scores.values()) or 1
    for layer in range(n_layers):
        score = layer_scores[layer]
        bar_len = int(abs(score) / max_abs * 30)
        if score < -0.02:
            bar = " " * 30 + "|" + ">" * bar_len + " SAFE"
        elif score > 0.02:
            bar = " " * (30 - bar_len) + "<" * bar_len + "|" + " " * 30
            bar = bar[:61] + " PROTECT"
        else:
            bar = " " * 30 + "|"
        print(f"  {layer:5d}  {score:+.4f}  {bar}")


def print_split_recommendations(split_results, n_layers=64):
    """Print GPU/CPU split recommendations."""
    print("\nGPU/CPU split recommendations:")
    print(f"  {'ngl':>4s}  {'CPU layers':>12s}  {'GPU layers':>12s}  {'CPU risk':>10s}  Note")
    print("  " + "-" * 60)

    for r in split_results:
        ngl = r["ngl"]
        if ngl == 0 or ngl == n_layers or ngl % 8 == 0:
            note = ""
            if r["cpu_risk"] < 0.01:
                note = "← all sensitive layers on GPU"
            print(f"  {ngl:4d}  {r['cpu_layers']:>12s}  {r['gpu_layers']:>12s}  "
                  f"{r['cpu_risk']:10.4f}  {note}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cerebellum — ablation-informed tensor precision and placement optimizer"
    )
    parser.add_argument("--ablation", required=True, help="Ablation results JSON")
    parser.add_argument("--plan", default=None, help="Ablation plan JSON (optional)")
    parser.add_argument("--source-gguf", help="F16 source GGUF (for allocation mode)")
    parser.add_argument("--budget-gb", type=float, help="Target size in GB (for allocation mode)")
    parser.add_argument("--base-type", default="Q4_K_M", help="Base quant type")
    parser.add_argument("--output", help="Output tensor types file")
    parser.add_argument("--quantize-bin", default=None)
    parser.add_argument("--analyze-only", action="store_true", help="Just analyze, don't allocate")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    baseline_ppl, ablation_map, plan_info = load_ablation_data(
        args.ablation, args.plan
    )

    print(f"Baseline PPL: {baseline_ppl}")
    print(f"Ablation tests: {len(ablation_map)}")

    # Classify tensors
    classifications = classify_tensors(ablation_map)
    print(f"\nTensor classifications (noise threshold ±0.02):")
    for cat in ["demote", "neutral", "sacred"]:
        tensors_in_cat = [t for t, c in classifications.items() if c == cat]
        if tensors_in_cat:
            print(f"  {cat}: {len(tensors_in_cat)}")
            for t in sorted(tensors_in_cat, key=lambda x: ablation_map[x]["delta"]):
                d = ablation_map[t]
                print(f"    {t:45s}  delta={d['delta']:+.4f}  ({d['hf_name']})")

    # Layer analysis
    layer_scores = extrapolate_layer_sensitivity(ablation_map)
    print_layer_heatmap(layer_scores)

    # GPU/CPU split recommendations
    split_results = optimal_gpu_split(layer_scores)
    print_split_recommendations(split_results)

    # Allocation mode
    if args.source_gguf and args.budget_gb:
        quantize_bin = find_quantize_bin(args.quantize_bin)
        budget_bytes = int(args.budget_gb * 1e9)

        base_type, overrides = ablation_aware_allocate(
            args.source_gguf, ablation_map, budget_bytes,
            quantize_bin, base_type=args.base_type, verbose=True,
        )

        if args.output and overrides:
            lines = [f"{name}={qtype}" for name, qtype in sorted(overrides.items())]
            with open(args.output, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\nWrote {len(lines)} tensor type overrides to {args.output}")

            print(f"\nTo build:")
            print(f"  llama-quantize --imatrix <imatrix.dat> \\")
            print(f"    --tensor-type-file {args.output} \\")
            print(f"    <source.gguf> output.gguf {base_type}")


if __name__ == "__main__":
    main()
