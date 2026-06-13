"""Cerebellum budget allocator — ablation-driven mixed-precision optimizer.

Given real PPL ablation data and a target file size, selects the optimal
set of promotions and demotions to maximize quality within a size budget.

Uses actual measured PPL deltas (not proxy metrics) to rank decisions.
Solves the bounded knapsack: maximize quality improvement subject to size constraint.

Usage:
    python cerebellum_budget.py \
        --ablation ablation_results.json \
        --baseline-ppl 5.6415 \
        --baseline-size 39.0 \
        --budget-gb 39.0 \
        --output override_v2.txt \
        -v

Ablation JSON format:
{
    "baseline_ppl": 5.6415,
    "baseline_size_gb": 38.97,
    "promotions": [
        {
            "group": "ffn_down_exps",
            "target_type": "Q3_K",
            "ppl": 5.1786,
            "size_delta_gb": 1.69,
            "layers": "all"
        },
        ...
    ],
    "demotions": [
        {
            "group": "ssm_out",
            "source_type": "q6_K",
            "target_type": "q2_K",
            "size_delta_gb": -0.55,
            "ppl_cost_estimate": 0.02,
            "layers": "all"
        },
        ...
    ]
}
"""
import argparse
import json
import sys
from pathlib import Path


LAYER_COUNT = 48
FULL_ATTN_LAYERS = list(range(3, 48, 4))  # 3,7,11,...,47
SSM_LAYERS = [i for i in range(48) if i not in FULL_ATTN_LAYERS]


def load_ablation(path):
    with open(path) as f:
        return json.load(f)


def compute_efficiency(item, baseline_ppl):
    """PPL improvement per GB spent (higher = better for promotions)."""
    ppl_delta = baseline_ppl - item["ppl"]
    size_cost = item["size_delta_gb"]
    if size_cost <= 0:
        return float("inf")
    return ppl_delta / size_cost


def generate_override_lines(group, target_type, layers="all"):
    """Generate override file lines for a tensor group."""
    lines = []
    if layers == "global":
        lines.append(f"{group}.weight={target_type}")
        return lines
    if layers == "all":
        layer_list = range(LAYER_COUNT)
    elif layers == "full_attn":
        layer_list = FULL_ATTN_LAYERS
    elif layers == "ssm":
        layer_list = SSM_LAYERS
    else:
        layer_list = layers

    for i in layer_list:
        lines.append(f"blk.{i}.{group}.weight={target_type}")
    return lines


def solve_budget(ablation, budget_gb, verbose=False):
    """Net-benefit solver: for each promotion, check if it's worth doing even with demotions.

    Strategy:
    1. Rank promotions by efficiency (PPL gain / GB cost)
    2. For each promotion, compute the total demotions needed to stay within budget
    3. Accept the promotion if net PPL gain (promotion benefit - demotion damage) is positive
    4. Also allow pure demotions if budget < baseline (shrink-only mode)
    """
    baseline_ppl = ablation["baseline_ppl"]
    baseline_size = ablation["baseline_size_gb"]

    # Score promotions by efficiency
    promotions = []
    for p in ablation.get("promotions", []):
        eff = compute_efficiency(p, baseline_ppl)
        promotions.append({**p, "efficiency": eff})
    promotions.sort(key=lambda x: x["efficiency"], reverse=True)

    # Score demotions by damage efficiency (least PPL cost per GB saved)
    demotions = []
    for d in ablation.get("demotions", []):
        savings = abs(d["size_delta_gb"])
        ppl_cost = d.get("ppl_cost_estimate", 0.01)
        demotions.append({**d, "damage_per_gb": ppl_cost / savings if savings > 0 else float("inf")})
    demotions.sort(key=lambda x: x["damage_per_gb"])

    available_budget = budget_gb - baseline_size
    total_demotion_capacity = sum(abs(d["size_delta_gb"]) for d in demotions)

    if verbose:
        print(f"\n=== BUDGET ALLOCATION ===")
        print(f"Baseline: {baseline_size:.2f} GB, PPL {baseline_ppl:.4f}")
        print(f"Target:   {budget_gb:.2f} GB")
        print(f"Budget:   {available_budget:+.2f} GB")
        print(f"Demotion capacity: {total_demotion_capacity:.2f} GB")

    selected_promotions = []
    selected_demotions = []
    size_used = 0  # net size change from promotions
    size_reclaimed = 0  # total reclaimed from demotions
    demotion_idx = 0  # track which demotions are consumed
    total_ppl_improvement = 0
    total_ppl_damage = 0

    if verbose:
        print(f"\n--- Evaluating promotions (ranked by efficiency) ---")

    for p in promotions:
        cost = p["size_delta_gb"]
        ppl_gain = baseline_ppl - p["ppl"]

        # How much total space needed if we add this promotion?
        new_size_used = size_used + cost
        overage = new_size_used - size_reclaimed - available_budget

        if overage <= 0:
            # Fits without additional demotions
            selected_promotions.append(p)
            size_used += cost
            total_ppl_improvement += ppl_gain
            if verbose:
                print(f"  SELECTED: {p['group']} → {p['target_type']}: "
                      f"PPL {ppl_gain:+.4f}, +{cost:.2f} GB (fits in budget)")
            continue

        # Need demotions to make room — check if net benefit is positive
        needed = overage
        candidate_demotions = []
        candidate_damage = 0
        candidate_savings = 0
        temp_idx = demotion_idx

        while candidate_savings < needed and temp_idx < len(demotions):
            d = demotions[temp_idx]
            candidate_demotions.append(d)
            candidate_savings += abs(d["size_delta_gb"])
            candidate_damage += d.get("ppl_cost_estimate", 0.01)
            temp_idx += 1

        if candidate_savings < needed:
            if verbose:
                print(f"  SKIP:     {p['group']} → {p['target_type']}: "
                      f"needs {needed:.2f} GB, only {candidate_savings:.2f} GB available")
            continue

        # Net benefit check: is the promotion worth the demotion damage?
        net_ppl = ppl_gain - candidate_damage
        if net_ppl <= 0:
            if verbose:
                print(f"  REJECT:   {p['group']} → {p['target_type']}: "
                      f"PPL gain {ppl_gain:.4f} < demotion cost {candidate_damage:.4f}")
            continue

        # Accept promotion + demotions
        selected_promotions.append(p)
        size_used += cost
        total_ppl_improvement += ppl_gain

        for d in candidate_demotions:
            selected_demotions.append(d)
            size_reclaimed += abs(d["size_delta_gb"])
            total_ppl_damage += d.get("ppl_cost_estimate", 0.01)
            demotion_idx += 1

        if verbose:
            print(f"  SELECTED: {p['group']} → {p['target_type']}: "
                  f"PPL {ppl_gain:+.4f}, +{cost:.2f} GB")
            print(f"    + {len(candidate_demotions)} demotions "
                  f"({candidate_savings:.2f} GB saved, ~{candidate_damage:.4f} PPL cost)")
            print(f"    Net: {net_ppl:+.4f} PPL improvement")

    # If budget < baseline (shrink mode), add demotions even without promotions
    remaining_overage = size_used - size_reclaimed - available_budget
    if remaining_overage > 0:
        if verbose:
            print(f"\n--- Additional demotions needed: {remaining_overage:.2f} GB ---")
        while remaining_overage > 0 and demotion_idx < len(demotions):
            d = demotions[demotion_idx]
            selected_demotions.append(d)
            savings = abs(d["size_delta_gb"])
            size_reclaimed += savings
            remaining_overage -= savings
            total_ppl_damage += d.get("ppl_cost_estimate", 0.01)
            demotion_idx += 1
            if verbose:
                print(f"  DEMOTE: {d['group']} → {d['target_type']}: "
                      f"-{savings:.2f} GB, PPL cost ~{d.get('ppl_cost_estimate', '?')}")

    final_size = baseline_size + size_used - size_reclaimed
    net_ppl_change = total_ppl_improvement - total_ppl_damage

    if verbose:
        print(f"\n--- Result ---")
        print(f"  Promotions: {len(selected_promotions)} (+{size_used:.2f} GB)")
        print(f"  Demotions:  {len(selected_demotions)} (-{size_reclaimed:.2f} GB)")
        print(f"  Final size: {final_size:.2f} GB (target: {budget_gb:.2f})")
        print(f"  PPL gain from promotions: ~{total_ppl_improvement:.4f}")
        print(f"  PPL cost from demotions:  ~{total_ppl_damage:.4f}")
        print(f"  Net PPL improvement:      ~{net_ppl_change:.4f}")
        est_ppl = baseline_ppl - net_ppl_change
        print(f"  Estimated final PPL:      ~{est_ppl:.4f}")

    return selected_promotions, selected_demotions, final_size


def build_override_file(selected_promotions, selected_demotions):
    """Convert selections to override file lines."""
    lines = []
    for p in selected_promotions:
        lines.extend(generate_override_lines(p["group"], p["target_type"], p.get("layers", "all")))
    for d in selected_demotions:
        lines.extend(generate_override_lines(d["group"], d["target_type"], d.get("layers", "all")))
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Cerebellum budget allocator — ablation-driven quant optimizer"
    )
    parser.add_argument("--ablation", required=True, help="Ablation results JSON")
    parser.add_argument("--budget-gb", type=float, required=True, help="Target file size in GB")
    parser.add_argument("--output", required=True, help="Output override file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    ablation = load_ablation(args.ablation)

    selected_promotions, selected_demotions, final_size = solve_budget(
        ablation, args.budget_gb, verbose=args.verbose
    )

    lines = build_override_file(selected_promotions, selected_demotions)

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {len(lines)} overrides to {args.output}")
    print(f"Estimated size: {final_size:.2f} GB")


if __name__ == "__main__":
    main()
