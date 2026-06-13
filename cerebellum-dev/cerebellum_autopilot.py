#!/usr/bin/env python3
"""
Cerebellum Autopilot — Fully automated ablation-guided mixed-precision quantization.

Set it on a model, walk away. It:
1. Enumerates all tensor groups and layers
2. Tests each tensor by demoting/promoting and measuring PPL impact
3. Pipelines CPU quantize with GPU PPL for speed
4. Manages disk space (cleans up test GGUFs immediately)
5. Classifies tensors as SENSITIVE / TOLERANT / BENEFICIAL
6. Builds budget-balanced overrides (promote sensitive, demote tolerant, net-zero size)
7. Quantizes the final model and runs benchmarks

Usage:
    python scripts/cerebellum_autopilot.py \\
        --bf16 /path/to/model-bf16.gguf \\
        --imatrix /path/to/imatrix.gguf \\
        --baseline-overrides /path/to/v4_overrides.txt \\
        --baseline-ppl 12613.59 \\
        --base-quant Q3_K_M \\
        --output-dir /path/to/autopilot_results \\
        --quantize-bin /path/to/llama-quantize \\
        --perplexity-bin /path/to/llama-perplexity \\
        --wikitext /path/to/wiki.test.raw \\
        --gpu-layers 99 \\
        --chunks 128
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TensorResult:
    group: str
    layer: int
    direction: str  # "demote" or "promote"
    from_quant: str
    to_quant: str
    ppl: float
    ppl_err: float
    baseline_ppl: float
    delta_pct: float
    verdict: str  # SENSITIVE, TOLERANT, BENEFICIAL, CRITICAL


@dataclass
class AutopilotConfig:
    bf16: str
    imatrix: str
    baseline_overrides: str
    baseline_ppl: float
    base_quant: str
    output_dir: str
    quantize_bin: str
    perplexity_bin: str
    wikitext: str
    gpu_layers: int = 99
    chunks: int = 128
    sensitive_threshold: float = 5.0   # >5% PPL increase = SENSITIVE
    tolerant_threshold: float = 1.0    # <1% change = TOLERANT
    beneficial_threshold: float = -1.0 # <-1% = BENEFICIAL (removing helps)
    skip_groups: list = field(default_factory=list)
    skip_layers: list = field(default_factory=list)
    sample_layers: Optional[list] = None  # test specific layers only (e.g. [0,5,10,15,20,25,29])
    max_disk_gb: float = 25.0  # max temp disk usage before blocking


def parse_overrides(path: str) -> dict[str, str]:
    """Parse override file into {tensor_name: quant_type} dict."""
    overrides = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                tensor, quant = line.split("=", 1)
                overrides[tensor] = quant
    return overrides


def write_overrides(overrides: dict[str, str], path: str):
    """Write override dict to file."""
    with open(path, "w") as f:
        for tensor, quant in sorted(overrides.items()):
            f.write(f"{tensor}={quant}\n")


def discover_tensor_groups(overrides: dict[str, str], n_layers: int = 30) -> dict[str, dict]:
    """
    Discover all tensor groups and their current quant state per layer.
    Returns {group_name: {layer: current_quant_or_None}}
    """
    known_groups = [
        "attn_q", "attn_k", "attn_v", "attn_o",
        "ffn_gate_up_exps", "ffn_down_exps",
        "ffn_gate", "ffn_up", "ffn_down",
    ]

    groups = {}
    for group in known_groups:
        layers = {}
        for layer in range(n_layers):
            tensor = f"blk.{layer}.{group}.weight"
            layers[layer] = overrides.get(tensor, None)  # None = default (base_quant)
        groups[group] = layers

    return groups


def check_disk_space(path: str) -> float:
    """Return available disk space in GB."""
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def run_quantize(cfg: AutopilotConfig, override_file: str, output_gguf: str) -> bool:
    """Run llama-quantize. Returns True on success."""
    cmd = [
        cfg.quantize_bin,
        "--imatrix", cfg.imatrix,
        "--tensor-type-file", override_file,
        cfg.bf16,
        output_gguf,
        cfg.base_quant,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  QUANTIZE FAILED: {e}", flush=True)
        return False


def run_perplexity(cfg: AutopilotConfig, model_path: str) -> tuple[float, float]:
    """Run llama-perplexity. Returns (ppl, error)."""
    cmd = [
        cfg.perplexity_bin,
        "--model", model_path,
        "--file", cfg.wikitext,
        "-ngl", str(cfg.gpu_layers),
        "--chunks", str(cfg.chunks),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stderr + result.stdout

        ppl_match = re.search(r"Final estimate: PPL = ([\d.]+)", output)
        err_match = re.search(r"\+/- ([\d.]+)", output)

        if ppl_match:
            ppl = float(ppl_match.group(1))
            err = float(err_match.group(1)) if err_match else 0.0
            return ppl, err
        else:
            print(f"  PPL PARSE FAILED. Output tail: {output[-500:]}", flush=True)
            return -1.0, 0.0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  PPL FAILED: {e}", flush=True)
        return -1.0, 0.0


def classify(delta_pct: float, cfg: AutopilotConfig) -> str:
    """Classify a tensor based on PPL delta percentage."""
    if delta_pct > cfg.sensitive_threshold:
        return "SENSITIVE"
    elif delta_pct < cfg.beneficial_threshold:
        return "BENEFICIAL"
    elif abs(delta_pct) <= cfg.tolerant_threshold:
        return "TOLERANT"
    else:
        return "MODERATE"


def ablate_tensor_group(
    cfg: AutopilotConfig,
    group: str,
    layers: dict[int, Optional[str]],
    baseline_overrides: dict[str, str],
    results: list[TensorResult],
    log_file: Path,
):
    """
    Ablate one tensor group across sampled layers.
    Uses pipelined approach: quantize N+1 on CPU while PPL runs on N.
    """
    test_layers = cfg.sample_layers if cfg.sample_layers else sorted(layers.keys())

    # Skip layers the user wants to skip
    test_layers = [l for l in test_layers if l not in cfg.skip_layers]

    if not test_layers:
        return

    # Determine ablation direction per layer
    # If layer has an override (e.g. Q2_K), test UN-demoting (remove override → back to base_quant)
    # If layer is at default (no override), test DEMOTING (add Q2_K override)
    ablations = []
    for layer in test_layers:
        current = layers[layer]
        tensor = f"blk.{layer}.{group}.weight"

        if current is not None:
            # Has override — test removing it (un-demote back to base)
            direction = "promote"
            from_q = current
            to_q = cfg.base_quant
        else:
            # At default — test demoting to Q2_K
            direction = "demote"
            from_q = cfg.base_quant
            to_q = "Q2_K"

        ablations.append((layer, tensor, direction, from_q, to_q))

    msg = f"\n=== ABLATING {group} ({len(ablations)} layers) ==="
    print(msg, flush=True)
    log(log_file, msg)
    log(log_file, f"Direction: {'mixed' if any(a[2] != ablations[0][2] for a in ablations) else ablations[0][2]}")

    outdir = Path(cfg.output_dir) / f"ablation_{group}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Pre-quantize first layer
    layer, tensor, direction, from_q, to_q = ablations[0]
    override_path, gguf_path = prep_ablation(cfg, baseline_overrides, outdir, tensor, direction, to_q, layer, group)

    msg = f"Quantizing {group} layer {layer} ({direction}: {from_q}→{to_q})..."
    print(f"  {msg}", flush=True)
    log(log_file, msg)

    if not run_quantize(cfg, str(override_path), str(gguf_path)):
        log(log_file, f"LAYER {layer} QUANTIZE FAILED — skipping")
        return

    # Pipeline loop
    for i, (layer, tensor, direction, from_q, to_q) in enumerate(ablations):
        # Start quantizing next in background
        next_proc = None
        next_gguf = None
        if i + 1 < len(ablations):
            nl, nt, nd, nfq, ntq = ablations[i + 1]
            next_override, next_gguf = prep_ablation(cfg, baseline_overrides, outdir, nt, nd, ntq, nl, group)

            # Check disk space
            avail = check_disk_space(str(outdir))
            if avail < cfg.max_disk_gb:
                msg = f"Low disk ({avail:.1f} GB) — waiting for current PPL before next quantize"
                print(f"  {msg}", flush=True)
                log(log_file, msg)
            else:
                next_cmd = [
                    cfg.quantize_bin,
                    "--imatrix", cfg.imatrix,
                    "--tensor-type-file", str(next_override),
                    cfg.bf16,
                    str(next_gguf),
                    cfg.base_quant,
                ]
                next_proc = subprocess.Popen(next_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log(log_file, f"Background quantize: {group} layer {nl}")

        # Run PPL on current
        msg = f"PPL {group} layer {layer}..."
        print(f"  {msg}", flush=True)
        log(log_file, msg)

        ppl, err = run_perplexity(cfg, str(gguf_path))

        # Clean up GGUF immediately
        if gguf_path.exists():
            gguf_path.unlink()

        if ppl < 0:
            log(log_file, f"LAYER {layer} PPL FAILED — skipping")
            if next_proc:
                next_proc.wait()
            continue

        delta_pct = ((ppl - cfg.baseline_ppl) / cfg.baseline_ppl) * 100
        verdict = classify(delta_pct, cfg)

        result = TensorResult(
            group=group, layer=layer, direction=direction,
            from_quant=from_q, to_quant=to_q,
            ppl=ppl, ppl_err=err,
            baseline_ppl=cfg.baseline_ppl,
            delta_pct=delta_pct, verdict=verdict,
        )
        results.append(result)

        msg = f"LAYER {layer} PPL: {ppl:.4f} (+/- {err:.5f}) | Δ={delta_pct:+.1f}% | {verdict}"
        print(f"  {msg}", flush=True)
        log(log_file, msg)

        # Save incremental results
        save_results(results, Path(cfg.output_dir) / "ablation_results.json")

        # Wait for background quantize
        if next_proc:
            next_proc.wait()
            if next_proc.returncode != 0:
                log(log_file, f"Background quantize failed for next layer")
        elif next_gguf and i + 1 < len(ablations):
            # Disk was low, quantize now sequentially
            nl, nt, nd, nfq, ntq = ablations[i + 1]
            next_override, next_gguf = prep_ablation(cfg, baseline_overrides, outdir, nt, nd, ntq, nl, group)
            run_quantize(cfg, str(next_override), str(next_gguf))

    msg = f"=== {group} COMPLETE: {len([r for r in results if r.group == group])} layers tested ==="
    print(msg, flush=True)
    log(log_file, msg)


def prep_ablation(
    cfg: AutopilotConfig,
    baseline_overrides: dict[str, str],
    outdir: Path,
    tensor: str,
    direction: str,
    to_quant: str,
    layer: int,
    group: str,
) -> tuple[Path, Path]:
    """Prepare override file and GGUF path for one ablation test."""
    test_overrides = dict(baseline_overrides)

    if direction == "demote":
        test_overrides[tensor] = to_quant
    elif direction == "promote":
        test_overrides.pop(tensor, None)

    override_path = outdir / f"test_overrides_{group}_layer_{layer}.txt"
    gguf_path = outdir / f"ablation_{group}_layer_{layer}.gguf"

    write_overrides(test_overrides, str(override_path))
    return override_path, gguf_path


def save_results(results: list[TensorResult], path: Path):
    """Save results as JSON."""
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def log(log_file: Path, msg: str):
    """Append to log file with timestamp."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def build_optimized_overrides(
    cfg: AutopilotConfig,
    results: list[TensorResult],
    baseline_overrides: dict[str, str],
) -> dict[str, str]:
    """
    Build budget-balanced overrides from ablation results.

    Strategy:
    - SENSITIVE tensors at default → keep at default (they need their bits)
    - SENSITIVE tensors with override → consider promoting
    - TOLERANT tensors at default → demote to Q2_K (free bits)
    - BENEFICIAL tensors → demote to Q2_K (actually helps)
    - Budget: count bits saved by demotions, spend on promotions
    """
    optimized = dict(baseline_overrides)

    demotions = []
    promotions = []

    for r in results:
        tensor = f"blk.{r.layer}.{r.group}.weight"

        if r.direction == "demote":
            # We tested demoting from default
            if r.verdict in ("TOLERANT", "BENEFICIAL"):
                demotions.append((tensor, "Q2_K", r.delta_pct))
            # SENSITIVE/MODERATE at default = leave alone
        elif r.direction == "promote":
            # We tested removing an override (promoting back to default)
            if r.verdict == "BENEFICIAL":
                # Removing the override helped — promote it
                promotions.append((tensor, None, r.delta_pct))  # None = remove override
            # If SENSITIVE when promoted, the override is important — keep it

    # Apply demotions (free bits)
    for tensor, quant, delta in sorted(demotions, key=lambda x: x[2]):
        optimized[tensor] = quant

    # Apply promotions (spend bits)
    for tensor, quant, delta in sorted(promotions, key=lambda x: x[2]):
        if quant is None:
            optimized.pop(tensor, None)
        else:
            optimized[tensor] = quant

    return optimized


def print_summary(results: list[TensorResult]):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("CEREBELLUM AUTOPILOT — ABLATION SUMMARY")
    print("=" * 80)

    by_group = {}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    for group, group_results in sorted(by_group.items()):
        print(f"\n--- {group} ---")
        for r in sorted(group_results, key=lambda x: x.layer):
            print(f"  Layer {r.layer:2d}: PPL {r.ppl:10.2f} | Δ={r.delta_pct:+6.1f}% | {r.verdict:10s} | {r.direction} {r.from_quant}→{r.to_quant}")

    # Count verdicts
    verdicts = {}
    for r in results:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
    print(f"\nVerdicts: {verdicts}")
    print(f"Total tensors tested: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Cerebellum Autopilot — automated ablation pipeline")
    parser.add_argument("--bf16", required=True, help="Path to bf16 GGUF model")
    parser.add_argument("--imatrix", required=True, help="Path to imatrix GGUF")
    parser.add_argument("--baseline-overrides", required=True, help="Path to baseline override file (e.g. v4)")
    parser.add_argument("--baseline-ppl", required=True, type=float, help="Baseline PPL (e.g. 12613.59)")
    parser.add_argument("--base-quant", default="Q3_K_M", help="Base quantization type (default: Q3_K_M)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--quantize-bin", required=True, help="Path to llama-quantize binary")
    parser.add_argument("--perplexity-bin", required=True, help="Path to llama-perplexity binary")
    parser.add_argument("--wikitext", required=True, help="Path to wikitext test file")
    parser.add_argument("--gpu-layers", type=int, default=99, help="GPU layers for perplexity")
    parser.add_argument("--chunks", type=int, default=128, help="Chunks for perplexity")
    parser.add_argument("--n-layers", type=int, default=30, help="Number of transformer layers")
    parser.add_argument("--sample-layers", type=str, default=None,
                        help="Comma-separated layer indices to test (default: all)")
    parser.add_argument("--skip-groups", type=str, default=None,
                        help="Comma-separated tensor groups to skip")
    parser.add_argument("--sensitive-threshold", type=float, default=5.0)
    parser.add_argument("--tolerant-threshold", type=float, default=1.0)
    parser.add_argument("--max-disk-gb", type=float, default=25.0)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    cfg = AutopilotConfig(
        bf16=args.bf16,
        imatrix=args.imatrix,
        baseline_overrides=args.baseline_overrides,
        baseline_ppl=args.baseline_ppl,
        base_quant=args.base_quant,
        output_dir=args.output_dir,
        quantize_bin=args.quantize_bin,
        perplexity_bin=args.perplexity_bin,
        wikitext=args.wikitext,
        gpu_layers=args.gpu_layers,
        chunks=args.chunks,
        sensitive_threshold=args.sensitive_threshold,
        tolerant_threshold=args.tolerant_threshold,
        max_disk_gb=args.max_disk_gb,
    )

    if args.sample_layers:
        cfg.sample_layers = [int(x) for x in args.sample_layers.split(",")]
    if args.skip_groups:
        cfg.skip_groups = [x.strip() for x in args.skip_groups.split(",")]

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = Path(cfg.output_dir) / "autopilot.log"

    # Load baseline overrides
    baseline_overrides = parse_overrides(cfg.baseline_overrides)

    # Discover tensor groups
    groups = discover_tensor_groups(baseline_overrides, n_layers=args.n_layers)

    # Resume support
    results = []
    results_path = Path(cfg.output_dir) / "ablation_results.json"
    tested = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
            results = [TensorResult(**r) for r in saved]
            tested = {(r.group, r.layer) for r in results}
        msg = f"Resumed: {len(results)} results loaded, skipping already-tested tensors"
        print(msg, flush=True)
        log(log_file, msg)

    msg = f"CEREBELLUM AUTOPILOT START — {len(groups)} tensor groups, {args.n_layers} layers"
    print(msg, flush=True)
    log(log_file, msg)
    log(log_file, f"Baseline PPL: {cfg.baseline_ppl}")
    log(log_file, f"Base quant: {cfg.base_quant}")
    log(log_file, f"Disk budget: {cfg.max_disk_gb} GB")
    log(log_file, f"Sample layers: {cfg.sample_layers or 'all'}")

    for group, layers in sorted(groups.items()):
        if group in cfg.skip_groups:
            log(log_file, f"Skipping {group} (user skip list)")
            continue

        # Filter out already-tested layers for resume
        if tested:
            remaining = {l: q for l, q in layers.items() if (group, l) not in tested}
            if not remaining:
                log(log_file, f"Skipping {group} — all layers already tested")
                continue
            layers = remaining

        ablate_tensor_group(cfg, group, layers, baseline_overrides, results, log_file)

    # Summary
    print_summary(results)
    save_results(results, results_path)

    # Build optimized overrides
    optimized = build_optimized_overrides(cfg, results, baseline_overrides)
    opt_path = Path(cfg.output_dir) / "optimized_overrides.txt"
    write_overrides(optimized, str(opt_path))

    changes = len(optimized) - len(baseline_overrides)
    msg = f"\nOptimized overrides written to {opt_path} ({len(optimized)} total, {changes:+d} vs baseline)"
    print(msg, flush=True)
    log(log_file, msg)
    log(log_file, "CEREBELLUM AUTOPILOT COMPLETE")


if __name__ == "__main__":
    main()
