#!/usr/bin/env python3
"""Dead Shard Detector — find row-blocks that contribute nothing useful.

Three detection methods:
1. Weight energy: shards with near-zero Frobenius norm (weights too small to matter)
2. Signal-to-noise: after quantization, how much signal survives vs becomes noise?
3. Output contribution: for calibration inputs, what fraction of output does each shard produce?

The key insight: at SHARD granularity (128 row-blocks per tensor), many blocks
have so little weight energy that quantizing them to Q2_K turns them into noise.
Those are dead shards — zero them out and save the VRAM/RAM.

Usage:
    # Analyze a GGUF for dead shards
    python scripts/dead_shard_detector.py \
        --gguf model-q2k.gguf \
        --reference-gguf model-f16.gguf \
        --n-shards 128 \
        --output dead_shards.json

    # Synthetic test
    python scripts/dead_shard_detector.py --synthetic
"""
import argparse
import json
import sys
import time

import numpy as np


def detect_dead_shards_by_energy(W, n_shards):
    """Method 1: weight energy per shard.
    Shards with tiny Frobenius norm relative to the tensor mean are candidates."""
    rows, cols = W.shape
    shard_size = rows // n_shards

    energies = np.zeros(n_shards)
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, rows)
        energies[i] = np.linalg.norm(W[start:end], 'fro') ** 2

    total_energy = energies.sum()
    relative = energies / (total_energy + 1e-10)
    uniform = 1.0 / n_shards

    return {
        "energies": energies,
        "relative": relative,
        "dead_mask": relative < uniform * 0.1,
        "weak_mask": relative < uniform * 0.5,
    }


def detect_dead_shards_by_snr(W_hifi, W_quant, n_shards):
    """Method 2: signal-to-noise ratio per shard after quantization.
    If quant error ≈ signal magnitude, the shard is noise."""
    rows, cols = W_hifi.shape
    shard_size = rows // n_shards

    snr_db = np.zeros(n_shards)
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, rows)
        signal = W_hifi[start:end]
        noise = W_hifi[start:end] - W_quant[start:end]
        signal_power = np.mean(signal ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        snr_db[i] = 10 * np.log10(signal_power / noise_power)

    return {
        "snr_db": snr_db,
        "dead_mask": snr_db < 3.0,  # <3dB = noise dominates
        "weak_mask": snr_db < 6.0,  # <6dB = marginal
    }


def detect_dead_shards_by_output(W, inputs, n_shards):
    """Method 3: output contribution per shard for calibration inputs."""
    rows, cols = W.shape
    shard_size = rows // n_shards

    full_output = inputs @ W.T  # (batch, rows)
    full_norm = np.linalg.norm(full_output, axis=1, keepdims=True) + 1e-10

    contributions = np.zeros(n_shards)
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, rows)
        shard_output = full_output[:, start:end]
        contributions[i] = np.mean(np.linalg.norm(shard_output, axis=1) / full_norm.squeeze())

    uniform = 1.0 / n_shards
    return {
        "contributions": contributions,
        "dead_mask": contributions < uniform * 0.1,
        "weak_mask": contributions < uniform * 0.5,
    }


def simulate_quant(W, bits=2):
    """Simulate quantization at different bit levels."""
    if bits == 2:
        levels = 3.5  # Q2_K ~ 4 levels per group
    elif bits == 4:
        levels = 7.5  # Q4_K ~ 16 levels
    elif bits == 8:
        levels = 127.5
    else:
        return W.copy()

    scale = np.abs(W).max(axis=1, keepdims=True) / levels
    scale = np.maximum(scale, 1e-10)
    return np.round(W / scale) * scale


def synthetic_test(n_shards=128):
    """Test dead shard detection on synthetic weights with realistic structure."""
    print(f"\n{'='*80}")
    print(f"DEAD SHARD DETECTOR — Synthetic Test ({n_shards} shards)")
    print(f"{'='*80}\n")

    np.random.seed(42)

    # Simulate a realistic transformer weight: low-rank + sparse activations
    out_dim, in_dim = 4096, 2048
    shard_size = out_dim // n_shards
    print(f"Weight: {out_dim}x{in_dim}, {n_shards} shards of {shard_size} rows")

    # Create weight with intentional dead zones
    # Most rows have signal, but ~20% are near-zero (dead neurons from training)
    W = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.01
    dead_rows = np.random.choice(out_dim, size=out_dim // 5, replace=False)
    W[dead_rows] *= 0.001  # kill 20% of rows

    # Also create some super-important rows (sacred)
    sacred_rows = np.random.choice(
        [r for r in range(out_dim) if r not in dead_rows], size=out_dim // 10, replace=False)
    W[sacred_rows] *= 5.0

    W_q2 = simulate_quant(W, bits=2)
    W_q4 = simulate_quant(W, bits=4)
    inputs = np.maximum(0, np.random.randn(512, in_dim).astype(np.float32) * 0.1)

    # Method 1: Energy
    print("\n--- Method 1: Weight Energy ---")
    energy = detect_dead_shards_by_energy(W, n_shards)
    dead_e = energy["dead_mask"].sum()
    weak_e = energy["weak_mask"].sum()
    print(f"Dead shards (energy <10% of uniform): {dead_e}/{n_shards} ({dead_e/n_shards:.0%})")
    print(f"Weak shards (energy <50% of uniform): {weak_e}/{n_shards} ({weak_e/n_shards:.0%})")

    # Method 2: SNR after Q2_K
    print("\n--- Method 2: Signal-to-Noise after Q2_K ---")
    snr = detect_dead_shards_by_snr(W, W_q2, n_shards)
    dead_s = snr["dead_mask"].sum()
    weak_s = snr["weak_mask"].sum()
    print(f"Dead shards (SNR <3dB): {dead_s}/{n_shards} ({dead_s/n_shards:.0%})")
    print(f"Weak shards (SNR <6dB): {weak_s}/{n_shards} ({weak_s/n_shards:.0%})")
    print(f"SNR range: {snr['snr_db'].min():.1f} to {snr['snr_db'].max():.1f} dB")

    # Method 2b: SNR after Q4_K
    snr4 = detect_dead_shards_by_snr(W, W_q4, n_shards)
    dead_s4 = snr4["dead_mask"].sum()
    print(f"Dead shards at Q4_K (SNR <3dB): {dead_s4}/{n_shards}")

    # Method 3: Output contribution
    print("\n--- Method 3: Output Contribution ---")
    contrib = detect_dead_shards_by_output(W, inputs, n_shards)
    dead_c = contrib["dead_mask"].sum()
    weak_c = contrib["weak_mask"].sum()
    print(f"Dead shards (contrib <10% of uniform): {dead_c}/{n_shards} ({dead_c/n_shards:.0%})")
    print(f"Weak shards (contrib <50% of uniform): {weak_c}/{n_shards} ({weak_c/n_shards:.0%})")

    # Combined verdict
    print("\n--- Combined Verdict ---")
    all_dead = energy["dead_mask"] & snr["dead_mask"] & contrib["dead_mask"]
    any_dead = energy["dead_mask"] | snr["dead_mask"] | contrib["dead_mask"]
    consensus_dead = (energy["dead_mask"].astype(int) +
                      snr["dead_mask"].astype(int) +
                      contrib["dead_mask"].astype(int)) >= 2

    print(f"Dead by ALL 3 methods: {all_dead.sum()}/{n_shards}")
    print(f"Dead by ANY method: {any_dead.sum()}/{n_shards}")
    print(f"Dead by 2/3 consensus: {consensus_dead.sum()}/{n_shards}")

    # VRAM savings from removing dead shards
    shard_mb = shard_size * in_dim * 2 / 1e6  # fp16
    savings_mb = consensus_dead.sum() * shard_mb
    total_mb = n_shards * shard_mb
    print(f"\nVRAM savings from removing consensus-dead shards:")
    print(f"  {consensus_dead.sum()} dead × {shard_mb:.2f}MB = {savings_mb:.1f}MB saved")
    print(f"  ({savings_mb/total_mb:.0%} of tensor)")

    # Per-shard classification
    print(f"\n--- Shard Classification ---")
    classifications = {
        "sacred": 0, "active": 0, "weak": 0, "dead": 0, "remove": 0
    }

    for i in range(n_shards):
        e = energy["relative"][i]
        s = snr["snr_db"][i]
        c = contrib["contributions"][i]
        uniform = 1.0 / n_shards

        if e > uniform * 3 and s > 20:
            classifications["sacred"] += 1
        elif consensus_dead[i]:
            classifications["remove"] += 1
        elif any_dead[i]:
            classifications["dead"] += 1
        elif energy["weak_mask"][i] or snr["weak_mask"][i]:
            classifications["weak"] += 1
        else:
            classifications["active"] += 1

    print(f"  Sacred (high energy + high SNR):  {classifications['sacred']:4d} → Q8_0 in VRAM")
    print(f"  Active (normal):                  {classifications['active']:4d} → Q4_K in RAM")
    print(f"  Weak (low energy or low SNR):     {classifications['weak']:4d} → Q2_K in RAM")
    print(f"  Dead (2/3 methods flag it):        {classifications['dead']:4d} → Q1_0 or zero")
    print(f"  Remove (all 3 methods flag it):    {classifications['remove']:4d} → zeroed out")

    # Project to full model
    print(f"\n--- Full Model Projection (27B, ~700 tensors) ---")
    n_tensors = 700
    dead_pct = consensus_dead.sum() / n_shards

    total_shards = n_tensors * n_shards
    projected_dead = int(total_shards * dead_pct)
    projected_sacred = int(total_shards * classifications["sacred"] / n_shards)

    print(f"Total micro-shards: {total_shards:,}")
    print(f"Projected dead: {projected_dead:,} ({dead_pct:.0%})")
    print(f"Projected sacred: {projected_sacred:,}")
    print(f"Projected savings: {projected_dead * shard_mb:.0f}MB")
    print(f"VRAM needed for sacred only: {projected_sacred * shard_mb:.0f}MB")

    return classifications


def profile_gguf_shards(gguf_path, ref_gguf_path, n_shards, output_path):
    """Profile dead shards in a real quantized GGUF by comparing to reference."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    print(f"Loading quantized: {gguf_path}")
    reader_q = GGUFReader(gguf_path)

    if ref_gguf_path:
        print(f"Loading reference: {ref_gguf_path}")
        reader_ref = GGUFReader(ref_gguf_path)
        ref_tensors = {t.name: t for t in reader_ref.tensors}
    else:
        ref_tensors = None

    results = {"model": gguf_path, "n_shards": n_shards, "tensors": {}}

    target_keys = ['ffn_down', 'ffn_up', 'ffn_gate', 'attn_q.', 'attn_k.', 'attn_v.',
                   'attn_output', 'attn_gate']

    total_dead = 0
    total_shards = 0
    total_sacred = 0

    for tensor in reader_q.tensors:
        if len(tensor.shape) != 2:
            continue
        if not any(k in tensor.name for k in target_keys):
            continue

        rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
        if rows < n_shards:
            continue

        W_q = tensor.data.astype(np.float32)
        if W_q.ndim == 1:
            W_q = W_q.reshape(rows, cols)

        actual_shards = min(n_shards, rows)
        shard_size = rows // actual_shards

        # Method 1: energy
        energy = detect_dead_shards_by_energy(W_q, actual_shards)

        # Method 2: SNR (only if we have reference)
        snr = None
        if ref_tensors and tensor.name in ref_tensors:
            W_ref = ref_tensors[tensor.name].data.astype(np.float32)
            if W_ref.ndim == 1:
                W_ref = W_ref.reshape(rows, cols)
            snr = detect_dead_shards_by_snr(W_ref, W_q, actual_shards)

        # Method 3: output contribution
        np.random.seed(42)
        inputs = np.maximum(0, np.random.randn(256, cols).astype(np.float32) * 0.1)
        contrib = detect_dead_shards_by_output(W_q, inputs, actual_shards)

        # Classify each shard
        shard_classes = []
        for i in range(actual_shards):
            e = energy["relative"][i]
            c = contrib["contributions"][i]
            uniform = 1.0 / actual_shards

            dead_votes = 0
            if energy["dead_mask"][i]:
                dead_votes += 1
            if snr and snr["dead_mask"][i]:
                dead_votes += 1
            if contrib["dead_mask"][i]:
                dead_votes += 1

            sacred_score = 0
            if e > uniform * 3:
                sacred_score += 1
            if snr and snr["snr_db"][i] > 20:
                sacred_score += 1
            if c > uniform * 2:
                sacred_score += 1

            if sacred_score >= 2:
                cls = "sacred"
                total_sacred += 1
            elif dead_votes >= 2:
                cls = "dead"
                total_dead += 1
            elif energy["weak_mask"][i] or contrib["weak_mask"][i]:
                cls = "weak"
            else:
                cls = "active"
            shard_classes.append(cls)

        total_shards += actual_shards
        n_dead = shard_classes.count("dead")
        n_sacred = shard_classes.count("sacred")

        if n_dead > 0 or n_sacred > 0:
            shard_mb = shard_size * cols * 2 / 1e6
            print(f"{tensor.name:50s} dead={n_dead:3d} sacred={n_sacred:3d} "
                  f"({shard_mb:.1f}MB/shard)")

        results["tensors"][tensor.name] = {
            "shape": [rows, cols],
            "n_shards": actual_shards,
            "classifications": shard_classes,
            "dead_count": n_dead,
            "sacred_count": n_sacred,
            "energy_relative": [round(float(e), 6) for e in energy["relative"]],
        }

    print(f"\n{'='*60}")
    print(f"Total shards: {total_shards}")
    print(f"Dead shards: {total_dead} ({total_dead/max(total_shards,1):.1%})")
    print(f"Sacred shards: {total_sacred} ({total_sacred/max(total_shards,1):.1%})")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Detect dead micro-shards in quantized models")
    parser.add_argument("--gguf", help="Quantized GGUF path")
    parser.add_argument("--reference-gguf", help="Reference (fp16/high-quant) GGUF for SNR")
    parser.add_argument("--n-shards", type=int, default=128)
    parser.add_argument("--output", help="Output JSON")
    parser.add_argument("--synthetic", action="store_true")

    args = parser.parse_args()

    if args.synthetic or not args.gguf:
        synthetic_test(n_shards=args.n_shards)
    else:
        profile_gguf_shards(args.gguf, args.reference_gguf, args.n_shards, args.output)


if __name__ == "__main__":
    main()
