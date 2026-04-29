#!/usr/bin/env python3
"""Shard Activation Profiler — records which row-blocks of each tensor
contribute most to the output during inference.

This produces the ground-truth training data for the cerebellum router:
for each token, which micro-shards are "active" (contribute >threshold
of output magnitude)?

Approach: Load a GGUF, dequantize weights, run synthetic forward passes
through individual MLP/attention blocks, measure per-row-block activation
magnitude.

Usage:
    # Profile a GGUF model with calibration text
    python scripts/shard_activation_profiler.py \
        --gguf model.gguf \
        --n-shards 128 \
        --calibration wikitext.txt \
        --output shard_activations.json

    # Synthetic test (no GGUF needed)
    python scripts/shard_activation_profiler.py --synthetic
"""
import argparse
import json
import sys
import time

import numpy as np


def analyze_row_block_activation(weight, inputs, n_shards):
    """For a weight matrix W (out_dim x in_dim) and a batch of input vectors,
    compute what fraction of the output each row-block is responsible for.

    Returns: (n_shards,) array of mean activation fractions.
    """
    out_dim, in_dim = weight.shape
    shard_size = out_dim // n_shards
    remainder = out_dim % n_shards

    output = inputs @ weight.T  # (batch, out_dim)
    output_magnitude = np.abs(output)
    total_magnitude = output_magnitude.sum(axis=1, keepdims=True) + 1e-10

    activations = np.zeros(n_shards)
    for i in range(n_shards):
        start = i * shard_size + min(i, remainder)
        end = (i + 1) * shard_size + min(i + 1, remainder)
        shard_mag = output_magnitude[:, start:end].sum(axis=1)
        activations[i] = (shard_mag / total_magnitude.squeeze()).mean()

    return activations


def synthetic_test(n_shards=128):
    """Test shard activation patterns on synthetic MLP weights.

    Key question: do real-ish weights have sparse activation patterns
    (most shards dormant) or uniform patterns (all shards equally active)?
    """
    print(f"\n=== Synthetic Shard Activation Test ({n_shards} shards) ===\n")

    np.random.seed(42)

    # Simulate ffn_down weight (17408 x 5120) — but smaller for speed
    out_dim, in_dim = 4096, 2048
    print(f"Weight shape: {out_dim}x{in_dim}")
    print(f"Shards: {n_shards} ({out_dim // n_shards} rows each)")

    # Low-rank weight (realistic — transformer weights have low effective rank)
    true_rank = 64
    U = np.random.randn(out_dim, true_rank).astype(np.float32) * 0.02
    V = np.random.randn(true_rank, in_dim).astype(np.float32) * 0.02
    W = U @ V + np.random.randn(out_dim, in_dim).astype(np.float32) * 0.001

    # Different input distributions
    test_cases = {
        "uniform_random": np.random.randn(256, in_dim).astype(np.float32),
        "sparse_input": (np.random.randn(256, in_dim).astype(np.float32)
                         * (np.random.rand(256, in_dim) > 0.9)),
        "clustered_input": np.random.randn(256, in_dim // 8).astype(np.float32)
                           @ np.random.randn(in_dim // 8, in_dim).astype(np.float32) * 0.1,
        "post_relu": np.maximum(0, np.random.randn(256, in_dim).astype(np.float32)),
    }

    print(f"\n{'Input Type':25s} | {'Active Shards':>14s} | {'Top-10% Load':>12s} | "
          f"{'Gini':>6s} | {'Sparsity':>8s}")
    print("-" * 80)

    results = {}
    for name, inputs in test_cases.items():
        activations = analyze_row_block_activation(W, inputs, n_shards)

        # How many shards carry >1% of output?
        threshold = 1.0 / n_shards  # uniform would be exactly this
        active_count = np.sum(activations > threshold * 0.5)

        # Top 10% of shards — what fraction of output do they carry?
        top_k = max(1, n_shards // 10)
        sorted_act = np.sort(activations)[::-1]
        top_10_load = sorted_act[:top_k].sum()

        # Gini coefficient (0 = perfectly equal, 1 = one shard does everything)
        sorted_vals = np.sort(activations)
        n = len(sorted_vals)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) / (n * np.sum(sorted_vals)) - (n + 1) / n)

        # Sparsity: fraction of shards below 50% of uniform expectation
        sparsity = np.mean(activations < threshold * 0.5)

        results[name] = {
            "activations": activations.tolist(),
            "active_shards": int(active_count),
            "top_10_load": float(top_10_load),
            "gini": float(gini),
            "sparsity": float(sparsity),
        }

        print(f"{name:25s} | {active_count:4d}/{n_shards:4d}      | "
              f"{top_10_load:11.1%} | {gini:5.3f} | {sparsity:7.1%}")

    # Test with ReLU-activated weight (like actual MLP)
    print(f"\n--- After ReLU activation (more realistic) ---")
    W_relu = np.maximum(0, W)  # dead neurons
    dead_rows = np.sum(np.all(W_relu == 0, axis=1))
    print(f"Dead rows after ReLU: {dead_rows}/{out_dim} ({dead_rows/out_dim:.1%})")

    inputs = test_cases["post_relu"]
    act_relu = analyze_row_block_activation(W_relu, inputs, n_shards)
    active = np.sum(act_relu > 1.0 / n_shards * 0.5)
    print(f"Active shards: {active}/{n_shards}")

    # Key finding: what shard count gives best sparsity?
    print(f"\n--- Optimal shard count (weight: {out_dim}x{in_dim}) ---")
    inputs = test_cases["post_relu"]
    print(f"{'Shards':>8s} | {'Rows/Shard':>10s} | {'Active':>8s} | {'Active%':>8s} | {'Shard MB':>10s}")
    print("-" * 55)
    for ns in [4, 8, 16, 32, 64, 128, 256]:
        if out_dim % ns != 0 and out_dim < ns:
            continue
        act = analyze_row_block_activation(W, inputs, ns)
        threshold = 1.0 / ns * 0.5
        active = np.sum(act > threshold)
        rows_per = out_dim // ns
        shard_mb = rows_per * in_dim * 2 / 1e6  # fp16
        print(f"{ns:8d} | {rows_per:10d} | {active:4d}/{ns:4d} | "
              f"{active/ns:7.1%} | {shard_mb:9.2f}MB")

    # VRAM savings projection
    print(f"\n--- VRAM Savings Projection (70B model, 4GB VRAM target) ---")
    total_params_70b = 70e9
    param_bytes_fp16 = total_params_70b * 2
    total_gb = param_bytes_fp16 / 1e9
    print(f"Full model: {total_gb:.0f}GB fp16")

    for activation_rate in [0.05, 0.10, 0.15, 0.20, 0.30]:
        active_gb = total_gb * activation_rate
        sacred_gb = 2.0  # always pinned
        vram_needed = sacred_gb + active_gb * 0.3  # scratch buffers
        print(f"  {activation_rate:5.0%} activation: {active_gb:.1f}GB active, "
              f"~{vram_needed:.1f}GB VRAM needed "
              f"({'FITS 4GB' if vram_needed <= 4 else 'needs ' + str(int(vram_needed)) + 'GB'})")

    return results


def profile_gguf(gguf_path, n_shards, calibration_path, output_path):
    """Profile real GGUF tensor activations."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    print(f"Loading {gguf_path}...")
    reader = GGUFReader(gguf_path)

    # Load calibration text and tokenize (simplified — use random vectors as proxy)
    print(f"Generating calibration inputs (random proxy)...")
    np.random.seed(42)

    results = {}
    for tensor in reader.tensors:
        if tensor.data.ndim != 2:
            continue
        if not any(k in tensor.name for k in ['ffn_down', 'ffn_up', 'ffn_gate',
                                                'attn_q', 'attn_k', 'attn_v',
                                                'attn_output']):
            continue

        W = tensor.data.astype(np.float32)
        out_dim, in_dim = W.shape

        actual_shards = min(n_shards, out_dim)
        if actual_shards < 2:
            continue

        inputs = np.random.randn(128, in_dim).astype(np.float32)
        inputs = np.maximum(0, inputs)  # post-ReLU proxy

        activations = analyze_row_block_activation(W, inputs, actual_shards)

        threshold = 1.0 / actual_shards * 0.5
        active_count = int(np.sum(activations > threshold))

        results[tensor.name] = {
            "shape": list(W.shape),
            "n_shards": actual_shards,
            "active_shards": active_count,
            "activation_rate": active_count / actual_shards,
            "activations": activations.tolist(),
        }

        print(f"  {tensor.name:45s} {out_dim:6d}x{in_dim:5d} "
              f"active={active_count}/{actual_shards} "
              f"({active_count/actual_shards:.0%})")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile micro-shard activation patterns")
    parser.add_argument("--gguf", help="GGUF model path")
    parser.add_argument("--n-shards", type=int, default=128, help="Shards per tensor")
    parser.add_argument("--calibration", help="Calibration text file")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic test")

    args = parser.parse_args()

    if args.synthetic or not args.gguf:
        results = synthetic_test(n_shards=args.n_shards)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        profile_gguf(args.gguf, args.n_shards, args.calibration, args.output)


if __name__ == "__main__":
    main()
