#!/usr/bin/env python3
"""Precision Patch Feasibility — can we SVD-compress Q2K→Q5K weight deltas?

Reads two GGUFs (Q2_K and Q5_K of the same model), dequantizes matching
tensors, computes the delta, and measures SVD compressibility.

If a rank-64 SVD captures 95%+ of the delta energy, the precision patch
concept is viable: store compressed patches in RAM (~5MB each), stream
to VRAM on demand.

Usage:
    python scripts/precision_patch_feasibility.py \
        --q2k model-q2k.gguf \
        --q5k model-q5k.gguf \
        --tensors "blk.63.ffn_down.weight,blk.0.ffn_up.weight" \
        --ranks 8,16,32,64,128,256

    # Without GGUFs — synthetic test with random weight-like matrices
    python scripts/precision_patch_feasibility.py --synthetic
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def synthetic_delta_test(shape=(17408, 5120), ranks=None):
    """Test SVD compression on synthetic weight deltas.

    Simulates the Q2K→Q5K delta: small, structured noise that represents
    quantization error. Real quantization error has structure (it's not
    random) because quantization rounds to grid points.
    """
    if ranks is None:
        ranks = [4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\nSynthetic delta test: {shape[0]}x{shape[1]} matrix")
    print(f"Full delta size: {shape[0] * shape[1] * 4 / 1e6:.1f} MB (float32)")
    print(f"Full delta size: {shape[0] * shape[1] * 2 / 1e6:.1f} MB (float16)")

    np.random.seed(42)

    # Simulate a realistic weight matrix
    # Real weights have low-rank structure + noise
    true_rank = 256
    U = np.random.randn(shape[0], true_rank).astype(np.float32) * 0.01
    V = np.random.randn(true_rank, shape[1]).astype(np.float32) * 0.01
    W = U @ V + np.random.randn(*shape).astype(np.float32) * 0.001

    # Simulate Q5_K quantization (fine grid, ~5.5 bpw)
    scale_q5 = np.abs(W).max(axis=1, keepdims=True) / 15.5
    W_q5 = np.round(W / scale_q5) * scale_q5

    # Simulate Q2_K quantization (coarse grid, ~2.5 bpw)
    scale_q2 = np.abs(W).max(axis=1, keepdims=True) / 3.5
    W_q2 = np.round(W / scale_q2) * scale_q2

    # The delta we want to compress
    delta = W_q5 - W_q2

    delta_norm = np.linalg.norm(delta, 'fro')
    delta_energy = delta_norm ** 2
    print(f"Delta Frobenius norm: {delta_norm:.4f}")
    print(f"Delta mean abs: {np.abs(delta).mean():.6f}")
    print(f"Delta max abs: {np.abs(delta).max():.6f}")

    # SVD of the delta
    print(f"\nComputing SVD of delta ({shape[0]}x{shape[1]})...")
    t0 = time.time()
    U_svd, S, Vt = np.linalg.svd(delta, full_matrices=False)
    svd_time = time.time() - t0
    print(f"SVD took {svd_time:.1f}s")

    # Energy capture at each rank
    cumulative_energy = np.cumsum(S ** 2) / delta_energy

    print(f"\n{'Rank':>6s}  {'Energy%':>8s}  {'Patch MB':>10s}  {'Compress':>10s}  {'Viable?':>8s}")
    print("-" * 55)

    full_size_mb = shape[0] * shape[1] * 2 / 1e6  # float16
    for r in ranks:
        if r > len(S):
            break
        energy_pct = cumulative_energy[r - 1] * 100
        # Patch size: U[:, :r] (shape[0]*r) + S[:r] (r) + Vt[:r, :] (r*shape[1]) in float16
        patch_bytes = (shape[0] * r + r + r * shape[1]) * 2  # float16
        patch_mb = patch_bytes / 1e6
        compression = full_size_mb / patch_mb
        viable = "YES" if energy_pct > 90 and patch_mb < 50 else ""
        print(f"{r:6d}  {energy_pct:7.2f}%  {patch_mb:9.1f}MB  {compression:9.1f}x  {viable:>8s}")

    # Find rank needed for 95% and 99%
    for target in [0.90, 0.95, 0.99]:
        idx = np.searchsorted(cumulative_energy, target)
        if idx < len(S):
            r = idx + 1
            patch_bytes = (shape[0] * r + r + r * shape[1]) * 2
            patch_mb = patch_bytes / 1e6
            pcie_ms = patch_mb / 14000  # 14 GB/s PCIe 3.0
            print(f"\n  {target*100:.0f}% energy: rank {r}, "
                  f"patch = {patch_mb:.1f}MB, "
                  f"PCIe transfer = {pcie_ms*1000:.1f}ms, "
                  f"compression = {full_size_mb/patch_mb:.1f}x")

    # Reconstruction error at different ranks
    print(f"\n{'Rank':>6s}  {'RMSE':>10s}  {'MaxErr':>10s}  {'RelErr%':>10s}")
    print("-" * 45)
    for r in [16, 32, 64, 128, 256]:
        if r > len(S):
            break
        reconstructed = U_svd[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        error = delta - reconstructed
        rmse = np.sqrt(np.mean(error ** 2))
        maxerr = np.abs(error).max()
        relerr = np.linalg.norm(error, 'fro') / delta_norm * 100
        print(f"{r:6d}  {rmse:10.6f}  {maxerr:10.6f}  {relerr:9.2f}%")


def gguf_delta_test(q2k_path, q5k_path, tensor_names, ranks=None):
    """Test SVD compression on real GGUF weight deltas.

    Requires gguf Python package (pip install gguf).
    """
    if ranks is None:
        ranks = [8, 16, 32, 64, 128, 256]

    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf  (needed to read GGUF files)")
        sys.exit(1)

    print(f"Loading Q2_K: {q2k_path}")
    reader_q2 = GGUFReader(q2k_path)
    print(f"Loading Q5_K: {q5k_path}")
    reader_q5 = GGUFReader(q5k_path)

    # Build tensor lookup
    tensors_q2 = {t.name: t for t in reader_q2.tensors}
    tensors_q5 = {t.name: t for t in reader_q5.tensors}

    results = {}
    for name in tensor_names:
        if name not in tensors_q2 or name not in tensors_q5:
            print(f"SKIP {name} — not found in both GGUFs")
            continue

        t_q2 = tensors_q2[name]
        t_q5 = tensors_q5[name]

        # Dequantize to float32
        print(f"\n{'='*60}")
        print(f"Tensor: {name}")
        print(f"  Q2_K shape: {t_q2.shape}, type: {t_q2.tensor_type}")
        print(f"  Q5_K shape: {t_q5.shape}, type: {t_q5.tensor_type}")

        w_q2 = t_q2.data.astype(np.float32)
        w_q5 = t_q5.data.astype(np.float32)

        if w_q2.ndim == 1:
            print("  1D tensor, skipping SVD")
            continue

        delta = w_q5 - w_q2
        delta_norm = np.linalg.norm(delta, 'fro')
        delta_energy = delta_norm ** 2

        print(f"  Delta norm: {delta_norm:.4f}")
        print(f"  Delta mean abs: {np.abs(delta).mean():.6f}")

        # SVD
        t0 = time.time()
        U_svd, S, Vt = np.linalg.svd(delta, full_matrices=False)
        print(f"  SVD: {time.time()-t0:.1f}s")

        cumulative = np.cumsum(S ** 2) / delta_energy
        shape = delta.shape
        full_mb = shape[0] * shape[1] * 2 / 1e6

        tensor_results = {"shape": list(shape), "ranks": {}}
        print(f"\n  {'Rank':>6s}  {'Energy%':>8s}  {'Patch MB':>10s}  {'Compress':>10s}")
        print(f"  {'-'*45}")

        for r in ranks:
            if r > len(S):
                break
            energy_pct = cumulative[r - 1] * 100
            patch_mb = (shape[0] * r + r + r * shape[1]) * 2 / 1e6
            compression = full_mb / patch_mb
            print(f"  {r:6d}  {energy_pct:7.2f}%  {patch_mb:9.1f}MB  {compression:9.1f}x")
            tensor_results["ranks"][str(r)] = {
                "energy_pct": round(energy_pct, 2),
                "patch_mb": round(patch_mb, 2),
                "compression": round(compression, 1),
            }

        for target in [0.95, 0.99]:
            idx = np.searchsorted(cumulative, target)
            if idx < len(S):
                r = idx + 1
                patch_mb = (shape[0] * r + r + r * shape[1]) * 2 / 1e6
                pcie_ms = patch_mb / 14
                print(f"\n  {target*100:.0f}%: rank {r}, {patch_mb:.1f}MB, "
                      f"PCIe = {pcie_ms:.1f}ms")
                tensor_results[f"rank_{int(target*100)}pct"] = r

        results[name] = tensor_results

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test SVD compressibility of quantization deltas"
    )
    parser.add_argument("--q2k", help="Q2_K GGUF path")
    parser.add_argument("--q5k", help="Q5_K GGUF path")
    parser.add_argument("--tensors", help="Comma-separated tensor names to test")
    parser.add_argument("--ranks", default="8,16,32,64,128,256",
                        help="Comma-separated ranks to test")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic test (no GGUFs needed)")
    parser.add_argument("--output", help="Save results to JSON")

    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]

    if args.synthetic or (not args.q2k and not args.q5k):
        synthetic_delta_test(ranks=ranks)
        # Also test smaller tensor sizes
        print("\n" + "=" * 60)
        print("Small tensor test (5120x1024 — like attn_k):")
        synthetic_delta_test(shape=(5120, 1024), ranks=ranks)
        return

    if not args.q2k or not args.q5k:
        print("Need both --q2k and --q5k, or use --synthetic")
        sys.exit(1)

    tensor_names = [t.strip() for t in args.tensors.split(",")]
    results = gguf_delta_test(args.q2k, args.q5k, tensor_names, ranks)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
