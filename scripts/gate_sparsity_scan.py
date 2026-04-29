#!/usr/bin/env python3
"""SwiGLU Gate Sparsity Scanner — the REAL brain activation map.

In SwiGLU MLPs: output = down_proj(silu(gate_proj(x)) * up_proj(x))
The gate output determines which neurons are ON vs OFF.
This is where the real sparsity lives — not in the linear projections.

Hooks into gate_proj output, measures per-shard gate activation:
- Which shards have their gate "open" (silu output > threshold)?
- Does gate sparsity vary by prompt type (code vs math vs creative)?
- Which shards are permanently gated off (dead neurons)?

Usage:
    python scripts/gate_sparsity_scan.py \
        --model /path/to/model \
        --n-shards 128 \
        --device cpu \
        --output gate_sparsity.json
"""
import argparse
import json
import gc
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def build_calibration_prompts():
    return {
        "code": (
            "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n"
            "        return arr\n    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n```"
        ),
        "math": (
            "The integral of sin(x)dx from 0 to pi equals 2. To prove this, "
            "we use the antiderivative -cos(x), evaluating at the bounds: "
            "-cos(pi) - (-cos(0)) = -(-1) - (-1) = 1 + 1 = 2."
        ),
        "conversation": (
            "Hey! How's it going? I was thinking we could grab lunch tomorrow "
            "at that new Thai place downtown. Sarah mentioned their pad thai "
            "is amazing. What do you think, around noon?"
        ),
        "creative": (
            "The ancient lighthouse stood sentinel against the storm, its beam "
            "cutting through sheets of rain like a golden sword. Below, the "
            "waves crashed against obsidian rocks."
        ),
        "technical": (
            "The transformer architecture uses multi-head self-attention with "
            "query, key, and value projections. Each attention head computes "
            "scaled dot-product attention: softmax(QK^T / sqrt(d_k))V."
        ),
        "reasoning": (
            "If all roses are flowers, and some flowers fade quickly, can we "
            "conclude that some roses fade quickly? No - this is the fallacy "
            "of the undistributed middle."
        ),
    }


class GateSparsityRecorder:
    def __init__(self, model, n_shards=128):
        self.n_shards = n_shards
        self.gate_recordings = defaultdict(list)
        self.hooks = []
        self._install_hooks(model)

    def _install_hooks(self, model):
        for name, module in model.named_modules():
            if '.mlp.gate_proj' in name and hasattr(module, 'weight'):
                hook = module.register_forward_hook(self._make_gate_hook(name))
                self.hooks.append(hook)
        print(f"Installed {len(self.hooks)} gate hooks")

    def _make_gate_hook(self, layer_name):
        def hook_fn(module, input_data, output):
            with torch.no_grad():
                gate_values = F.silu(output[0]).cpu().numpy()
                for t in range(gate_values.shape[0]):
                    gate = gate_values[t]
                    out_dim = gate.shape[0]
                    n_shards = min(self.n_shards, out_dim)
                    shard_size = out_dim // n_shards
                    shard_gate_activity = np.zeros(n_shards)
                    shard_gate_on_fraction = np.zeros(n_shards)
                    for i in range(n_shards):
                        start = i * shard_size
                        end = min(start + shard_size, out_dim)
                        shard_vals = gate[start:end]
                        shard_gate_activity[i] = np.abs(shard_vals).mean()
                        shard_gate_on_fraction[i] = np.mean(np.abs(shard_vals) > 0.01)
                    self.gate_recordings[layer_name].append({
                        "activity": shard_gate_activity,
                        "on_fraction": shard_gate_on_fraction,
                    })
        return hook_fn

    def clear(self):
        self.gate_recordings.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self):
        summary = {}
        for layer_name, token_recordings in self.gate_recordings.items():
            if not token_recordings:
                continue
            activities = np.array([r["activity"] for r in token_recordings])
            on_fractions = np.array([r["on_fraction"] for r in token_recordings])
            n_tokens, n_shards = activities.shape
            mean_activity = activities.mean(axis=0)
            mean_on_frac = on_fractions.mean(axis=0)
            activity_std = activities.std(axis=0)
            gated_off = int(np.sum(mean_on_frac < 0.3))
            gated_low = int(np.sum(mean_on_frac < 0.5))
            fully_on = int(np.sum(mean_on_frac > 0.9))
            per_token_off = np.sum(on_fractions < 0.3, axis=1)
            sorted_vals = np.sort(mean_activity)
            n = len(sorted_vals)
            denom = n * np.sum(sorted_vals) + 1e-10
            gini = float(2 * np.sum(np.arange(1, n+1) * sorted_vals) / denom - (n+1)/n)
            activity_cv = activity_std / (mean_activity + 1e-10)
            summary[layer_name] = {
                "n_tokens": n_tokens,
                "n_shards": n_shards,
                "mean_gate_activity": mean_activity.tolist(),
                "mean_on_fraction": mean_on_frac.tolist(),
                "activity_std": activity_std.tolist(),
                "gated_off_shards": gated_off,
                "gated_low_shards": gated_low,
                "fully_on_shards": fully_on,
                "gini_gate_activity": gini,
                "mean_off_per_token": float(per_token_off.mean()),
                "max_off_per_token": int(per_token_off.max()),
                "activity_cv": activity_cv.tolist(),
            }
        return summary


def run_gate_scan(model_path, n_shards, device, max_tokens, output_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path} on {device}...")
    t0 = time.time()
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = device if device == "cuda" else None
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map=device_map,
    )
    if device == "cpu":
        model = model.float()
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    recorder = GateSparsityRecorder(model, n_shards=n_shards)
    prompts = build_calibration_prompts()
    results = {"model": model_path, "n_shards": n_shards, "prompt_types": {}}

    for prompt_type, text in prompts.items():
        print(f"\n--- Gate scan: {prompt_type} ---")
        recorder.clear()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        n_tokens = inputs["input_ids"].shape[1]
        print(f"  Tokens: {n_tokens}")
        t0 = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        print(f"  Forward pass: {time.time()-t0:.1f}s")
        summary = recorder.get_summary()
        results["prompt_types"][prompt_type] = summary
        for layer_name, data in sorted(summary.items()):
            off = data["gated_off_shards"]
            low = data["gated_low_shards"]
            full = data["fully_on_shards"]
            gini = data["gini_gate_activity"]
            mean_off = data["mean_off_per_token"]
            max_off = data["max_off_per_token"]
            print(f"  {layer_name:45s} off={off:3d} low={low:3d} on={full:3d} "
                  f"gini={gini:.3f} off/tok={mean_off:.1f} max_off={max_off}")

    recorder.remove_hooks()

    # Cross-prompt analysis
    print(f"\n{'='*80}")
    print("GATE SPARSITY CROSS-PROMPT ANALYSIS")
    print(f"{'='*80}\n")

    all_layers = set()
    for pdata in results["prompt_types"].values():
        all_layers.update(pdata.keys())

    print(f"{'Layer':45s} {'Avg Off':>8s} {'Avg Low':>8s} {'Avg Gini':>9s} {'Off Range':>12s}")
    print("-" * 90)
    for layer in sorted(all_layers):
        off_values = []
        low_values = []
        gini_values = []
        for ptype, pdata in results["prompt_types"].items():
            if layer in pdata:
                off_values.append(pdata[layer]["gated_off_shards"])
                low_values.append(pdata[layer]["gated_low_shards"])
                gini_values.append(pdata[layer]["gini_gate_activity"])
        if off_values:
            print(f"{layer:45s} {np.mean(off_values):8.1f} {np.mean(low_values):8.1f} "
                  f"{np.mean(gini_values):9.3f} {min(off_values)}-{max(off_values):>8s}")

    # Verdict
    total_off = sum(d["gated_off_shards"] for pd in results["prompt_types"].values() for d in pd.values())
    total_shards = sum(d["n_shards"] for pd in results["prompt_types"].values() for d in pd.values())
    off_pct = total_off / max(total_shards, 1) * 100

    print(f"\nOverall: {total_off}/{total_shards} shards gated off ({off_pct:.1f}%)")
    if off_pct > 10:
        print("VERDICT: SIGNIFICANT gate sparsity! Cerebellum routing IS viable.")
    elif off_pct > 3:
        print("VERDICT: MODERATE gate sparsity. Some routing benefit possible.")
    else:
        print("VERDICT: LOW gate sparsity. Dense model uses most neurons.")

    if output_path:
        class NE(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, (np.integer,)): return int(obj)
                return super().default(obj)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NE)
        print(f"\nSaved to {output_path}")

    del model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="SwiGLU Gate Sparsity Scanner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-shards", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", help="Output JSON")
    args = parser.parse_args()
    run_gate_scan(args.model, args.n_shards, args.device, args.max_tokens, args.output)


if __name__ == "__main__":
    main()
