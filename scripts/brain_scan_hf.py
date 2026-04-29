#!/usr/bin/env python3
"""LLM Brain Scan via HuggingFace — real activation recording with hooks.

Runs actual inference on calibration text, records per-shard activation
magnitudes at every layer for every token. This is the real fMRI — not
random input vectors, but actual hidden states flowing through the network.

Key question: do micro-shards show INPUT-DEPENDENT sparsity? i.e., do
different tokens activate different shards?

Usage:
    python scripts/brain_scan_hf.py \
        --model /path/to/model \
        --n-shards 128 \
        --device cpu \
        --output brain_scan_real.json

    # Quick test with fewer tokens
    python scripts/brain_scan_hf.py --model /path/to/model --max-tokens 32
"""
import argparse
import json
import gc
import sys
import time
from collections import defaultdict

import numpy as np
import torch


def build_calibration_prompts():
    """Different prompt types to test input-dependent shard activation."""
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
            "waves crashed against obsidian rocks, sending plumes of salt spray "
            "into the howling darkness."
        ),
        "technical": (
            "The transformer architecture uses multi-head self-attention with "
            "query, key, and value projections. Each attention head computes "
            "scaled dot-product attention: softmax(QK^T / sqrt(d_k))V, where "
            "d_k is the key dimension."
        ),
        "reasoning": (
            "If all roses are flowers, and some flowers fade quickly, can we "
            "conclude that some roses fade quickly? No - this is the fallacy "
            "of the undistributed middle. The flowers that fade quickly might "
            "not include any roses."
        ),
    }


class ShardActivationRecorder:
    """Hook-based recorder that captures per-shard activation magnitudes."""

    def __init__(self, model, n_shards=128):
        self.n_shards = n_shards
        self.recordings = defaultdict(list)
        self.hooks = []
        self._install_hooks(model)

    def _install_hooks(self, model):
        target_names = [
            '.mlp.down_proj', '.mlp.up_proj', '.mlp.gate_proj',
            '.self_attn.q_proj', '.self_attn.k_proj',
            '.self_attn.v_proj', '.self_attn.o_proj',
        ]
        for name, module in model.named_modules():
            if any(k in name for k in target_names):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
        print(f"Installed {len(self.hooks)} activation hooks")

    def _make_hook(self, layer_name):
        def hook_fn(module, input_data, output):
            with torch.no_grad():
                if output.dim() == 3:
                    # (batch, seq_len, features) — record ALL token positions
                    acts = output[0].float().cpu().numpy()  # (seq_len, features)
                elif output.dim() == 2:
                    acts = output[0:1].float().cpu().numpy()  # (1, features)
                else:
                    return

                for t in range(acts.shape[0]):
                    act = acts[t]
                    out_dim = act.shape[0]
                    n_shards = min(self.n_shards, out_dim)
                    shard_size = out_dim // n_shards

                    shard_magnitudes = np.zeros(n_shards)
                    total_mag = np.abs(act).sum() + 1e-10

                    for i in range(n_shards):
                        start = i * shard_size
                        end = min(start + shard_size, out_dim)
                        shard_magnitudes[i] = np.abs(act[start:end]).sum() / total_mag

                    self.recordings[layer_name].append(shard_magnitudes)
        return hook_fn

    def clear(self):
        self.recordings.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self):
        summary = {}
        for layer_name, token_activations in self.recordings.items():
            if not token_activations:
                continue

            acts = np.array(token_activations)
            n_tokens, n_shards = acts.shape
            uniform = 1.0 / n_shards

            mean_act = acts.mean(axis=0)
            std_act = acts.std(axis=0)

            per_token_active = np.sum(acts > uniform * 0.5, axis=1)
            activation_variance = std_act / (mean_act + 1e-10)
            never_active = np.all(acts < uniform * 0.2, axis=0)
            always_active = np.all(acts > uniform * 1.5, axis=0)

            sorted_vals = np.sort(mean_act)
            n = len(sorted_vals)
            denom = n * np.sum(sorted_vals) + 1e-10
            gini = float(2 * np.sum(np.arange(1, n+1) * sorted_vals) / denom - (n+1)/n)

            top_k = max(1, n_shards // 10)
            top_10_load = float(np.sort(mean_act)[::-1][:top_k].sum())

            summary[layer_name] = {
                "n_tokens": n_tokens,
                "n_shards": n_shards,
                "mean_activation": mean_act.tolist(),
                "activation_std": std_act.tolist(),
                "activation_variance_coeff": activation_variance.tolist(),
                "dead_shards": int(never_active.sum()),
                "sacred_shards": int(always_active.sum()),
                "mean_active_per_token": float(per_token_active.mean()),
                "gini": gini,
                "top_10_pct_load": top_10_load,
                "per_token_active_shards": per_token_active.tolist(),
            }
        return summary


def run_brain_scan(model_path, n_shards, device, max_tokens, output_path):
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
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    recorder = ShardActivationRecorder(model, n_shards=n_shards)
    prompts = build_calibration_prompts()

    results = {
        "model": model_path,
        "n_shards": n_shards,
        "device": device,
        "prompt_types": {},
    }

    for prompt_type, text in prompts.items():
        print(f"\n--- Scanning: {prompt_type} ({len(text)} chars) ---")
        recorder.clear()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        n_tokens = inputs["input_ids"].shape[1]
        print(f"  Tokens: {n_tokens}")

        # Single forward pass — hooks capture all token positions at once
        t0 = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        elapsed = time.time() - t0
        print(f"  Forward pass: {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)")

        summary = recorder.get_summary()
        results["prompt_types"][prompt_type] = summary

        for layer_name, data in sorted(summary.items()):
            if 'down_proj' in layer_name or 'q_proj' in layer_name:
                dead = data["dead_shards"]
                sacred = data["sacred_shards"]
                gini = data["gini"]
                mean_active = data["mean_active_per_token"]
                if dead > 0 or sacred > 0 or gini > 0.05:
                    print(f"  {layer_name:50s} dead={dead:3d} sacred={sacred:3d} "
                          f"gini={gini:.3f} active/tok={mean_active:.0f}/{n_shards}")

    recorder.remove_hooks()

    # Cross-prompt comparison
    print(f"\n{'='*80}")
    print("CROSS-PROMPT ANALYSIS: Input-dependent shard activation")
    print(f"{'='*80}\n")

    all_layers = set()
    for ptype_data in results["prompt_types"].values():
        all_layers.update(ptype_data.keys())

    cross_prompt = {}
    for layer in sorted(all_layers):
        profiles = {}
        for ptype, ptype_data in results["prompt_types"].items():
            if layer in ptype_data:
                profiles[ptype] = np.array(ptype_data[layer]["mean_activation"])

        if len(profiles) < 2:
            continue

        profile_matrix = np.array(list(profiles.values()))
        cross_variance = profile_matrix.var(axis=0)
        mean_cross_var = float(cross_variance.mean())

        threshold_high = cross_variance.mean() * 3
        threshold_low = cross_variance.mean() * 0.1

        prompt_dependent = int(np.sum(cross_variance > threshold_high))
        prompt_independent = int(np.sum(cross_variance < threshold_low))

        cross_prompt[layer] = {
            "mean_cross_variance": mean_cross_var,
            "prompt_dependent_shards": prompt_dependent,
            "prompt_independent_shards": prompt_independent,
        }

        if prompt_dependent > 0 and ('down_proj' in layer or 'q_proj' in layer):
            print(f"{layer:50s} {prompt_dependent:4d} prompt-dependent, "
                  f"{prompt_independent:4d} prompt-independent")

    results["cross_prompt_analysis"] = cross_prompt

    # Verdict
    print(f"\n--- VERDICT ---")
    total_dependent = sum(v["prompt_dependent_shards"] for v in cross_prompt.values())
    total_independent = sum(v["prompt_independent_shards"] for v in cross_prompt.values())
    total_shards_all = len(cross_prompt) * n_shards if cross_prompt else 1

    print(f"Total shard-layer combinations: {total_shards_all}")
    print(f"Prompt-dependent: {total_dependent} ({total_dependent/total_shards_all:.1%})")
    print(f"Prompt-independent: {total_independent} ({total_independent/total_shards_all:.1%})")

    if total_dependent > total_shards_all * 0.05:
        print("\nVERDICT: YES - significant input-dependent shard sparsity exists!")
        print("The cerebellum router concept is viable.")
    else:
        print("\nVERDICT: Shards appear too uniform for selective activation.")
        print("May need SwiGLU gate analysis or architectural changes.")

    if output_path:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                return super().default(obj)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved to {output_path}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Brain Scan - real activation recording")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--n-shards", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    run_brain_scan(args.model, args.n_shards, args.device, args.max_tokens, args.output)


if __name__ == "__main__":
    main()
