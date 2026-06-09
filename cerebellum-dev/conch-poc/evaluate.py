"""
Conch Shell Evaluation — Compare baseline vs conch at different revolution depths.

Tests:
1. Perplexity on held-out text (does looping help?)
2. Adaptive exit behavior (does it exit early on easy text?)
3. Generation quality comparison (baseline 135M vs conch)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from pathlib import Path

from model import load_conch


def compute_perplexity(model, tokenizer, text_path, block_size=512, device="cuda"):
    """Compute perplexity on text file."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    total_loss = 0
    n_chunks = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = torch.tensor(tokens[i : i + block_size], dtype=torch.long).unsqueeze(0).to(device)
            labels = chunk.clone()

            if hasattr(model, "loop_block"):
                outputs = model(input_ids=chunk, labels=labels)
                loss = outputs["loss"]
            else:
                outputs = model(input_ids=chunk, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
            n_chunks += 1

            if n_chunks >= 128:
                break

    avg_loss = total_loss / n_chunks
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl, n_chunks


def test_adaptive_exit(model, tokenizer, prompts, device="cuda"):
    """Test if model exits early on easy prompts and loops more on hard ones."""
    model.eval()
    results = []

    with torch.no_grad():
        for prompt_info in prompts:
            text = prompt_info["text"]
            difficulty = prompt_info["difficulty"]

            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(input_ids=inputs["input_ids"])

            results.append({
                "text": text[:50] + "...",
                "difficulty": difficulty,
                "revolutions": outputs["revolution_count"].item(),
                "exit_confidences": [c.item() for c in outputs["exit_confidences"]],
            })

    return results


def generate_comparison(model, tokenizer, baseline_model, prompts, device="cuda", max_new_tokens=100):
    """Generate from both models and compare."""
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline generation (simple greedy)
        baseline_model.eval()
        with torch.no_grad():
            baseline_ids = baseline_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)

        # Conch generation (greedy with adaptive loops)
        model.eval()
        generated = inputs["input_ids"].clone()
        total_revolutions = 0
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                total_revolutions += outputs["revolution_count"].item()
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        conch_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "baseline": baseline_text,
            "conch": conch_text,
            "avg_revolutions": total_revolutions / max_new_tokens,
        })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--data", type=str, default="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # Load conch model
    print("Loading Conch model...")
    model, tokenizer = load_conch(args.base_model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load baseline for comparison
    print("Loading baseline model...")
    baseline = AutoModelForCausalLM.from_pretrained(args.base_model).to(device)
    baseline.eval()

    # 1. Perplexity comparison
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON")
    print("=" * 60)

    conch_ppl, n = compute_perplexity(model, tokenizer, args.data, device=device)
    baseline_ppl, _ = compute_perplexity(baseline, tokenizer, args.data, device=device)

    print(f"  Baseline SmolLM-135M PPL: {baseline_ppl:.4f}")
    print(f"  Conch (adaptive) PPL:     {conch_ppl:.4f}")
    print(f"  Delta: {conch_ppl - baseline_ppl:+.4f} ({(conch_ppl - baseline_ppl)/baseline_ppl*100:+.2f}%)")

    # 2. Adaptive exit test
    print("\n" + "=" * 60)
    print("ADAPTIVE EXIT BEHAVIOR")
    print("=" * 60)

    test_prompts = [
        {"text": "The cat sat on the", "difficulty": "easy"},
        {"text": "Hello, my name is", "difficulty": "easy"},
        {"text": "The capital of France is", "difficulty": "easy"},
        {"text": "In quantum mechanics, the uncertainty principle states that", "difficulty": "medium"},
        {"text": "The proof of Fermat's last theorem relies on", "difficulty": "hard"},
        {"text": "Consider a recursive function f(n) where f(0)=1 and f(n)=n*f(n-1). Compute f(10):", "difficulty": "hard"},
    ]

    exit_results = test_adaptive_exit(model, tokenizer, test_prompts, device=device)
    for r in exit_results:
        print(f"  [{r['difficulty']:6s}] revs={r['revolutions']:.0f} conf={r['exit_confidences'][-1]:.3f} | {r['text']}")

    # 3. Generation comparison
    print("\n" + "=" * 60)
    print("GENERATION COMPARISON")
    print("=" * 60)

    gen_prompts = [
        "Once upon a time",
        "The best way to learn programming is",
        "Explain why the sky is blue:",
    ]

    gen_results = generate_comparison(model, tokenizer, baseline, gen_prompts, device=device)
    for r in gen_results:
        print(f"\n  Prompt: {r['prompt']}")
        print(f"  Avg revolutions per token: {r['avg_revolutions']:.1f}")
        print(f"  Baseline: {r['baseline'][:200]}")
        print(f"  Conch:    {r['conch'][:200]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    params = model.count_parameters()
    print(f"  Unique params: {params['total_unique_params']:,}")
    print(f"  Effective params (max rev): {params['effective_at_max_rev']:,}")
    print(f"  Depth multiplier: {params['effective_at_max_rev']/params['total_unique_params']:.1f}x")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"  Conch PPL: {conch_ppl:.4f}")

    conch_wins = conch_ppl < baseline_ppl
    print(f"\n  VERDICT: {'CONCH WINS' if conch_wins else 'NEEDS MORE TRAINING'}")
    print(f"  (Conch has {params['effective_at_max_rev']/params['total_unique_params']:.1f}x effective depth from {params['total_unique_params']:,} stored params)")


if __name__ == "__main__":
    main()
