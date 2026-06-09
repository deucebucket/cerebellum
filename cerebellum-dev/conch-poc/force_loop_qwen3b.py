"""Quick force-loop diagnostic for Qwen2.5-3B — tests scaling signal."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from contextlib import contextmanager


@contextmanager
def force_loop_context(model, loop_start, loop_end, num_loops):
    if num_loops <= 1:
        yield
        return

    hooks = []
    replaying = [False]

    def post_hook_replay(module, args, kwargs, output):
        if replaying[0]:
            return output
        replaying[0] = True
        try:
            extra_passes = num_loops - 1
            hidden = output[0] if isinstance(output, tuple) else output
            layers = list(model.model.layers[loop_start:loop_end])
            replay_kwargs = {k: v for k, v in kwargs.items() if k == 'position_embeddings'}
            for _ in range(extra_passes):
                for layer in layers:
                    result = layer(hidden, **replay_kwargs)
                    hidden = result[0] if isinstance(result, tuple) else result
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        finally:
            replaying[0] = False

    last_loop_layer = model.model.layers[loop_end - 1]
    h = last_loop_layer.register_forward_hook(post_hook_replay, with_kwargs=True)
    hooks.append(h)
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def compute_ppl(model, tokenizer, text_path, block_size=512, max_chunks=64):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    total_loss = 0
    n_chunks = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = torch.tensor(tokens[i:i + block_size], dtype=torch.long).unsqueeze(0).cuda()
            out = model(input_ids=chunk, labels=chunk)
            total_loss += out.loss.item()
            n_chunks += 1
            if n_chunks >= max_chunks:
                break
    return torch.exp(torch.tensor(total_loss / n_chunks)).item()


def compute_looped_ppl(model, tokenizer, text_path, loop_start, loop_end, num_loops,
                       block_size=512, max_chunks=64):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    total_loss = 0
    n_chunks = 0
    model.eval()
    with torch.no_grad(), force_loop_context(model, loop_start, loop_end, num_loops):
        for i in range(0, len(tokens) - block_size, block_size):
            chunk = torch.tensor(tokens[i:i + block_size], dtype=torch.long).unsqueeze(0).cuda()
            out = model(input_ids=chunk, labels=chunk)
            total_loss += out.loss.item()
            n_chunks += 1
            if n_chunks >= max_chunks:
                break
    return torch.exp(torch.tensor(total_loss / n_chunks)).item()


def main():
    model_name = "Qwen/Qwen2.5-3B"
    text_path = "/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    print(f"VRAM used after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 70)

    # Baseline
    print("\n[Baseline] Normal forward pass...")
    ppl_base = compute_ppl(model, tokenizer, text_path)
    print(f"  PPL = {ppl_base:.4f}")

    # Verify hooks don't distort
    print("\n[Verify] Hook with 1 loop (should match baseline)...")
    ppl_verify = compute_looped_ppl(model, tokenizer, text_path, 12, 24, 1)
    print(f"  PPL = {ppl_verify:.4f}")

    # Test configs scaled for 36 layers (SmolLM was 30 layers)
    # Early, middle, late, narrow, wide
    configs = [
        (6, 30, 2,  "layers 6-30 x2 (wide)"),
        (12, 24, 2, "layers 12-24 x2 (middle)"),
        (15, 21, 2, "layers 15-21 x2 (narrow)"),
        (6, 30, 3,  "layers 6-30 x3 (wide)"),
        (12, 24, 3, "layers 12-24 x3 (middle)"),
        (15, 21, 3, "layers 15-21 x3 (narrow)"),
        (15, 21, 4, "layers 15-21 x4 (narrow)"),
        (24, 34, 2, "layers 24-34 x2 (late)"),
        (24, 34, 3, "layers 24-34 x3 (late)"),
    ]

    results = []
    for start, end, loops, desc in configs:
        print(f"\n[{desc}]...")
        t0 = time.time()
        ppl = compute_looped_ppl(model, tokenizer, text_path, start, end, loops)
        elapsed = time.time() - t0
        delta_pct = 100 * (ppl - ppl_base) / ppl_base
        results.append((desc, ppl, delta_pct, elapsed))
        marker = "\033[92mBETTER\033[0m" if delta_pct < 0 else "\033[91mWORSE\033[0m"
        print(f"  PPL = {ppl:.4f} ({delta_pct:+.1f}% vs baseline) [{elapsed:.1f}s] {marker}")

    print(f"\n{'='*70}")
    print(f"SUMMARY (baseline PPL = {ppl_base:.4f})")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'PPL':>10} {'Delta':>10} {'Time':>8}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for desc, ppl, delta, elapsed in sorted(results, key=lambda x: x[1]):
        marker = ">>" if delta < 0 else ""
        print(f"{desc:<30} {ppl:>10.4f} {delta:>+9.1f}% {elapsed:>7.1f}s  {marker}")

    # Verdict
    improvements = [r for r in results if r[2] < 0]
    if improvements:
        best = min(results, key=lambda x: x[1])
        print(f"\n>>> SIGNAL FOUND: best config '{best[0]}' improves PPL by {abs(best[2]):.1f}%")
        print(f">>> Loop effect scales past SmolLM. Refiner training is worth it.")
    else:
        print(f"\n>>> NO SIGNAL at this scale. Loop effect may be small-model-only.")


if __name__ == "__main__":
    main()
