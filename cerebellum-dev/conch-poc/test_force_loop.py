"""
Force-loop test: Run a range of layers multiple times with ZERO training.
Uses model.forward() with a hook that captures/replays hidden states.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from contextlib import contextmanager


@contextmanager
def force_loop_context(model, loop_start, loop_end, num_loops):
    """Context manager that forces layers [loop_start:loop_end] to run num_loops times.

    Works by hooking into the model's layer execution. After the loop range
    completes, we feed the output back through those same layers again.
    """
    if num_loops <= 1:
        yield
        return

    hooks = []
    replaying = [False]

    def post_hook_replay(module, args, kwargs, output):
        """After last loop layer, feed output back through the range."""
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
                output = (hidden,) + output[1:]
            else:
                output = hidden
            return output
        finally:
            replaying[0] = False

    # Hook on the last layer of the loop range
    last_loop_layer = model.model.layers[loop_end - 1]
    h = last_loop_layer.register_forward_hook(post_hook_replay, with_kwargs=True)
    hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def compute_baseline_ppl(model, tokenizer, text_path, block_size=512, max_chunks=128):
    """Normal forward pass PPL."""
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


def compute_looped_ppl(model, tokenizer, text_path, loop_start, loop_end, num_loops, block_size=512, max_chunks=128):
    """Forward pass with forced layer loops via hooks."""
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
    model_name = "HuggingFaceTB/SmolLM-135M"
    text_path = "/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    print("=" * 70)

    # Baseline
    print("\n[Baseline] Normal forward pass...")
    ppl_base = compute_baseline_ppl(model, tokenizer, text_path)
    print(f"  PPL = {ppl_base:.4f}")

    # Verify hooks don't change result with 1 loop
    print("\n[Verify] Hook with 1 loop (should match baseline)...")
    ppl_verify = compute_looped_ppl(model, tokenizer, text_path, 10, 20, 1)
    print(f"  PPL = {ppl_verify:.4f}")

    # Test configs
    configs = [
        (10, 20, 2, "layers 10-20 x2"),
        (10, 20, 3, "layers 10-20 x3"),
        (5, 25, 2, "layers 5-25 x2"),
        (12, 18, 2, "layers 12-18 x2 (narrow)"),
        (12, 18, 3, "layers 12-18 x3 (narrow)"),
        (12, 18, 4, "layers 12-18 x4 (narrow)"),
        (20, 28, 2, "layers 20-28 x2 (late)"),
        (25, 30, 2, "layers 25-30 x2 (final)"),
        (25, 30, 3, "layers 25-30 x3 (final)"),
    ]

    results = []
    for start, end, loops, desc in configs:
        print(f"\n[{desc}]...")
        t0 = time.time()
        ppl = compute_looped_ppl(model, tokenizer, text_path, start, end, loops)
        elapsed = time.time() - t0
        delta_pct = 100 * (ppl - ppl_base) / ppl_base
        results.append((desc, ppl, delta_pct, elapsed))
        marker = "BETTER" if delta_pct < 0 else "WORSE"
        print(f"  PPL = {ppl:.4f} ({delta_pct:+.1f}% vs baseline) [{elapsed:.1f}s] {marker}")

    print(f"\n{'='*70}")
    print(f"SUMMARY (baseline PPL = {ppl_base:.4f})")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'PPL':>10} {'Delta':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10}")
    for desc, ppl, delta, _ in sorted(results, key=lambda x: x[1]):
        print(f"{desc:<35} {ppl:>10.4f} {delta:>+9.1f}%")


if __name__ == "__main__":
    main()
