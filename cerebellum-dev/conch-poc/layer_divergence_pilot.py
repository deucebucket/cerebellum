"""
layer_divergence_pilot.py -- Per-layer knowing/ignorant divergence curve.

For a sample of stdlib symbols: run a "knowing" pass (doc in context) and an
"ignorant" pass (context replaced by space padding of equal token length),
capture last-token hidden states at EVERY layer, and report the divergence
(L2 norm and cosine distance) per layer.

Purpose: find the DETECT address (earliest layer where knowing/ignorant
separate -- where the "huh?" signal lives) and characterize where knowledge
deltas are largest, to aim the layer-injection sweep that follows.

Diagnostic, PyTorch-side. Not a benchmark.
"""
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
N_SYMBOLS = 24
SEED = 42
STATUS_FILE = "DEADBLOCK_STATUS.md"


def log_status(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n[{ts}] [LayerSweep] {msg}"
    print(line, flush=True)
    with open(STATUS_FILE, "a") as fh:
        fh.write(line + "\n")


def main():
    docs = torch.load("python_13k_docs.pt", map_location="cpu", weights_only=True)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(docs), generator=g)[:N_SYMBOLS].tolist()
    sample = [docs[i] for i in idx]

    log_status(f"pilot start: {N_SYMBOLS} symbols, seed {SEED}, model {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    norm_sum = torch.zeros(n_layers + 1)   # +1: embeddings row
    cos_sum = torch.zeros(n_layers + 1)

    for doc in sample:
        symbol = doc.split("\n", 1)[0].strip()
        context = doc
        question = f"\n\nQ: What is {symbol}?\nA:"

        known = context + question
        ctx_len = len(tok(context).input_ids)
        # ignorant: context replaced by space padding of equal token length
        pad = tok.decode([tok(" ").input_ids[0]] * ctx_len)
        ignorant = pad + question

        outs = []
        for prompt in (known, ignorant):
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True, use_cache=False)
            # last-token hidden state at every layer (incl. embeddings at [0])
            outs.append(torch.stack([h[0, -1].float().cpu() for h in out.hidden_states]))
        hk, hi = outs
        delta = hk - hi
        norm_sum += delta.norm(dim=-1)
        cos_sum += 1 - torch.nn.functional.cosine_similarity(hk, hi, dim=-1)

    log_status("per-layer mean divergence (layer: |delta| L2, cosine distance):")
    lines = []
    for layer in range(n_layers + 1):
        tag = "emb" if layer == 0 else f"L{layer - 1:02d}"
        lines.append(
            f"  {tag}: |d|={norm_sum[layer] / N_SYMBOLS:8.3f}  "
            f"cosdist={cos_sum[layer] / N_SYMBOLS:.4f}"
        )
    # log compactly: 4 per line
    for i in range(0, len(lines), 4):
        log_status(" | ".join(s.strip() for s in lines[i:i + 4]))
    log_status("pilot complete")


if __name__ == "__main__":
    main()
