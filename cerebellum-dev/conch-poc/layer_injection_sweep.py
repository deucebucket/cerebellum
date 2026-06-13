"""
layer_injection_sweep.py -- Causal sweep: inject the knowing-delta at each
depth, measure fact recovery AND collateral interference per layer.

For each sampled symbol:
  1. Extract per-layer knowing/ignorant deltas (last token, doc-in-context vs
     space-padded context) -- same protocol as layer_divergence_pilot.py.
  2. For each test layer L: register a forward hook on layers[L] adding
     delta_L to every position, then greedy-generate from the BARE question
     (no context). Score content overlap vs the doc.
  3. With the same injection active, generate from a fixed neutral code
     prompt; measure drift vs the un-injected base generation and degeneracy.

Output: per-layer (recall_overlap, code_drift, degeneracy) -- the
recovery-vs-interference tradeoff curve. Diagnostic, PyTorch-side.
"""
import datetime
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_live_block import _is_degenerate

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"
N_SYMBOLS = 16
SEED = 42
SWEEP_LAYERS = [1, 3, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
GEN_TOKENS = 48
CODE_PROMPT = 'def merge_sorted(a, b):\n    """Merge two sorted lists into one sorted list."""\n'
STATUS_FILE = "DEADBLOCK_STATUS.md"

STOP = set("the a an of to in is and or for with that this it as be are on by".split())


def log_status(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n[{ts}] [LayerSweep] {msg}"
    print(line, flush=True)
    with open(STATUS_FILE, "a") as fh:
        fh.write(line + "\n")


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
            if w not in STOP and len(w) > 2}


def generate(model, tok, prompt: str, n_tokens: int) -> list:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = ids
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(out, use_cache=False).logits
            nxt = logits[0, -1].argmax().view(1, 1)
            out = torch.cat([out, nxt], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
    return out[0, ids.shape[1]:].tolist()


def main():
    docs = torch.load("python_13k_docs.pt", map_location="cpu", weights_only=True)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(docs), generator=g)[:N_SYMBOLS].tolist()
    sample = [docs[i] for i in idx]

    log_status(f"injection sweep start: {N_SYMBOLS} symbols, layers {SWEEP_LAYERS}, "
               f"{GEN_TOKENS} gen tokens, scale 1.0")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.eval()
    layers = model.model.layers

    # Base (un-injected) generations for reference
    base_code_gen = generate(model, tok, CODE_PROMPT, 32)

    # 1. Extract per-layer deltas per symbol
    deltas = []   # [symbol][layer] -> tensor(hidden)
    questions = []
    contents = []
    for doc in sample:
        symbol = doc.split("\n", 1)[0].strip()
        body = doc.split("\n", 1)[1] if "\n" in doc else ""
        question = f"Q: What is {symbol}?\nA:"
        questions.append(question)
        contents.append(content_words(body))
        ctx_len = len(tok(doc).input_ids)
        pad = tok.decode([tok(" ").input_ids[0]] * ctx_len)
        hs = []
        for prompt in (doc + "\n\n" + question, pad + "\n\n" + question):
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True, use_cache=False)
            hs.append(torch.stack([h[0, -1] for h in out.hidden_states]))
        deltas.append(hs[0] - hs[1])   # [n_layers+1, hidden]; index l+1 = after layer l
    log_status("delta extraction done")

    # 2/3. Sweep
    inj = {"delta": None}

    def hook(_mod, _args, output):
        if inj["delta"] is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + inj["delta"].to(output[0].dtype),) + output[1:]
        return output + inj["delta"].to(output.dtype)

    for L in SWEEP_LAYERS:
        handle = layers[L].register_forward_hook(hook)
        overlaps, drifts, degens = [], [], []
        for s in range(N_SYMBOLS):
            inj["delta"] = deltas[s][L + 1]   # delta AFTER layer L
            gen = generate(model, tok, questions[s], GEN_TOKENS)
            text = tok.decode(gen, skip_special_tokens=True)
            cw = contents[s]
            overlaps.append(len(content_words(text) & cw) / max(len(cw), 1))
            code_gen = generate(model, tok, CODE_PROMPT, 32)
            n = min(len(code_gen), len(base_code_gen))
            drifts.append(sum(1 for i in range(n) if code_gen[i] != base_code_gen[i]) / max(n, 1))
            degens.append(_is_degenerate(code_gen))
        inj["delta"] = None
        handle.remove()
        log_status(f"L{L:02d}: recall_overlap={sum(overlaps)/len(overlaps):.3f} "
                   f"code_drift={sum(drifts)/len(drifts):.3f} "
                   f"code_degenerate={sum(degens)}/{N_SYMBOLS}")
    log_status("injection sweep complete")


if __name__ == "__main__":
    main()
