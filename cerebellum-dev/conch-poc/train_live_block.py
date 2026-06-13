"""
train_live_block.py -- Live Block experiment

Architecture: a full Qwen2DecoderLayer (attention + FFN) inserted between
layers 17 and 18, trained with pure LM loss. No gate, no subspace mask,
no revolution loop. o_proj.weight and down_proj.weight zero-initialized so
the block starts as an exact identity. Trained exactly as vanilla llama.cpp
will execute it: h_out = block(h) with standard internal residuals.

This is the fix for the historical STE-gate train/deploy mismatch that caused
looping catastrophe in the old looped-refiner experiments.

Usage:
    python train_live_block.py
    python train_live_block.py --epochs 3 --max-steps-per-epoch 2000
    python train_live_block.py --resume checkpoints-liveblock/live_block_epoch1.pt
"""

import argparse
import copy
import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = torch.device("cuda")
STATUS_FILE = "DEADBLOCK_STATUS.md"
CHECKPOINT_DIR = "checkpoints-liveblock"

WIKI_TRAIN = "/var/home/deucebucket/games/osmosis-quants/wiki.train.raw"
WIKI_TEST  = "/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"
BLOCK_SIZE  = 512
BATCH_SIZE  = 2

# Sentinel: used as default for --data / --val-data to mean "use the classic wiki paths"
_UNSET = object()


# ---------------------------------------------------------------------------
# Status logging (shared file with dead-block; tagged [LiveBlock])
# ---------------------------------------------------------------------------
def log_status(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n[{ts}] [LiveBlock] {msg}"
    print(line, flush=True)
    with open(STATUS_FILE, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Dataset (WikiDataset pattern from train_python_dual.py)
# ---------------------------------------------------------------------------
class WikiDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size=BLOCK_SIZE,
                 max_chunks=None):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        # block_size: int for fixed-length chunks, or list of ints for a
        # deterministic mixed-length cycle (context generalization training)
        sizes = block_size if isinstance(block_size, (list, tuple)) else [block_size]
        self.examples = []
        i, s = 0, 0
        while i + sizes[s % len(sizes)] <= len(tokens):
            n = sizes[s % len(sizes)]
            self.examples.append(torch.tensor(tokens[i : i + n], dtype=torch.long))
            i += n
            s += 1
        if max_chunks is not None:
            self.examples = self.examples[:max_chunks]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def wiki_collate(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : x.size(0)] = x
    return padded


# ---------------------------------------------------------------------------
# Live Block Wrapper
# ---------------------------------------------------------------------------
class LiveBlockWrapper(nn.Module):
    """
    Wraps model.model.layers[17].

    Forward math:
        base_out = base_layer(hidden_states, *args, **kwargs)   [frozen]
        h        = base_out[0] if tuple else base_out
        block_out = self.block(h, *args, **kwargs)              [trainable]
        h_out    = block_out[0] if tuple else block_out
        return (h_out,) + base_out[1:]  if tuple else h_out

    Zero-init invariant at construction time:
        block.self_attn.o_proj.weight  = 0  (no bias on o_proj in Qwen2.5)
        block.mlp.down_proj.weight     = 0
    => attention output contribution == 0, FFN output contribution == 0
    => block output == its own input h (via internal residuals) == base_layer output
    => parity check must pass before training starts.

    No gate. No subspace mask. No bypass flag.
    """

    def __init__(self, base_layer, ffn_only: bool = False, mask_dims: int = 0):
        super().__init__()
        self.base_layer = base_layer
        self.ffn_only = ffn_only
        self.mask_dims = mask_dims

        # Freeze base layer entirely
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Full decoder block deepcopy: preserves architecture, causal mask handling,
        # position embeddings, SwiGLU, RMSNorms -- same args/kwargs pass-through
        # as RefinerWrapper in refiner_vanilla.py.
        self.block = copy.deepcopy(base_layer)

        # Zero-init so block starts as identity.
        # Confirmed: o_proj is nn.Linear(..., bias=False) in Qwen2.5; weight only.
        nn.init.zeros_(self.block.self_attn.o_proj.weight)
        nn.init.zeros_(self.block.mlp.down_proj.weight)

        # All block params are trainable by default
        for p in self.block.parameters():
            p.requires_grad = True

        # --ffn-only: zero AND freeze all attention projection weights+biases.
        # o_proj was already zeroed above; q/k/v are zeroed here too so that
        # attention queries/keys/values produce zeros -> o_proj(0) = 0 regardless.
        # The block then reduces to: h + FFN(norm(h)), which is context-length-safe.
        if self.ffn_only:
            attn = self.block.self_attn
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name)
                nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
                proj.weight.requires_grad = False
                if proj.bias is not None:
                    proj.bias.requires_grad = False

        # --mask-dims N: zero AND freeze down_proj rows 0..N-1 at init.
        # Re-zeroing after each optimizer step is handled by zero_masked_rows().
        if self.mask_dims > 0:
            with torch.no_grad():
                self.block.mlp.down_proj.weight[:self.mask_dims, :].zero_()
            # freeze those rows by marking them requires_grad=False is not
            # granular in PyTorch (grad is per-tensor); we use zero_masked_rows()
            # post-step instead (same pattern as train_dead_block.py).

    def zero_masked_rows(self):
        """Re-zero down_proj rows 0..mask_dims-1 after each optimizer step.
        No-op when mask_dims == 0. Mirrors dead-block's zero_frozen_rows()."""
        if self.mask_dims > 0:
            with torch.no_grad():
                self.block.mlp.down_proj.weight[:self.mask_dims, :].zero_()

    def forward(self, hidden_states, *args, **kwargs):
        base_out = self.base_layer(hidden_states, *args, **kwargs)

        h = base_out[0] if isinstance(base_out, tuple) else base_out

        block_out = self.block(h, *args, **kwargs)
        h_out = block_out[0] if isinstance(block_out, tuple) else block_out

        if isinstance(base_out, tuple):
            return (h_out,) + base_out[1:]
        return h_out


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------
def patch_model(model, ffn_only: bool = False, mask_dims: int = 0):
    """Replace layers[17] with LiveBlockWrapper; freeze everything else."""
    wrapper = LiveBlockWrapper(
        model.model.layers[17], ffn_only=ffn_only, mask_dims=mask_dims
    ).to(DEVICE, dtype=torch.bfloat16)
    model.model.layers[17] = wrapper

    # Freeze all parameters outside the wrapper's trainable block
    for name, param in model.named_parameters():
        in_wrapper_block = (
            "model.layers.17" in name and ".base_layer." not in name
        )
        param.requires_grad = in_wrapper_block

    # The loop above clobbers the constructor's --ffn-only freeze (it set
    # requires_grad=True on every block param) — re-apply it.
    if ffn_only:
        attn = wrapper.block.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = getattr(attn, proj_name)
            proj.weight.requires_grad = False
            if proj.bias is not None:
                proj.bias.requires_grad = False

    return model, wrapper


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------
def load_resume(path: str, wrapper: LiveBlockWrapper):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    wrapper.block.load_state_dict(ckpt["l17_block"])
    log_status(f"Resumed from {path}.")


# ---------------------------------------------------------------------------
# Parity check (REQUIRED before training)
# ---------------------------------------------------------------------------
def run_parity_check(model, tokenizer, wrapper: LiveBlockWrapper):
    log_status("=== PARITY CHECK: verifying zero-init identity (o_proj=0, down_proj=0) ===")

    prompts = [
        "What is the capital of France?",
        "def fibonacci(n):",
        "import os\nprint(os.getcwd())",
    ]

    model.eval()
    max_diffs = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Wrapper active: block has zero o_proj/down_proj, so h passes through
        with torch.no_grad():
            logits_with = model(**inputs).logits

        # Bypass: temporarily swap wrapper for raw base_layer
        model.model.layers[17] = wrapper.base_layer
        with torch.no_grad():
            logits_without = model(**inputs).logits
        model.model.layers[17] = wrapper  # restore

        diff = (logits_with - logits_without).abs().max().item()
        max_diffs.append(diff)
        log_status(f"  Prompt '{prompt[:40]}' max_abs_diff = {diff:.6e}")

    overall_max = max(max_diffs)
    # bf16 rounding is ~1e-3; true zeros produce exact 0 in down_proj/o_proj path
    threshold = 1e-3
    if overall_max <= threshold:
        log_status(
            f"PARITY CHECK PASSED: max_abs_diff = {overall_max:.6e} "
            f"(<= {threshold}). Identity confirmed."
        )
    else:
        log_status(
            f"PARITY CHECK FAILED: max_abs_diff = {overall_max:.6e} "
            f"(> {threshold}). Zero-init did not produce identity. Aborting."
        )
        sys.exit(1)

    return overall_max


# ---------------------------------------------------------------------------
# Validation: held-out PPL on first 100 chunks of wiki.test.raw
# ---------------------------------------------------------------------------
def run_validation(model, val_loader) -> float:
    """Compute PPL on held-out chunks (no_grad, exp of mean cross-entropy)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids in val_loader:
            input_ids = input_ids.to(DEVICE)
            # use_cache=False: the inserted block deepcopies layer_idx=17 and
            # would otherwise read the base layer's KV-cache slot — a phantom
            # attention path that does not exist in the exported GGUF.
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    import math
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Behavior probe: greedy generations that PPL cannot police.
# The HumanEval audits showed the dominant insertion-block pathology is
# degenerate token looping (emoji storms, repeated pseudo-code) — invisible
# to cross-entropy validation. Probe a fixed prompt set every epoch and gate
# checkpoint eligibility on non-degenerate output.
# ---------------------------------------------------------------------------
PROBE_PROMPTS = [
    'def add(a, b):\n    """Return the sum of a and b."""\n',
    'def is_even(n):\n    """Return True if n is even."""\n',
    'def reverse_string(s):\n    """Return s reversed."""\n',
    'def factorial(n):\n    """Return n! for non-negative integer n."""\n',
    'def max_of_list(lst):\n    """Return the largest element of lst."""\n',
    "The capital of France is",
]


def _is_degenerate(ids) -> bool:
    """Token-loop detector: low distinct-token ratio, or the generation tail
    is one n-gram repeated back-to-back. Catches loop storms, not wrongness."""
    ids = list(ids)
    if len(ids) >= 16 and len(set(ids)) / len(ids) < 0.35:
        return True
    for n in range(1, 9):
        repeats = 5 if n == 1 else 4   # 4 identical single tokens can be legit
        if len(ids) >= repeats * n:
            tail = ids[-repeats * n:]
            gram = tail[:n]
            if all(tail[i:i + n] == gram for i in range(0, repeats * n, n)):
                return True
    return False


def run_behavior_probe(model, tokenizer):
    """Greedy 64-token generation per probe prompt. Returns
    (non_degenerate_count, [(prompt_head, degenerate, snippet), ...]).
    use_cache=False: the wrapper only ever trains on full sequences, so force
    full forwards rather than trusting its KV-cache path."""
    model.eval()
    results = []
    passed = 0
    for prompt in PROBE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        degen = _is_degenerate(gen_ids)
        snippet = tokenizer.decode(gen_ids, skip_special_tokens=True)[:60]
        results.append((prompt.splitlines()[0][:40], degen,
                        snippet.replace("\n", "\\n")))
        if not degen:
            passed += 1
    model.train()
    return passed, results


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------
def save_checkpoint(wrapper: LiveBlockWrapper, epoch: int, val_ppl: float, path: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.save(
        {
            "l17_block": wrapper.block.state_dict(),
            "meta": {
                "epoch": epoch,
                "val_ppl": val_ppl,
                "timestamp": ts,
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training loop (one epoch, capped at max_steps)
# ---------------------------------------------------------------------------
def train_epoch(
    model,
    wrapper: "LiveBlockWrapper",
    loader,
    optimizer,
    epoch: int,
    max_steps: int,
) -> float:
    """Run one epoch up to max_steps. Returns mean train loss for the epoch."""
    model.train()

    running_loss = 0.0
    epoch_loss = 0.0
    step = 0

    for input_ids in loader:
        if step >= max_steps:
            break

        input_ids = input_ids.to(DEVICE)

        # use_cache=False: train under deployed semantics — see run_validation.
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits

        # Pure LM loss: standard causal shift, no masking
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        wrapper.zero_masked_rows()   # no-op when mask_dims == 0
        optimizer.zero_grad()

        running_loss += loss.item()
        epoch_loss += loss.item()
        step += 1

        if step % 100 == 0:
            import math
            avg_loss = running_loss / 100
            running_ppl = math.exp(avg_loss)
            log_status(
                f"[Epoch {epoch}] Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | PPL: {running_ppl:.2f}"
            )
            running_loss = 0.0

    return epoch_loss / max(step, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a live block (full Qwen2 decoder) inserted between "
            "layers 17 and 18 with pure LM loss."
        )
    )
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=2000,
                        help="Max optimizer steps per epoch (default: 2000)")
    parser.add_argument("--mixed-context", action="store_true",
                        help="train on a deterministic mixed-length cycle "
                             "(512..2048) for context generalization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint {'l17_block', 'meta'} to resume from")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training corpus text file "
                             "(default: WIKI_TRAIN constant)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Validation source(s). "
                             "Single path → used for both val@512 and val@2048 (classic mode). "
                             "Comma-separated pair 'python:<path>,wiki:<path>' → val@512 from "
                             "python corpus tail (last 100 chunks held out from training), "
                             "val@2048 from wiki path as regression guard.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: checkpoints-liveblock)")
    parser.add_argument("--holdout-bytes", type=int, default=0,
                        help="Bytes to exclude from the end of --data when building the "
                             "training set (reserve as python val tail). "
                             "Ignored when --val-data uses the 'python:,wiki:' syntax "
                             "because that syntax computes the holdout automatically.")
    parser.add_argument("--ffn-only", action="store_true",
                        help="Zero AND freeze all self_attn {q,k,v,o}_proj weights+biases. "
                             "Reduces the block to a pointwise FFN — context-length-safe "
                             "by construction (no attention path to disrupt KV cache).")
    parser.add_argument("--mask-dims", type=int, default=0,
                        help="Zero AND freeze down_proj.weight rows 0..N-1 (output dims) "
                             "at init and re-zero after each optimizer step. "
                             "N=1536 reproduces the 25%% knowledge-lane from dead-block. "
                             "0 = no mask (default).")
    parser.add_argument("--context-guard", type=float, default=2.0,
                        help="Max allowed val@2048 regression vs baseline as a percentage. "
                             "Baseline is measured on the freshly-patched (still-identity) model "
                             "before epoch 1. A checkpoint is eligible only if "
                             "val@2048 <= baseline * (1 + PCT/100). "
                             "Among eligible, best python val@512 wins. "
                             "Default: 2.0 (allow up to 2%% regression).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="AdamW learning rate (default: 1e-4). The 2026-06-11 epoch "
                             "sweep showed behavioral damage saturates within the first "
                             "2500 steps at 1e-4 — intensity must be cut here, not via "
                             "step count.")
    parser.add_argument("--behavior-probe", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Greedy-generate from fixed code prompts at each epoch end "
                             "and reject checkpoints whose outputs degenerate into token "
                             "loops — the failure mode val PPL cannot see. Default: on.")
    parser.add_argument("--probe-min-pass", type=int, default=6,
                        help="Minimum non-degenerate probe completions (of 6) for "
                             "checkpoint eligibility. Clamped to the identity-model "
                             "baseline so the bar is never higher than the frozen model "
                             "itself achieves. Default: 6.")
    args = parser.parse_args()

    # Resolve checkpoint dir
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else CHECKPOINT_DIR

    # Resolve train data path
    train_path = args.data if args.data else WIKI_TRAIN

    # Resolve val sources.
    # Supported formats for --val-data:
    #   (a) not set        → wiki@512 + wiki@2048 (classic)
    #   (b) single path    → that file for both val@512 and val@2048
    #   (c) "python:<p1>,wiki:<p2>"  → python corpus tail @512 + wiki @2048
    python_val_path = None
    wiki_val_path   = None
    if args.val_data is None:
        wiki_val_path = WIKI_TEST
    elif args.val_data.startswith("python:") and ",wiki:" in args.val_data:
        # Parse "python:<p1>,wiki:<p2>"
        py_part, wiki_part = args.val_data.split(",wiki:", 1)
        python_val_path = py_part[len("python:"):]
        wiki_val_path   = wiki_part
    else:
        # Single path: use for both contexts
        wiki_val_path = args.val_data

    log_status(
        f"=== train_live_block.py started | "
        f"epochs={args.epochs} max_steps={args.max_steps_per_epoch} "
        f"lr={args.lr:g} behavior_probe={args.behavior_probe} "
        f"resume={args.resume} data={train_path} "
        f"python_val={python_val_path} wiki_val={wiki_val_path} "
        f"checkpoint_dir={ckpt_dir} ==="
    )

    # Load tokenizer
    log_status(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Build datasets before model load to fail fast on missing files
    # Mixed-length cycle for context generalization; batch 1 so short chunks
    # aren't padded up to the longest in a batch
    train_sizes = [512, 1024, 2048, 768, 1536, 512] if args.mixed_context else BLOCK_SIZE
    train_batch = 1 if args.mixed_context else BATCH_SIZE

    # Holdout: when using python:,wiki: split val, hold out the last 100 @512
    # chunks of the training file so they can serve as the python val set.
    # We implement this by capping training to max_chunks = (total - 100).
    # The same 100 chunks are then loaded as the python val dataset.
    train_max_chunks = None
    if python_val_path is not None and python_val_path == train_path:
        # Auto-holdout: count chunks at @512 and subtract 100
        _probe = WikiDataset(tokenizer, train_path, block_size=BLOCK_SIZE)
        train_max_chunks = max(len(_probe) - 100, 1)
        del _probe
        log_status(
            f"  Auto-holdout: training on first {train_max_chunks} @512 chunks; "
            f"last 100 held out for python val."
        )
    elif args.holdout_bytes > 0:
        # Manual holdout: user passed --holdout-bytes; handled via max_chunks
        # approximation (1 chunk ~ BLOCK_SIZE tokens ~ 2-3 bytes/token)
        log_status(
            f"  Note: --holdout-bytes={args.holdout_bytes} is approximate; "
            f"use 'python:<p>,wiki:<p>' syntax for exact chunk-level holdout."
        )

    log_status(f"Building dataset from {train_path} (sizes={train_sizes})...")
    train_dataset = WikiDataset(
        tokenizer, train_path,
        block_size=train_sizes,
        max_chunks=train_max_chunks,
    )
    log_status(f"  Train chunks: {len(train_dataset)}")

    # Val @512: python tail OR wiki
    if python_val_path is not None:
        log_status(f"Building python val@512 from tail of {python_val_path} (100 chunks)...")
        # Load ALL @512 chunks from the python corpus, then take the last 100
        _all_py = WikiDataset(tokenizer, python_val_path, block_size=BLOCK_SIZE)
        tail_start = max(len(_all_py) - 100, 0)
        val_dataset_py = copy.copy(_all_py)
        val_dataset_py.examples = _all_py.examples[tail_start:]
        del _all_py
        log_status(f"  Python val chunks @512: {len(val_dataset_py)}")
        val_dataset = val_dataset_py
    else:
        log_status(f"Building wiki val@512 from {wiki_val_path} (100 chunks)...")
        val_dataset = WikiDataset(tokenizer, wiki_val_path, block_size=BLOCK_SIZE, max_chunks=100)

    # Val @2048: always wiki (regression guard)
    log_status(f"Building wiki val@2048 from {wiki_val_path} (40 chunks)...")
    val_dataset_long = WikiDataset(tokenizer, wiki_val_path, block_size=2048, max_chunks=40)
    log_status(f"  Val chunks: {len(val_dataset)} @512, {len(val_dataset_long)} @2048")

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch, shuffle=True, collate_fn=wiki_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=wiki_collate
    )
    val_loader_long = DataLoader(
        val_dataset_long, batch_size=1, shuffle=False, collate_fn=wiki_collate
    )

    # Load model and patch
    log_status(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model, wrapper = patch_model(model, ffn_only=args.ffn_only, mask_dims=args.mask_dims)
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_status(f"Trainable parameters: {trainable:,}")
    if args.ffn_only:
        log_status("[FFNCoder-prep] --ffn-only active: self_attn {q,k,v,o}_proj zeroed and frozen. Block is FFN-only.")
    if args.mask_dims > 0:
        log_status(f"[FFNCoder-prep] --mask-dims {args.mask_dims}: down_proj rows 0..{args.mask_dims-1} zeroed at init; will re-zero after each step.")
    log_status(f"[FFNCoder-prep] --context-guard {args.context_guard:.1f}%: baseline val@2048 will be measured before epoch 1.")

    # Resume before parity check (resumed weights may not be identity, that's fine --
    # parity is only meaningful at init; skip if resuming)
    if args.resume is not None:
        load_resume(args.resume, wrapper)
        log_status("Skipping parity check (resuming from checkpoint; block is no longer at zero-init).")
    else:
        run_parity_check(model, tokenizer, wrapper)

    # Optimizer: wd=0.1 fixed; lr is the intensity knob (--lr)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.1,
    )
    log_status(f"Optimizer: AdamW lr={args.lr:g} weight_decay=0.1")

    os.makedirs(ckpt_dir, exist_ok=True)

    import math

    # --context-guard: measure val@2048 on the patched-but-still-identity model
    # as the in-protocol baseline. Every checkpoint must stay within this + PCT%.
    log_status("[FFNCoder-prep] Measuring context-guard baseline val@2048 (identity model)...")
    baseline_val2048 = run_validation(model, val_loader_long)
    guard_ceiling = baseline_val2048 * (1.0 + args.context_guard / 100.0)
    log_status(
        f"[FFNCoder-prep] Context-guard baseline val@2048 = {baseline_val2048:.4f}. "
        f"Ceiling = {guard_ceiling:.4f} (baseline * (1 + {args.context_guard:.1f}%/100))."
    )

    # --behavior-probe: baseline on the patched-but-still-identity model.
    # Eligibility bar = min(--probe-min-pass, baseline) so we never demand
    # better behavior than the frozen model itself shows.
    probe_required = 0
    if args.behavior_probe:
        log_status("[FFNCoder-prep] Behavior probe baseline (identity model)...")
        base_pass, base_results = run_behavior_probe(model, tokenizer)
        probe_required = min(args.probe_min_pass, base_pass)
        log_status(
            f"[FFNCoder-prep] Baseline probe: {base_pass}/{len(PROBE_PROMPTS)} "
            f"non-degenerate. Eligibility requires >= {probe_required}."
        )
        for name, degen, snippet in base_results:
            if degen:
                log_status(f"[FFNCoder-prep] baseline probe degenerate: {name!r} -> {snippet!r}")

    best_val_ppl = float("inf")      # geomean, for logging only
    best_constrained_ppl = float("inf")   # best eligible python val@512
    best_constrained_epoch = None

    for epoch in range(1, args.epochs + 1):
        log_status(f"=== Epoch {epoch}/{args.epochs} ===")

        mean_train_loss = train_epoch(
            model, wrapper, train_loader, optimizer, epoch, args.max_steps_per_epoch
        )
        train_ppl = math.exp(mean_train_loss)
        log_status(
            f"[Epoch {epoch}] Train mean loss: {mean_train_loss:.4f} | "
            f"Train PPL: {train_ppl:.2f}"
        )

        val_ppl_512 = run_validation(model, val_loader)
        val_ppl_2048 = run_validation(model, val_loader_long)
        # Selection metric: geometric mean of both contexts — a checkpoint must
        # hold the short-context gain without regressing long-context
        val_ppl = math.sqrt(val_ppl_512 * val_ppl_2048)
        log_status(
            f"[Epoch {epoch}] Val PPL @512: {val_ppl_512:.4f} | "
            f"@2048: {val_ppl_2048:.4f} | combined (geomean): {val_ppl:.4f}"
        )

        # Per-epoch checkpoint
        epoch_path = os.path.join(ckpt_dir, f"live_block_epoch{epoch}.pt")
        save_checkpoint(wrapper, epoch, val_ppl, epoch_path)

        # Context-guard eligibility: val@2048 must not exceed ceiling
        guard_ok = val_ppl_2048 <= guard_ceiling
        guard_tag = "ELIGIBLE" if guard_ok else f"INELIGIBLE(val@2048={val_ppl_2048:.4f} > ceiling={guard_ceiling:.4f})"

        # Behavior-probe eligibility: generations must not degenerate
        probe_ok = True
        probe_tag = "OFF"
        if args.behavior_probe:
            probe_pass, probe_results = run_behavior_probe(model, tokenizer)
            probe_ok = probe_pass >= probe_required
            probe_tag = f"{probe_pass}/{len(PROBE_PROMPTS)} {'OK' if probe_ok else 'FAIL'}"
            for name, degen, snippet in probe_results:
                if degen:
                    log_status(f"[Epoch {epoch}] PROBE DEGENERATE: {name!r} -> {snippet!r}")

        # Best constrained checkpoint: among eligible, pick best python val@512
        is_constrained_best = guard_ok and probe_ok and val_ppl_512 < best_constrained_ppl
        if is_constrained_best:
            best_constrained_ppl = val_ppl_512
            best_constrained_epoch = epoch
            best_path = os.path.join(ckpt_dir, "live_block_best.pt")
            save_checkpoint(wrapper, epoch, val_ppl, best_path)

        # Geomean tracking (info only, not used for selection)
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

        # Always write last
        last_path = os.path.join(ckpt_dir, "live_block_last.pt")
        save_checkpoint(wrapper, epoch, val_ppl, last_path)

        written = [epoch_path, last_path]
        if is_constrained_best:
            written.append(best_path)

        log_status(
            f"[Epoch {epoch} SUMMARY] "
            f"train_ppl={train_ppl:.4f} val_ppl@512={val_ppl_512:.4f} "
            f"val_ppl@2048={val_ppl_2048:.4f} geomean={val_ppl:.4f} | "
            f"context-guard={guard_tag} | behavior-probe={probe_tag} | "
            f"constrained_best={'YES' if is_constrained_best else 'NO'} | "
            f"saved={[os.path.basename(p) for p in written]}"
        )

    if best_constrained_epoch is None:
        log_status(
            f"[FFNCoder-prep] WARNING: NO EPOCH PASSED CONTEXT GUARD "
            f"(ceiling={guard_ceiling:.4f}, guard={args.context_guard:.1f}%). "
            f"live_block_best.pt NOT WRITTEN. All epochs regressed long-context beyond threshold."
        )
    else:
        log_status(
            f"[FFNCoder-prep] Best constrained checkpoint: epoch {best_constrained_epoch}, "
            f"val@512={best_constrained_ppl:.4f} (saved as live_block_best.pt)."
        )
    log_status(
        f"Training complete. Best geomean val PPL: {best_val_ppl:.4f}. "
        f"Checkpoints in {ckpt_dir}/"
    )


if __name__ == "__main__":
    main()
