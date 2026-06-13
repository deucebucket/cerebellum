"""
train_dead_block.py -- Dead Block experiment (EXPERIMENTAL_PATHS.md paths 1+2)

Architecture: attention-free, subspace-masked, zero-init FFN-only refiner that
is EXACTLY representable as a standard Qwen2 decoder block.

Usage:
    python train_dead_block.py --stage a    # parity check + stage A (first 2000 samples)
    python train_dead_block.py --stage b    # full 13k stage B (launched by stage A if gate passes)
    python train_dead_block.py --stage b --epochs 3 --resume checkpoints-deadblock-13k/dead_blocks_epoch1.pt
"""

import argparse
import os
import sys
import copy
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = torch.device("cuda")
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 11008
SUBSPACE_START = 1536   # rows 0..1535 stay zero forever; rows 1536..2047 train
STATUS_FILE = "DEADBLOCK_STATUS.md"
CHECKPOINT_DIR = "checkpoints-deadblock-13k"


# ---------------------------------------------------------------------------
# Status logging helper
# ---------------------------------------------------------------------------
def log_status(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"\n[{ts}] {msg}"
    print(line, flush=True)
    with open(STATUS_FILE, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Dead Block Wrapper
# ---------------------------------------------------------------------------
class DeadBlockWrapper(nn.Module):
    """
    Wraps a frozen Qwen2 decoder layer and appends an injection-free FFN delta.

    Math (when bypass=False):
        out    = base_layer(hidden_states, *args, **kwargs)   [frozen]
        h      = out[0] if tuple else out
        ffn_in = norm2(h)
        delta  = down_proj(silu(gate_proj(ffn_in)) * up_proj(ffn_in))
        h_out  = h + delta
        return (h_out,) + out[1:]  if tuple else h_out

    down_proj rows 0..SUBSPACE_START-1 are always zero.
    """

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # norm2: deepcopy from base layer post_attention_layernorm, trainable
        self.norm2 = copy.deepcopy(base_layer.post_attention_layernorm)
        for p in self.norm2.parameters():
            p.requires_grad = True

        # gate_proj / up_proj: deepcopy from base mlp, trainable
        self.gate_proj = copy.deepcopy(base_layer.mlp.gate_proj)
        for p in self.gate_proj.parameters():
            p.requires_grad = True

        self.up_proj = copy.deepcopy(base_layer.mlp.up_proj)
        for p in self.up_proj.parameters():
            p.requires_grad = True

        # down_proj: zero-init Linear(intermediate_size -> hidden_size, no bias)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False,
                                   dtype=torch.bfloat16)
        nn.init.zeros_(self.down_proj.weight)

        self.bypass = False

    def zero_frozen_rows(self):
        """Zero down_proj rows outside the knowledge subspace. Call after every optimizer.step()."""
        with torch.no_grad():
            self.down_proj.weight[:SUBSPACE_START, :].zero_()

    def forward(self, hidden_states, *args, **kwargs):
        out = self.base_layer(hidden_states, *args, **kwargs)

        if self.bypass:
            return out

        h = out[0] if isinstance(out, tuple) else out

        ffn_in = self.norm2(h)
        delta = self.down_proj(F.silu(self.gate_proj(ffn_in)) * self.up_proj(ffn_in))
        h_out = h + delta

        if isinstance(out, tuple):
            return (h_out,) + out[1:]
        return h_out


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------
def patch_model(model):
    """Replace layers[17] and layers[30] with DeadBlockWrapper."""
    wrapper17 = DeadBlockWrapper(model.model.layers[17]).to(DEVICE, dtype=torch.bfloat16)
    wrapper30 = DeadBlockWrapper(model.model.layers[30]).to(DEVICE, dtype=torch.bfloat16)

    model.model.layers[17] = wrapper17
    model.model.layers[30] = wrapper30

    # Freeze everything outside the two wrappers' own trainable params
    for name, param in model.named_parameters():
        is_wrapper_trainable = (
            ("model.layers.17" in name and ".base_layer." not in name) or
            ("model.layers.30" in name and ".base_layer." not in name)
        )
        param.requires_grad = is_wrapper_trainable

    return model, wrapper17, wrapper30


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------
def load_resume(path, wrapper17, wrapper30):
    """Load {'l17','l30'} state dicts from a checkpoint and re-zero frozen rows."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    wrapper17.load_state_dict(ckpt["l17"])
    wrapper30.load_state_dict(ckpt["l30"])
    # Re-zero frozen rows immediately to guard against any dtype drift in the saved file
    wrapper17.zero_frozen_rows()
    wrapper30.zero_frozen_rows()
    log_status(f"Resumed from {path}; frozen rows re-zeroed.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DeltaDataset(Dataset):
    def __init__(self, train_data, deltas, max_samples=None):
        if max_samples is not None:
            train_data = train_data[:max_samples]
            deltas = deltas[:max_samples]
        self.train_data = train_data
        self.deltas = deltas

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.train_data[idx]["input_ids"],
            "target_mask": self.train_data[idx]["target_mask"],
            "target_delta": self.deltas[idx],
        }


def collate_fn(batch):
    max_len = min(128, max(len(x["input_ids"]) for x in batch))
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_deltas = torch.stack([x["target_delta"] for x in batch])
    for i, x in enumerate(batch):
        ln = min(max_len, len(x["input_ids"]))
        input_ids[i, :ln] = torch.tensor(x["input_ids"][:ln])
        target_mask[i, :ln] = torch.tensor(x["target_mask"][:ln])
    return input_ids, target_mask, target_deltas


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def run_validation(model, wrapper17, wrapper30, val_loader):
    """Compute mean delta cosine similarity and mean delta MSE on held-out set."""
    model.eval()
    h_container = {}

    def hook_fn(module, inp, output):
        h_container["h"] = output[0] if isinstance(output, tuple) else output

    hook_handle = model.model.layers[30].register_forward_hook(hook_fn)

    total_cosine = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, t_mask, target_deltas in val_loader:
            input_ids = input_ids.to(DEVICE)
            target_deltas = target_deltas.to(DEVICE, dtype=torch.bfloat16).squeeze(1)

            wrapper17.bypass = False
            wrapper30.bypass = False
            _ = model(input_ids)
            h_active = h_container["h"][:, -1, :].clone()

            wrapper17.bypass = True
            wrapper30.bypass = True
            _ = model(input_ids)
            h_bypass = h_container["h"][:, -1, :].clone()

            wrapper17.bypass = False
            wrapper30.bypass = False

            predicted_delta = h_active - h_bypass
            total_cosine += F.cosine_similarity(predicted_delta, target_deltas, dim=-1).mean().item()
            total_mse += F.mse_loss(predicted_delta, target_deltas).item()
            n_batches += 1

    hook_handle.remove()
    val_cosine = total_cosine / max(n_batches, 1)
    val_mse = total_mse / max(n_batches, 1)
    return val_cosine, val_mse


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------
def save_checkpoint(wrapper17, wrapper30, epoch, val_cosine, val_mse, path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.save(
        {
            "l17": wrapper17.state_dict(),
            "l30": wrapper30.state_dict(),
            "meta": {
                "epoch": epoch,
                "val_cosine": val_cosine,
                "val_mse": val_mse,
                "timestamp": ts,
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------
def run_parity_check(model, tokenizer, wrapper17, wrapper30):
    log_status("=== PARITY CHECK: verifying zero-delta with down_proj=0 ===")

    prompts = [
        "What is the capital of France?",
        "def fibonacci(n):",
        "import os\nprint(os.getcwd())",
    ]

    model.eval()
    max_diffs = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Active pass (bypass=False, down_proj is all zeros -> delta=0)
        wrapper17.bypass = False
        wrapper30.bypass = False
        with torch.no_grad():
            logits_active = model(**inputs).logits

        # Bypass pass
        wrapper17.bypass = True
        wrapper30.bypass = True
        with torch.no_grad():
            logits_bypass = model(**inputs).logits

        # Reset
        wrapper17.bypass = False
        wrapper30.bypass = False

        diff = (logits_active - logits_bypass).abs().max().item()
        max_diffs.append(diff)
        log_status(f"  Prompt '{prompt[:40]}' max_abs_diff = {diff}")

    overall_max = max(max_diffs)
    if overall_max == 0.0:
        log_status("PARITY CHECK PASSED: max_abs_diff == 0 across all prompts.")
    else:
        log_status(
            f"PARITY CHECK FAILED: max_abs_diff = {overall_max}. "
            "Debug before training. Aborting."
        )
        sys.exit(1)

    return overall_max


# ---------------------------------------------------------------------------
# Spot-check recall
# ---------------------------------------------------------------------------
def spot_check_recall(model, tokenizer, docs, wrapper17, wrapper30, n=10):
    log_status("=== SPOT-CHECK RECALL (10 symbols, greedy 60 tokens) ===")
    model.eval()
    results = []

    for i in range(n):
        doc = docs[i]
        lines = doc.strip().split("\n")
        symbol = lines[0].strip()
        prompt = f"Question: How do I use {symbol} in Python?\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        wrapper17.bypass = False
        wrapper30.bypass = False
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        symbol_in = symbol.lower() in completion.lower()
        verdict = "HIT" if symbol_in else "MISS"
        results.append((symbol, completion[:120], verdict))
        log_status(f"  [{verdict}] Symbol: {symbol}\n    Completion: {completion[:120]}")

    hits = sum(1 for _, _, v in results if v == "HIT")
    log_status(f"Spot-check recall: {hits}/{n} symbols mentioned in completions.")
    return hits, n


# ---------------------------------------------------------------------------
# Training loop (shared for stage A and B)
# ---------------------------------------------------------------------------
def train_loop(model, tokenizer, wrapper17, wrapper30, loader, optimizer, stage_name, total_steps_hint):
    model.train()

    h_container = {}

    def hook_fn(module, inp, output):
        h_container["h"] = output[0] if isinstance(output, tuple) else output

    hook_handle = model.model.layers[30].register_forward_hook(hook_fn)

    running_lm = 0.0
    running_delta = 0.0
    running_cosine = 0.0
    step = 0
    early_delta_losses = []
    late_delta_losses = []

    for input_ids, t_mask, target_deltas in loader:
        model.train()
        input_ids = input_ids.to(DEVICE)
        t_mask = t_mask.to(DEVICE)
        target_deltas = target_deltas.to(DEVICE, dtype=torch.bfloat16).squeeze(1)

        # --- Active forward ---
        wrapper17.bypass = False
        wrapper30.bypass = False
        outputs = model(input_ids)
        logits = outputs.logits
        h_active = h_container["h"][:, -1, :].clone()

        # --- Ignorant (bypass) forward ---
        with torch.no_grad():
            wrapper17.bypass = True
            wrapper30.bypass = True
            _ = model(input_ids)
            h_bypass = h_container["h"][:, -1, :].clone()
            wrapper17.bypass = False
            wrapper30.bypass = False

        # --- LM loss ---
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = t_mask[:, 1:].contiguous()
        flat_labels = shift_labels.view(-1)
        flat_labels[shift_mask.view(-1) == 0] = -100
        loss_lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            flat_labels,
            ignore_index=-100,
        )

        # --- Delta loss ---
        predicted_delta = h_active - h_bypass
        loss_mse = F.mse_loss(predicted_delta, target_deltas)
        cos_sim = F.cosine_similarity(predicted_delta, target_deltas, dim=-1).mean()
        loss_delta = loss_mse + (1.0 - cos_sim)

        loss = loss_lm + 10.0 * loss_delta
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Enforce subspace constraint
        wrapper17.zero_frozen_rows()
        wrapper30.zero_frozen_rows()

        running_lm += loss_lm.item()
        running_delta += loss_delta.item()
        running_cosine += cos_sim.item()
        step += 1

        if step <= 200:
            early_delta_losses.append(loss_delta.item())
        if step >= max(1, total_steps_hint - 200):
            late_delta_losses.append(loss_delta.item())

        if step % 100 == 0:
            avg_lm = running_lm / 100
            avg_d = running_delta / 100
            avg_cos = running_cosine / 100
            log_status(
                f"[{stage_name}] Step {step}/{total_steps_hint} | "
                f"LossLM: {avg_lm:.4f} | LossDelta: {avg_d:.4f} | CosSim: {avg_cos:.4f}"
            )
            running_lm = 0.0
            running_delta = 0.0
            running_cosine = 0.0

    hook_handle.remove()
    return early_delta_losses, late_delta_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b"], default="a")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (stage B only; stage A is always 1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint with {'l17','l30'} keys to resume from")
    parser.add_argument("--val-size", type=int, default=200,
                        help="Number of samples held out from the END of the dataset for validation")
    args = parser.parse_args()

    log_status(f"=== train_dead_block.py started, stage={args.stage} epochs={args.epochs} ===")

    # Load data
    log_status("Loading data...")
    py_train_data = torch.load("python_13k_train_corpus.pt", weights_only=False)
    py_deltas = torch.load("python_13k_deltas.pt", weights_only=False)
    py_docs = torch.load("python_13k_docs.pt", weights_only=False)
    log_status(f"Loaded {len(py_train_data)} train samples.")

    # Load model
    log_status(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model, wrapper17, wrapper30 = patch_model(model)
    model = model.to(DEVICE)

    # Resume if requested (before any training, after patch so wrappers exist)
    if args.resume is not None:
        load_resume(args.resume, wrapper17, wrapper30)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_status(f"Trainable parameters: {trainable:,}")

    # -----------------------------------------------------------------------
    # Stage A
    # -----------------------------------------------------------------------
    if args.stage == "a":
        run_parity_check(model, tokenizer, wrapper17, wrapper30)

        max_frozen_init = max(
            wrapper17.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
            wrapper30.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
        )
        log_status(f"Init frozen rows max abs: {max_frozen_init} (expected 0.0)")

        log_status("=== STAGE A: training on first 2000 samples, 1 epoch ===")
        dataset_a = DeltaDataset(py_train_data, py_deltas, max_samples=2000)
        loader_a = DataLoader(dataset_a, batch_size=1, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=0.1,
        )

        early_losses, late_losses = train_loop(
            model, tokenizer, wrapper17, wrapper30,
            loader_a, optimizer, "StageA", total_steps_hint=2000,
        )

        max_frozen_after = max(
            wrapper17.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
            wrapper30.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
        )
        log_status(f"After stage A, frozen rows max abs: {max_frozen_after} (expected 0.0)")

        spot_check_recall(model, tokenizer, py_docs, wrapper17, wrapper30, n=10)

        # Gate decision
        if early_losses and late_losses:
            avg_early = sum(early_losses) / len(early_losses)
            avg_late = sum(late_losses) / len(late_losses)
            pct_drop = (avg_early - avg_late) / (avg_early + 1e-9) * 100.0
        else:
            avg_early = avg_late = pct_drop = 0.0

        log_status(
            f"Gate evaluation: avg_early_delta={avg_early:.4f}, "
            f"avg_late_delta={avg_late:.4f}, drop={pct_drop:.1f}%"
        )
        gate_pass = pct_drop >= 30.0
        log_status(f"GATE DECISION: {'PASS' if gate_pass else 'FAIL'} (need >=30% drop, got {pct_drop:.1f}%)")

        if gate_pass:
            log_status("Gate passed. Launching Stage B as background process.")
            cmd = "nohup python train_dead_block.py --stage b > deadblock_stageb.log 2>&1 &"
            os.system(cmd)
            log_status(f"Stage B launched: `{cmd}`")
            log_status("Monitor with: tail -f deadblock_stageb.log")
        else:
            log_status(
                "Gate FAILED. Stage B NOT launched.\n"
                "Analysis: delta loss did not decrease >=30% during stage A.\n"
                "Likely causes: (1) without runtime injection the FFN cannot reconstruct "
                "the injection-derived delta from prompt alone with only 2000 samples; "
                "(2) the subspace constraint limits expressivity; "
                "(3) lr or warmup may need tuning.\n"
                "Next steps: inspect loss trajectory above; consider staged warmup, "
                "larger stage-A window, or relaxing the subspace constraint to rows 512+."
            )

    # -----------------------------------------------------------------------
    # Stage B
    # -----------------------------------------------------------------------
    elif args.stage == "b":
        n_total = len(py_train_data)
        val_size = min(args.val_size, n_total - 1)
        train_size = n_total - val_size

        # Deterministic split: last val_size samples held out, no shuffle of the split
        train_data = py_train_data[:train_size]
        train_deltas = py_deltas[:train_size]
        val_data = py_train_data[train_size:]
        val_deltas = py_deltas[train_size:]

        log_status(
            f"=== STAGE B: {train_size} train / {val_size} val, "
            f"{args.epochs} epoch(s) ===")

        dataset_b = DeltaDataset(train_data, train_deltas)
        val_dataset = DeltaDataset(val_data, val_deltas)
        loader_b = DataLoader(dataset_b, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=0.1,
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        best_val_cosine = -float("inf")

        for epoch in range(1, args.epochs + 1):
            stage_label = f"StageB-ep{epoch}"
            log_status(f"=== Epoch {epoch}/{args.epochs} ===")

            early_losses, late_losses = train_loop(
                model, tokenizer, wrapper17, wrapper30,
                loader_b, optimizer, stage_label, total_steps_hint=len(dataset_b),
            )

            # Safety invariant: verify frozen rows after each epoch
            max_frozen_after = max(
                wrapper17.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
                wrapper30.down_proj.weight[:SUBSPACE_START, :].abs().max().item(),
            )
            log_status(
                f"[Epoch {epoch}] Frozen rows max abs after training: "
                f"{max_frozen_after} (must be 0.0)"
            )
            if max_frozen_after != 0.0:
                log_status("WARNING: frozen rows non-zero after epoch — re-zeroing now.")
                wrapper17.zero_frozen_rows()
                wrapper30.zero_frozen_rows()

            # Validation
            val_cosine, val_mse = run_validation(model, wrapper17, wrapper30, val_loader)
            log_status(
                f"[Epoch {epoch}] Val cosine: {val_cosine:.6f} | Val MSE: {val_mse:.6f}"
            )

            # Per-epoch checkpoint
            epoch_path = os.path.join(CHECKPOINT_DIR, f"dead_blocks_epoch{epoch}.pt")
            save_checkpoint(wrapper17, wrapper30, epoch, val_cosine, val_mse, epoch_path)

            # Best checkpoint
            is_best = val_cosine > best_val_cosine
            if is_best:
                best_val_cosine = val_cosine
                best_path = os.path.join(CHECKPOINT_DIR, "dead_blocks_best.pt")
                save_checkpoint(wrapper17, wrapper30, epoch, val_cosine, val_mse, best_path)

            # Always write last
            last_path = os.path.join(CHECKPOINT_DIR, "dead_blocks_last.pt")
            save_checkpoint(wrapper17, wrapper30, epoch, val_cosine, val_mse, last_path)

            # Legacy name = best (keeps exporter default working)
            legacy_path = os.path.join(CHECKPOINT_DIR, "dead_blocks.pt")
            save_checkpoint(wrapper17, wrapper30, epoch, val_cosine, val_mse, legacy_path)

            # Compute train loss summary for log
            if early_losses:
                avg_early = sum(early_losses) / len(early_losses)
            else:
                avg_early = float("nan")
            if late_losses:
                avg_late = sum(late_losses) / len(late_losses)
            else:
                avg_late = float("nan")

            written = [epoch_path, last_path, legacy_path]
            if is_best:
                written.append(best_path)

            log_status(
                f"[Epoch {epoch} SUMMARY] "
                f"train_delta_early={avg_early:.4f} train_delta_late={avg_late:.4f} | "
                f"val_cosine={val_cosine:.6f} val_mse={val_mse:.6f} | "
                f"best_improved={'YES' if is_best else 'NO'} | "
                f"saved={[os.path.basename(p) for p in written]}"
            )

        log_status(
            f"Stage B complete. Best val cosine: {best_val_cosine:.6f}. "
            f"Checkpoints in {CHECKPOINT_DIR}/"
        )


if __name__ == "__main__":
    main()
