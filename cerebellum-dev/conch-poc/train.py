"""
Conch Shell Training — Phase 1: Fixed Loops

Curriculum:
  Phase 1: Fixed 2 loops, learn to use repetition (this script)
  Phase 2: Fixed 4 loops, learn deeper refinement
  Phase 3: Adaptive exit with teacher signal

Training data: WikiText-2 for proof of concept, NIM teachers for production.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import autocast
import time
import json
from pathlib import Path

from model import load_conch


class TextDataset(Dataset):
    """Simple chunked text dataset."""

    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(torch.tensor(tokens[i : i + block_size], dtype=torch.long))

        print(f"Dataset: {len(self.examples)} chunks of {block_size} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class NIMDataset(Dataset):
    """Dataset from NIM teacher generations (JSONL format)."""

    def __init__(self, tokenizer, file_path, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", data.get("content", data.get("response", "")))
                if text:
                    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
                    if len(tokens) >= 32:
                        self.examples.append(torch.tensor(tokens, dtype=torch.long))

        print(f"NIM Dataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : x.size(0)] = x
        masks[i, : x.size(0)] = 1
    return padded, masks


def train_phase(
    model,
    tokenizer,
    data_path,
    output_dir,
    fixed_revolutions=2,
    epochs=3,
    batch_size=4,
    lr=5e-5,
    block_size=512,
    grad_accum_steps=4,
    use_nim_data=False,
):
    """Train one phase of the conch curriculum."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Load data
    if use_nim_data:
        dataset = NIMDataset(tokenizer, data_path, max_length=block_size)
    else:
        dataset = TextDataset(tokenizer, data_path, block_size=block_size)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    # Optimizer — Phase 1: only loop components. Phase 2+: unfreeze all.
    trainable_params = []
    frozen_params = []
    if fixed_revolutions <= 2:
        for name, param in model.named_parameters():
            if any(k in name for k in ["loop_block", "exit_head", "revolution_embeddings"]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
                frozen_params.append(param)
    else:
        for param in model.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen params: {sum(p.numel() for p in frozen_params):,}")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    log_every = 50

    print(f"\nPhase: {fixed_revolutions} fixed revolutions")
    print(f"Epochs: {epochs}, Batch: {batch_size}, Grad accum: {grad_accum_steps}")
    print(f"Effective batch: {batch_size * grad_accum_steps}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_steps = 0
        t0 = time.time()

        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    fixed_revolutions=fixed_revolutions,
                )
                loss = outputs["loss"] / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs["loss"].item()
            epoch_steps += 1

            if (step + 1) % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                tok_per_sec = (epoch_steps * batch_size * block_size) / elapsed
                confidences = [c.mean().item() for c in outputs["exit_confidences"]]
                print(
                    f"  [{epoch+1}/{epochs}] step {step+1}/{len(dataloader)} | "
                    f"loss: {avg_loss:.4f} | "
                    f"tok/s: {tok_per_sec:.0f} | "
                    f"exit_conf: {confidences[-1]:.3f}"
                )

        # Epoch summary
        avg_loss = epoch_loss / epoch_steps
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} done | avg_loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "fixed_revolutions": fixed_revolutions,
                },
                os.path.join(output_dir, "best_checkpoint.pt"),
            )
            print(f"  Saved best checkpoint (loss: {best_loss:.4f})")

    # Save final
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "loss": avg_loss,
            "fixed_revolutions": fixed_revolutions,
        },
        os.path.join(output_dir, "final_checkpoint.pt"),
    )
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Conch Shell model")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--nim-data", action="store_true", help="Use NIM JSONL format")
    args = parser.parse_args()

    # Default data path
    if args.data is None:
        args.data = "/var/home/deucebucket/games/osmosis-quants/wiki.test.raw"

    # Revolution schedule per phase
    revolutions = {1: 2, 2: 4, 3: 6}
    fixed_rev = revolutions[args.phase]

    print(f"Conch Shell Training — Phase {args.phase} ({fixed_rev} revolutions)")
    print(f"Base model: {args.base_model}")
    print(f"Data: {args.data}")

    # Load model
    model, tokenizer = load_conch(args.base_model)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    # Train
    output_dir = os.path.join(args.output, f"phase{args.phase}")
    train_phase(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data,
        output_dir=output_dir,
        fixed_revolutions=fixed_rev,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        block_size=args.block_size,
        use_nim_data=args.nim_data,
    )


if __name__ == "__main__":
    main()
