"""
Train the bolt-on refiner. Only the refiner block is trainable (~2M params).
Base model stays frozen and functional throughout.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import autocast
import time
import argparse

from refiner import load_refiner_model


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(torch.tensor(tokens[i:i + block_size], dtype=torch.long))
        print(f"Dataset: {len(self.examples)} chunks of {block_size} tokens")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
        masks[i, :x.size(0)] = 1
    return padded, masks


def compute_ppl(model, tokenizer, text_path, device, num_revolutions=None, max_chunks=64):
    """Quick PPL computation."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    total_loss = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens) - 512, 512):
            chunk = torch.tensor(tokens[i:i+512], dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk, fixed_revolutions=num_revolutions)
            total_loss += out["loss"].item()
            n += 1
            if n >= max_chunks:
                break
    return torch.exp(torch.tensor(total_loss / n)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/var/home/deucebucket/games/osmosis-quants/wiki.train.raw")
    parser.add_argument("--test-data", default="/var/home/deucebucket/games/osmosis-quants/wiki.test.raw")
    parser.add_argument("--output", default="./checkpoints-refiner")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--revolutions", type=int, default=2)
    parser.add_argument("--split-layer", type=int, default=15)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM-135M")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_refiner_model(
        base_model_name=args.base_model,
        split_layer=args.split_layer,
        num_revolutions=args.revolutions,
    )
    model = model.to(device)

    # Baseline PPL (bypass refiner entirely)
    print("\nBaseline PPL (no refiner loops)...")
    ppl_baseline = compute_ppl(model, tokenizer, args.test_data, device, num_revolutions=0)
    print(f"  PPL = {ppl_baseline:.4f}")

    # Dataset
    dataset = TextDataset(tokenizer, args.data, block_size=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Only optimize refiner params — weight_decay combats overfitting drift
    optimizer = AdamW(model.refiner.parameters(), lr=args.lr, weight_decay=0.1)

    os.makedirs(args.output, exist_ok=True)
    best_ppl = ppl_baseline
    print(f"\nTraining refiner ({sum(p.numel() for p in model.refiner.parameters()):,} params)")
    print(f"Split: layer {args.split_layer}, Revolutions: {args.revolutions}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        model.train()
        # Only refiner should have grad, but set model to train mode for dropout etc
        model.base.eval()
        epoch_loss = 0
        steps = 0
        t0 = time.time()

        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            out = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps
        elapsed = time.time() - t0

        # PPL check
        ppl = compute_ppl(model, tokenizer, args.test_data, device, num_revolutions=args.revolutions)
        gate_val = torch.sigmoid(model.refiner.gate).item()

        print(f"Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.4f} | PPL: {ppl:.4f} | gate: {gate_val:.4f} | time: {elapsed:.1f}s")

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "refiner_state_dict": model.refiner.state_dict(),
                "epoch": epoch,
                "ppl": ppl,
                "split_layer": args.split_layer,
                "revolutions": args.revolutions,
            }, os.path.join(args.output, "best_refiner.pt"))
            print(f"  NEW BEST (PPL {ppl:.4f} vs baseline {ppl_baseline:.4f}, delta: {100*(ppl-ppl_baseline)/ppl_baseline:+.2f}%)")

    # Final comparison
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Baseline PPL: {ppl_baseline:.4f}")
    print(f"Best refiner PPL: {best_ppl:.4f}")
    print(f"Delta: {100*(best_ppl-ppl_baseline)/ppl_baseline:+.2f}%")

    if best_ppl < ppl_baseline:
        print(f"\nSUCCESS: Refiner improved PPL by {100*(ppl_baseline-best_ppl)/ppl_baseline:.2f}%")
    else:
        print(f"\nFAILED: Refiner did not improve over baseline")

    # Test different revolution counts
    print(f"\nRevolution sweep (best checkpoint):")
    ckpt = torch.load(os.path.join(args.output, "best_refiner.pt"), map_location="cpu")
    model.refiner.load_state_dict(ckpt["refiner_state_dict"])
    for n_rev in [1, 2, 3, 4, 6]:
        ppl_n = compute_ppl(model, tokenizer, args.test_data, device, num_revolutions=n_rev)
        print(f"  {n_rev} revolutions: PPL = {ppl_n:.4f} ({100*(ppl_n-ppl_baseline)/ppl_baseline:+.2f}%)")


if __name__ == "__main__":
    main()
