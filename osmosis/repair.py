"""Repair LoRA: distill original model knowledge into crushed model via LoRA adapter.

Two-phase offline distillation: cache teacher logits first (then free teacher),
train student LoRA against cached logits. Fits on single GPU even for large models.
"""
import gc
import math
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from osmosis.loader import OsmosisModel


def load_calibration_data(tokenizer, num_samples=512, max_length=256, seed=42):
    parquet_path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()

    rng = random.Random(seed)
    rng.shuffle(texts)

    samples = []
    for text in texts:
        text = text.strip()
        if len(text) < 50:
            continue
        tokens = tokenizer(
            text, truncation=True, max_length=max_length,
            return_tensors="pt", padding=False,
        )
        if tokens["input_ids"].shape[1] >= 32:
            samples.append(tokens)
        if len(samples) >= num_samples:
            break

    print(f"Loaded {len(samples)} calibration samples (max_length={max_length})")
    return samples


def train_repair_lora(
    original_model_path: str,
    crush_dir: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    epochs: int = 3,
    batch_size: int = 4,
    num_samples: int = 512,
    max_length: int = 256,
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Osmosis Repair LoRA Training ===")
    print(f"Original: {original_model_path}")
    print(f"Crush dir: {crush_dir}")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}, lr={lr}")
    print(f"Samples: {num_samples}, max_length: {max_length}, epochs: {epochs}")

    tokenizer = AutoTokenizer.from_pretrained(
        original_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading calibration data...")
    samples = load_calibration_data(tokenizer, num_samples, max_length)

    # Phase 1: cache teacher predictions, then free teacher
    # Store top-K logprobs per token (not full vocab — that's 248K * samples * seq_len)
    top_k = 64
    print(f"\n--- Phase 1: Caching teacher top-{top_k} logprobs ---")
    teacher = AutoModelForCausalLM.from_pretrained(
        original_model_path, dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    teacher.eval()

    cached_teacher = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.float()
            log_probs = F.log_softmax(logits, dim=-1)
            topk_vals, topk_ids = log_probs.topk(top_k, dim=-1)
            cached_teacher.append({
                "topk_logp": topk_vals.cpu(),
                "topk_ids": topk_ids.cpu(),
                "argmax": logits.argmax(dim=-1).cpu(),
            })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(samples)} samples cached")

    cache_mb = sum(
        t["topk_logp"].nbytes + t["topk_ids"].nbytes + t["argmax"].nbytes
        for t in cached_teacher
    ) / 1e6
    print(f"  Cached {len(cached_teacher)} samples ({cache_mb:.0f} MB)")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print("  Teacher freed")

    # Phase 2: load student + LoRA, train against cached logits
    print("\n--- Phase 2: Training student LoRA ---")
    student_wrapper = OsmosisModel(
        crush_dir, original_model_path,
        device="cpu", dtype=torch.float16,
    )
    student = student_wrapper.model
    student = student.to(device)
    print(f"  Student moved to {device}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    student = get_peft_model(student, lora_config)
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"LoRA params: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.2f}%)")
    print("Gradient checkpointing enabled")

    if hasattr(student, "hf_device_map"):
        input_device = next(iter(student.hf_device_map.values()))
        if isinstance(input_device, int):
            input_device = f"cuda:{input_device}"
    else:
        input_device = device
    print(f"Input device: {input_device}")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    num_steps = epochs * math.ceil(len(samples) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    print(f"\nTraining: {epochs} epochs, {num_steps} steps")
    print("=" * 60)

    student.train()
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        indices = torch.randperm(len(samples))

        for i in range(0, len(samples), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_loss = 0.0

            for idx in batch_indices:
                sample = samples[idx]
                input_ids = sample["input_ids"].to(input_device)
                attention_mask = sample["attention_mask"].to(input_device)
                teacher_cache = cached_teacher[idx]
                teacher_argmax = teacher_cache["argmax"].to(input_device)
                teacher_topk_logp = teacher_cache["topk_logp"].to(input_device)
                teacher_topk_ids = teacher_cache["topk_ids"].to(input_device)

                student_out = student(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logp = F.log_softmax(
                    student_out.logits.float(), dim=-1
                )

                # Sparse KL: gather student logprobs at teacher's top-K positions
                student_at_topk = student_logp.gather(-1, teacher_topk_ids)
                teacher_topk_probs = teacher_topk_logp.exp()
                kl = (teacher_topk_probs * (teacher_topk_logp - student_at_topk)).sum(-1).mean()

                # CE against teacher's hard predictions (shifted)
                ce = F.cross_entropy(
                    student_out.logits[0, :-1].float(),
                    teacher_argmax[0, 1:],
                )
                loss = 0.7 * kl + 0.3 * ce
                batch_loss += loss

            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += batch_loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % 20 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (num_steps - global_step) / max(steps_per_sec, 0.01)
                print(f"  step {global_step}/{num_steps}  "
                      f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"ETA={eta / 60:.1f}min")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            student.save_pretrained(str(output_path))
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"LoRA saved to: {output_path}")

    lora_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    print(f"LoRA size: {lora_size / 1024 / 1024:.1f} MB")

    return str(output_path)


@torch.no_grad()
def evaluate_repair(
    original_model_path: str,
    crush_dir: str,
    lora_path: str,
    device: str = "cuda",
    num_prompts: int = 8,
):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        original_model_path, trust_remote_code=True
    )

    prompts = [
        "The meaning of life is",
        "In a shocking finding, scientists discovered",
        "def fibonacci(n):",
        "The capital of France is",
        "Once upon a time in a land far away",
        "The key difference between TCP and UDP is",
        "import torch\n\ndef train_step(",
        "According to recent studies, climate change",
    ][:num_prompts]

    # Cache original logits then free
    print("Loading original for logit caching...")
    original = AutoModelForCausalLM.from_pretrained(
        original_model_path, dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    original.eval()

    orig_data = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = original(**tokens)
        orig_data.append({
            "logp": F.log_softmax(out.logits.float(), dim=-1).cpu(),
            "ppl": torch.exp(F.cross_entropy(
                out.logits[0, :-1], tokens["input_ids"][0, 1:]
            )).item(),
            "input_ids": tokens["input_ids"].cpu(),
        })
    del original
    gc.collect()
    torch.cuda.empty_cache()

    # Load crushed (no LoRA) and evaluate
    print("Loading crushed (no LoRA)...")
    bare = OsmosisModel(crush_dir, original_model_path, device=device)

    bare_kls, bare_ppls = [], []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        bare_out = bare.model(**tokens)
        bare_logp = F.log_softmax(bare_out.logits.float(), dim=-1)
        orig_logp = orig_data[i]["logp"].to(device)
        bare_kls.append(F.kl_div(bare_logp, orig_logp.exp(), reduction="batchmean", log_target=False).item())
        bare_ppls.append(torch.exp(F.cross_entropy(bare_out.logits[0, :-1], tokens["input_ids"][0, 1:])).item())
    del bare
    gc.collect()
    torch.cuda.empty_cache()

    # Load crushed + repair LoRA and evaluate
    print("Loading crushed + repair LoRA...")
    repaired_wrapper = OsmosisModel(crush_dir, original_model_path, device=device)
    repaired = PeftModel.from_pretrained(repaired_wrapper.model, lora_path)
    repaired.eval()

    repaired_kls, repaired_ppls = [], []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        rep_out = repaired(**tokens)
        rep_logp = F.log_softmax(rep_out.logits.float(), dim=-1)
        orig_logp = orig_data[i]["logp"].to(device)
        repaired_kls.append(F.kl_div(rep_logp, orig_logp.exp(), reduction="batchmean", log_target=False).item())
        repaired_ppls.append(torch.exp(F.cross_entropy(rep_out.logits[0, :-1], tokens["input_ids"][0, 1:])).item())
    del repaired
    gc.collect()
    torch.cuda.empty_cache()

    # Print results
    print(f"\nResults on {len(prompts)} prompts:")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}] {prompt[:50]}...")
        print(f"  Original PPL:   {orig_data[i]['ppl']:8.2f}")
        print(f"  Crushed PPL:    {bare_ppls[i]:8.2f}  KL={bare_kls[i]:.4f}")
        print(f"  Repaired PPL:   {repaired_ppls[i]:8.2f}  KL={repaired_kls[i]:.4f}")
        kl_red = (1 - repaired_kls[i] / max(bare_kls[i], 1e-8)) * 100
        print(f"  KL reduction:   {kl_red:.1f}%")

    orig_ppls = [d["ppl"] for d in orig_data]
    print(f"\n{'=' * 70}")
    print(f"Mean KL  -- bare: {sum(bare_kls)/len(bare_kls):.4f}  "
          f"repaired: {sum(repaired_kls)/len(repaired_kls):.4f}  "
          f"reduction: {(1 - sum(repaired_kls)/max(sum(bare_kls), 1e-8)) * 100:.1f}%")
    print(f"Mean PPL -- orig: {sum(orig_ppls)/len(orig_ppls):.1f}  "
          f"bare: {sum(bare_ppls)/len(bare_ppls):.1f}  "
          f"repaired: {sum(repaired_ppls)/len(repaired_ppls):.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Osmosis repair LoRA")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train repair LoRA")
    train.add_argument("--model", required=True, help="Original model path")
    train.add_argument("--crush-dir", required=True, help="Crushed model directory")
    train.add_argument("--output", required=True, help="Output directory for LoRA")
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--num-samples", type=int, default=512)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--device", default="cuda")

    evl = sub.add_parser("evaluate", help="Evaluate repaired model")
    evl.add_argument("--model", required=True, help="Original model path")
    evl.add_argument("--crush-dir", required=True, help="Crushed model directory")
    evl.add_argument("--lora", required=True, help="Path to repair LoRA")
    evl.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.command == "train":
        train_repair_lora(
            args.model, args.crush_dir, args.output,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, num_samples=args.num_samples,
            max_length=args.max_length, device=args.device,
        )
    elif args.command == "evaluate":
        evaluate_repair(args.model, args.crush_dir, args.lora, device=args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
