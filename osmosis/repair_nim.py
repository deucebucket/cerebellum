"""Train repair LoRA using NIM remote teacher — zero VRAM for teacher.

Generates training data from NIM, then trains LoRA on the student model
with simulated quantization applied to match the GGUF allocation. The LoRA
learns to compensate for the specific quant noise in each tensor.

The teacher never touches the local GPU — only the student needs VRAM.
"""
import gc
import json
import math
import random
import time
from pathlib import Path

import httpx
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


PMS_BASE = "http://localhost:7700"


# --- Simulated quantization (matches sensitivity_multi.py) ---

def quantize_blockwise(tensor, bits, group_size=32):
    flat = tensor.float().reshape(-1)
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    max_val = (1 << (bits - 1)) - 1
    normalized = groups / scales
    quantized = (normalized * max_val).round().clamp(-max_val, max_val)
    dequantized = (quantized / max_val) * scales
    return dequantized.reshape(-1)[:n].reshape(tensor.shape).to(tensor.dtype)


# Map GGUF quant tier names to simulated bit depths
TIER_TO_BITS = {
    "q2_K": 2, "q3_K": 3, "q4_K": 4, "q5_K": 5,
    "q6_K": 6, "q8_0": 8, "f16": 16, "f32": 32,
}

# HF module name → GGUF tensor name component
HF_TO_GGUF = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.in_proj_z": "ssm_in",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def load_tensor_type_map(tensor_types_path):
    """Load tensor type assignments from budget allocator output."""
    assignments = {}
    with open(tensor_types_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                name, qtype = line.split("=", 1)
                assignments[name] = qtype
    return assignments


def apply_simulated_quantization(model, tensor_types_path):
    """Apply simulated GGUF quantization to model weights in-place."""
    assignments = load_tensor_type_map(tensor_types_path)
    applied = 0
    skipped = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip — LoRA params shouldn't be quantized
        if "lora_" in name:
            continue

        # Map HF name to GGUF name to look up assignment
        # e.g. model.layers.0.mlp.down_proj.weight → blk.0.ffn_down.weight
        gguf_name = _hf_name_to_gguf(name)
        if gguf_name and gguf_name in assignments:
            qtype = assignments[gguf_name]
            bits = TIER_TO_BITS.get(qtype, 16)
            if bits < 16:
                with torch.no_grad():
                    param.data = quantize_blockwise(param.data, bits)
                applied += 1
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"Simulated quantization: {applied} tensors quantized, {skipped} kept as-is")
    return applied


def _hf_name_to_gguf(hf_name):
    """Convert HF parameter name to GGUF tensor name."""
    import re
    # model.layers.X.mlp.down_proj.weight → blk.X.ffn_down.weight
    # model.language_model.layers.X... → blk.X...
    m = re.match(r"(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(.+?)\.weight$", hf_name)
    if not m:
        return None
    layer_num = m.group(1)
    component = m.group(2)
    gguf_comp = HF_TO_GGUF.get(component)
    if gguf_comp:
        return f"blk.{layer_num}.{gguf_comp}.weight"
    return None


# --- Training data generators ---

def load_wikitext_prompts(num_samples=512, seed=42):
    if HAS_PYARROW:
        try:
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
                if len(text) < 100:
                    continue
                samples.append(text[:512])
                if len(samples) >= num_samples:
                    break
            print(f"Loaded {len(samples)} wikitext prompts")
            return samples
        except Exception as e:
            print(f"Wikitext download failed: {e}")

    print("Using synthetic calibration prompts")
    prompts = [
        "The theory of general relativity describes gravity as a geometric property of spacetime.",
        "In computer science, a hash table is a data structure that implements an associative array.",
        "The mitochondria is the powerhouse of the cell, producing ATP through oxidative phosphorylation.",
        "Machine learning algorithms can be broadly categorized into supervised, unsupervised, and reinforcement learning.",
        "The human genome contains approximately 3 billion base pairs of DNA organized into 23 chromosome pairs.",
        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement.",
        "Neural networks consist of interconnected nodes organized in layers that process information.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
        "Economic theory suggests that markets tend toward equilibrium where supply meets demand.",
        "Artificial intelligence has made remarkable progress in natural language processing and computer vision.",
        "The laws of thermodynamics govern energy transfer and the direction of natural processes.",
        "The periodic table organizes chemical elements by atomic number and recurring chemical properties.",
        "Climate change is driven primarily by the accumulation of greenhouse gases in the atmosphere.",
        "Protein folding determines the three-dimensional structure and function of biological molecules.",
        "Modern cryptography relies on mathematical problems that are computationally difficult to solve.",
        "Evolutionary biology explains the diversity of life through natural selection and genetic variation.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        "Calculus was independently developed by Newton and Leibniz in the late 17th century.",
        "The human brain contains roughly 86 billion neurons connected by trillions of synapses.",
        "The Standard Model of particle physics describes the fundamental forces and elementary particles.",
    ]
    rng = random.Random(seed)
    while len(prompts) < num_samples:
        prompts.extend(prompts[:20])
    rng.shuffle(prompts)
    return prompts[:num_samples]


VADUGWI_PROMPTS = [
    "I feel like I'm falling apart and nobody notices. Write a response that validates this feeling.",
    "My partner said something that really hurt me. I don't know if I'm overreacting.",
    "I failed at something I worked really hard on and I feel worthless.",
    "Everyone seems to have their life together except me.",
    "I've been putting on a brave face but inside I'm struggling.",
    "Someone took credit for my work and nobody believes me.",
    "I keep getting interrupted and talked over in meetings.",
    "My roommate keeps disrespecting my boundaries no matter how many times I ask.",
    "I'm furious about an injustice I witnessed and nobody seems to care.",
    "Someone I trusted betrayed my confidence.",
    "I saw someone being cruel to an animal and I can't stop thinking about it.",
    "A public figure said something deeply offensive about people like me.",
    "I found out my company has been doing something unethical.",
    "Someone I respected turned out to be a terrible person.",
    "The hypocrisy in this situation is making me sick.",
    "I have a major life decision to make and I'm paralyzed by the options.",
    "I don't know if I should stay in this relationship.",
    "My career path doesn't feel right but I don't know what else to do.",
    "I'm not sure if my friend is genuine or just using me.",
    "Everything I believed in is being questioned and I don't know what's true anymore.",
    "Someone did something incredibly kind for me today, completely unprompted.",
    "I just realized how lucky I am to have the people in my life that I do.",
    "After years of struggle, something finally went right.",
    "A stranger's small act of kindness changed my whole day.",
    "I want to express how thankful I am but words feel inadequate.",
    "I desperately want to change my life but I don't know where to start.",
    "I wish I could go back and do things differently.",
    "I want deep connection but I'm afraid of getting hurt again.",
    "I crave purpose and meaning but everything feels empty.",
    "I want someone to truly understand me without having to explain everything.",
    "I just had a breakthrough idea and I'm buzzing with excitement.",
    "I witnessed an act of extraordinary courage today.",
    "Someone's story of overcoming adversity just gave me hope.",
    "I feel a calling to do something bigger than myself.",
    "For the first time in a long time, I feel like anything is possible.",
    "I'm happy for my friend's success but jealous at the same time and I feel terrible about it.",
    "I love my family but being around them makes me anxious.",
    "I got the promotion I wanted but now I feel like an impostor.",
    "I'm grieving someone who's still alive but no longer in my life.",
    "I feel relief that a difficult situation is over but guilty for feeling relieved.",
    "Explain the difference between empathy and sympathy with concrete examples.",
    "How do you help someone who's grieving without making it about yourself?",
    "What does emotional regulation look like in practice during a heated argument?",
    "Describe how unprocessed anger manifests in seemingly unrelated behaviors.",
    "What are the signs that someone is masking their true feelings?",
    "How do cultural differences affect the expression of grief?",
    "What is emotional flooding and how do you recover from it?",
    "Explain attachment styles and how they affect adult relationships.",
    "How do you set boundaries without damaging a relationship?",
    "What does healthy vulnerability look like in a professional setting?",
]


def load_domain_prompts(domain="vadugwi", num_samples=256, seed=42):
    if domain == "vadugwi":
        base_prompts = VADUGWI_PROMPTS
    else:
        raise ValueError(f"Unknown domain: {domain}")

    rng = random.Random(seed)
    prompts = base_prompts.copy()
    while len(prompts) < num_samples:
        prompts.extend(base_prompts)
    rng.shuffle(prompts)
    return prompts[:num_samples]


# --- NIM teacher ---

def get_nim_training_data(prompts, card="carl", max_tokens=256, system_prompt=None):
    client = httpx.Client(timeout=180.0)
    training_pairs = []

    for i, prompt in enumerate(prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = client.post(
                f"{PMS_BASE}/cards/{card}/v1/chat/completions",
                json={
                    "model": card,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            completion = msg.get("content") or msg.get("reasoning_content") or ""
            if completion:
                training_pairs.append({
                    "prompt": prompt,
                    "completion": completion.strip(),
                    "system": system_prompt or "",
                })
        except Exception as e:
            if i < 3:
                print(f"  Warning: NIM call {i} failed: {e}")

        if (i + 1) % 25 == 0:
            ok = sum(1 for t in training_pairs if t["completion"])
            print(f"  NIM: {i+1}/{len(prompts)} done ({ok} good)")

    ok = sum(1 for t in training_pairs if t["completion"])
    print(f"Got {ok}/{len(prompts)} completions from NIM teacher")
    return training_pairs


# --- Tokenization ---

def tokenize_training_data(pairs, tokenizer, max_length=256):
    samples = []
    for pair in pairs:
        full_text = f"{pair['prompt']}\n\n{pair['completion']}"
        prompt_text = f"{pair['prompt']}\n\n"

        full_tokens = tokenizer(
            full_text, truncation=True, max_length=max_length,
            return_tensors="pt", padding=False,
        )
        prompt_tokens = tokenizer(
            prompt_text, truncation=True, max_length=max_length,
            return_tensors="pt", padding=False,
        )

        seq_len = full_tokens["input_ids"].shape[1]
        prompt_len = prompt_tokens["input_ids"].shape[1]

        if seq_len < prompt_len + 8:
            continue

        labels = full_tokens["input_ids"].clone()
        labels[0, :prompt_len] = -100

        samples.append({
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        })

    print(f"Tokenized {len(samples)} training samples (min completion 8 tokens)")
    return samples


# --- Training ---

def train_nim_lora(
    model_path: str,
    output_dir: str,
    tensor_types_path: str = None,
    nim_card: str = "carl",
    domain: str = "repair",
    system_prompt: str = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 1,
    num_samples: int = 256,
    max_length: int = 256,
    nim_max_tokens: int = 256,
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Osmosis Repair LoRA Training (NIM Teacher)")
    print("=" * 60)
    print(f"Model:      {model_path}")
    print(f"NIM card:   {nim_card}")
    print(f"Domain:     {domain}")
    print(f"LoRA:       r={lora_r}, alpha={lora_alpha}, lr={lr}")
    print(f"Training:   {num_samples} samples, max_length={max_length}, {epochs} epochs")
    if tensor_types_path:
        print(f"Quant map:  {tensor_types_path}")

    # Phase 1: Generate training data from NIM teacher (zero GPU)
    print(f"\n--- Phase 1: Generating training data from NIM ({nim_card}) ---")

    cache_path = output_path / "training_data.json"
    if cache_path.exists():
        print(f"Loading cached training data from {cache_path}")
        with open(cache_path) as f:
            pairs = json.load(f)
        print(f"  {len(pairs)} cached pairs")
    else:
        if domain == "repair":
            prompts = load_wikitext_prompts(num_samples)
            nim_system = system_prompt or "Continue this text naturally and coherently."
        elif domain == "vadugwi":
            prompts = load_domain_prompts("vadugwi", num_samples)
            nim_system = system_prompt or (
                "You are an emotionally intelligent AI with deep understanding of human "
                "emotions. Respond with genuine empathy, emotional awareness, and nuance. "
                "Acknowledge the full complexity of feelings. Never dismiss or minimize. "
                "Show that you truly understand the emotional landscape of what's being shared."
            )
        else:
            prompts_file = Path(domain)
            if prompts_file.exists():
                prompts = [l.strip() for l in prompts_file.read_text().splitlines() if l.strip()][:num_samples]
                print(f"Loaded {len(prompts)} prompts from {domain}")
                nim_system = system_prompt or "Respond helpfully and thoroughly."
            else:
                raise ValueError(f"Unknown domain '{domain}' and not a file path")

        pairs = get_nim_training_data(prompts, card=nim_card, max_tokens=nim_max_tokens,
                                       system_prompt=nim_system)
        with open(cache_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Cached training data to {cache_path}")

    # Phase 2: Tokenize
    print(f"\n--- Phase 2: Tokenizing ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = tokenize_training_data(pairs, tokenizer, max_length=max_length)
    if len(samples) < 10:
        print(f"ERROR: Only {len(samples)} usable samples. Need at least 10.")
        return None

    # Phase 3: Load model + apply simulated quantization + attach LoRA
    print(f"\n--- Phase 3: Loading student model ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    if tensor_types_path:
        print(f"Applying simulated quantization from {tensor_types_path}")
        apply_simulated_quantization(model, tensor_types_path)

    # Freeze base model, attach LoRA
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.2f}%)")

    gc.collect()
    torch.cuda.empty_cache()

    # Phase 4: Train
    print(f"\n--- Phase 4: Training ---")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    num_steps = epochs * math.ceil(len(samples) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    print(f"Steps: {num_steps} ({epochs} epochs x {math.ceil(len(samples) / batch_size)} batches)")
    print("=" * 60)

    model.train()
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
                input_ids = sample["input_ids"].to(model.device)
                attention_mask = sample["attention_mask"].to(model.device)
                labels = sample["labels"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                if torch.isnan(loss):
                    continue
                batch_loss += loss

            if batch_loss == 0.0:
                global_step += 1
                continue

            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_val = batch_loss.item()
            if not math.isnan(loss_val):
                epoch_loss += loss_val
                epoch_steps += 1
            global_step += 1

            if global_step % 20 == 0:
                avg = epoch_loss / max(epoch_steps, 1)
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (num_steps - global_step) / max(steps_per_sec, 0.01)
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  step {global_step}/{num_steps}  "
                      f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"vram={vram:.1f}GB  ETA={eta / 60:.1f}min")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss and not math.isnan(avg_loss):
            best_loss = avg_loss
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"LoRA saved to: {output_path}")

    lora_size = sum(f.stat().st_size for f in output_path.glob("*") if f.is_file())
    print(f"LoRA size: {lora_size / 1024 / 1024:.1f} MB")

    meta = {
        "domain": domain,
        "nim_card": nim_card,
        "model_path": model_path,
        "tensor_types": tensor_types_path,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "epochs": epochs,
        "num_samples": len(samples),
        "best_loss": best_loss,
        "training_time_minutes": elapsed / 60,
        "trainable_params": trainable,
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Osmosis repair LoRA trainer (NIM teacher)")
    parser.add_argument("--model", required=True, help="HF model path (for tokenizer + weights)")
    parser.add_argument("--output", required=True, help="Output directory for LoRA")
    parser.add_argument("--tensor-types", default=None,
                        help="Tensor type file from budget allocator (for simulated quant)")
    parser.add_argument("--nim-card", default="carl", help="PMS card for NIM teacher")
    parser.add_argument("--domain", default="repair",
                        help="Training domain: repair, vadugwi, or path to prompts file")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--nim-max-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    train_nim_lora(
        args.model, args.output,
        tensor_types_path=args.tensor_types,
        nim_card=args.nim_card,
        domain=args.domain,
        system_prompt=args.system_prompt,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, num_samples=args.num_samples,
        max_length=args.max_length, nim_max_tokens=args.nim_max_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
