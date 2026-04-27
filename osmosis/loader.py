"""Loader for Osmosis-compressed models — mixed-precision inference."""
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


BLOCK_SIZE = 32


def _unpack_blocks(data: bytes, shape: list, bits: int) -> torch.Tensor:
    """Unpack block-wise quantized data: [f16 scale | packed data] per block."""
    num_elements = int(np.prod(shape))
    n_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    if bits == 1:
        data_bytes_per_block = BLOCK_SIZE // 8  # 4
    elif bits == 2:
        data_bytes_per_block = BLOCK_SIZE // 4  # 8
    else:
        data_bytes_per_block = BLOCK_SIZE // 2  # 16
    block_bytes = 2 + data_bytes_per_block  # 2 for f16 scale

    out = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16).astype(np.float32)[0]
        raw = np.frombuffer(data[offset + 2:offset + block_bytes], dtype=np.uint8)

        if bits == 1:
            vals = np.zeros(BLOCK_SIZE, dtype=np.float32)
            for bit in range(8):
                vals[bit::8] = (raw >> bit) & 1
            out[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = (vals * 2 - 1) * scale
        elif bits == 2:
            vals = np.zeros(BLOCK_SIZE, dtype=np.float32)
            for pos in range(4):
                vals[pos::4] = (raw >> (pos * 2)) & 0x3
            out[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = (vals - 1.5) * scale
        else:
            low = (raw & 0xF).astype(np.float32)
            high = (raw >> 4).astype(np.float32)
            vals = np.empty(BLOCK_SIZE, dtype=np.float32)
            vals[0::2] = low
            vals[1::2] = high
            out[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = (vals - 7.5) * scale

    return torch.tensor(out[:num_elements]).reshape(shape)


def load_osm_tensor(path: Path) -> torch.Tensor:
    with open(path, "rb") as f:
        first_byte = struct.unpack("<B", f.read(1))[0]

        if first_byte == 2:
            bits, ndims = struct.unpack("<BB", f.read(2))
            shape = list(struct.unpack(f"<{ndims}I", f.read(ndims * 4)))
            data = f.read()
            return _unpack_blocks(data, shape, bits)
        else:
            f.seek(0)
            header = f.read(13)
            bits, scale, rows, cols = struct.unpack("<BfII", header)
            data = f.read()
            shape = [rows, cols] if cols > 1 else [rows]
            return _unpack_blocks_v1(data, shape, bits, scale)


def _unpack_blocks_v1(data: bytes, shape: list, bits: int, scale: float) -> torch.Tensor:
    """Legacy v1 global-scale unpack."""
    num_elements = int(np.prod(shape))
    raw = np.frombuffer(data, dtype=np.uint8)
    if bits == 1:
        vals = np.zeros(len(raw) * 8, dtype=np.float32)
        for bit in range(8):
            vals[bit::8] = (raw >> bit) & 1
        vals = vals[:num_elements]
        return torch.tensor((vals * 2 - 1) * scale).reshape(shape)
    elif bits == 2:
        vals = np.zeros(len(raw) * 4, dtype=np.float32)
        for pos in range(4):
            vals[pos::4] = (raw >> (pos * 2)) & 0x3
        vals = vals[:num_elements]
        return torch.tensor((vals - 1.5) * scale).reshape(shape)
    else:
        low = (raw & 0xF).astype(np.float32)
        high = (raw >> 4).astype(np.float32)
        interleaved = np.empty(len(raw) * 2, dtype=np.float32)
        interleaved[0::2] = low
        interleaved[1::2] = high
        interleaved = interleaved[:num_elements]
        return torch.tensor((interleaved - 7.5) * scale).reshape(shape)


class OsmosisModel:
    """Load an Osmosis-compressed model and run inference."""

    def __init__(self, crush_dir: str, original_model: str,
                 device="cpu", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        crush_path = Path(crush_dir)

        with open(crush_path / "manifest.json") as f:
            self.manifest = json.load(f)

        print(f"Loading architecture from: {original_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            original_model, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            original_model, torch_dtype=dtype, device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        layers = self.manifest["layers"]
        patched = 0

        for key, info in layers.items():
            file_name = info["file"]
            bits = info["bits"]

            if bits == 16:
                st_path = crush_path / file_name
                with safe_open(str(st_path), framework="pt", device="cpu") as sf:
                    for tk in sf.keys():
                        tensor = sf.get_tensor(tk)
                        self._patch_weight(tk, tensor)
                        patched += 1
            else:
                osm_path = crush_path / file_name
                tensor = load_osm_tensor(osm_path)
                self._patch_weight(key, tensor)
                patched += 1

        print(f"Patched {patched} tensors")
        avg = self.manifest.get("average_bits", "?")
        ratio = self.manifest.get("compression", {}).get("ratio", "?")
        print(f"Average bits: {avg}, compression ratio: {ratio}x")

    def _patch_weight(self, key: str, tensor: torch.Tensor):
        # Strip architecture prefixes that don't exist on the causal LM variant
        for prefix in ("model.language_model.", "language_model."):
            if key.startswith(prefix):
                key = "model." + key[len(prefix):]
                break
        parts = key.split(".")
        module = self.model
        try:
            for part in parts[:-1]:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            param_name = parts[-1]
            param = getattr(module, param_name)
        except (AttributeError, IndexError, KeyError):
            return

        if param.shape != tensor.shape:
            return

        target_device = param.device if self.device == "auto" else self.device
        param.data = tensor.to(dtype=self.dtype, device=target_device)

    def generate(self, prompt: str, max_new_tokens=256, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **kwargs
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @torch.no_grad()
    def logits(self, prompt: str, max_length=128):
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        output = self.model(**tokens)
        return F.log_softmax(output.logits[0].float(), dim=-1)


@torch.no_grad()
def compare(original_path: str, crush_dir: str, prompts: list,
            max_length: int = 128, device: str = "cpu",
            dtype=torch.float16):
    """Compare original vs Osmosis-compressed model on same prompts."""
    print("Loading original model...")
    tokenizer = AutoTokenizer.from_pretrained(
        original_path, trust_remote_code=True
    )
    original = AutoModelForCausalLM.from_pretrained(
        original_path, torch_dtype=dtype, device_map=device,
        trust_remote_code=True,
    )
    original.eval()

    print("Loading crushed model...")
    crushed = OsmosisModel(crush_dir, original_path, device=device, dtype=dtype)

    print(f"\nComparing on {len(prompts)} prompts (max_length={max_length})")
    print("=" * 70)

    all_kl = []
    for i, prompt in enumerate(prompts):
        tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        orig_out = original(**tokens)
        orig_logp = F.log_softmax(orig_out.logits[0].float(), dim=-1)

        crush_out = crushed.model(**tokens)
        crush_logp = F.log_softmax(crush_out.logits[0].float(), dim=-1)

        kl = F.kl_div(
            crush_logp, torch.exp(orig_logp),
            reduction="batchmean", log_target=False
        ).item()
        all_kl.append(kl)

        orig_ppl = torch.exp(
            F.cross_entropy(
                orig_out.logits[0, :-1],
                tokens["input_ids"][0, 1:],
            )
        ).item()
        crush_ppl = torch.exp(
            F.cross_entropy(
                crush_out.logits[0, :-1],
                tokens["input_ids"][0, 1:],
            )
        ).item()

        print(f"\n[Prompt {i+1}] {prompt[:60]}...")
        print(f"  KL divergence:     {kl:.6f}")
        print(f"  Original PPL:      {orig_ppl:.2f}")
        print(f"  Crushed PPL:       {crush_ppl:.2f}")
        print(f"  PPL delta:         {crush_ppl - orig_ppl:+.2f}")

    print(f"\n{'=' * 70}")
    print(f"Mean KL:  {np.mean(all_kl):.6f}")
    print(f"Max KL:   {np.max(all_kl):.6f}")
    print(f"Min KL:   {np.min(all_kl):.6f}")
    return all_kl


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Osmosis model loader")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate text from crushed model")
    gen.add_argument("--crush-dir", required=True)
    gen.add_argument("--model", required=True, help="Original model path")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=256)

    cmp = sub.add_parser("compare", help="Compare original vs crushed")
    cmp.add_argument("--crush-dir", required=True)
    cmp.add_argument("--model", required=True, help="Original model path")
    cmp.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    if args.command == "generate":
        m = OsmosisModel(args.crush_dir, args.model)
        print(m.generate(args.prompt, max_new_tokens=args.max_tokens))
    elif args.command == "compare":
        prompts = [
            "The meaning of life is",
            "In a shocking finding, scientists discovered",
            "def fibonacci(n):",
            "The capital of France is",
        ]
        compare(args.model, args.crush_dir, prompts, args.max_length)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
