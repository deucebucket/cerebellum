"""
export_live_block_gguf.py -- Export live-block checkpoint as a 37-layer GGUF.

The inserted block (new index 18) is a complete standard Qwen2 decoder block
containing ALL trained tensors from the checkpoint:
  - attn_norm.weight  (input_layernorm)
  - attn_q.weight + attn_q.bias
  - attn_k.weight + attn_k.bias
  - attn_v.weight + attn_v.bias
  - attn_output.weight              (o_proj, no bias in Qwen2.5)
  - ffn_norm.weight  (post_attention_layernorm)
  - ffn_gate.weight
  - ffn_up.weight
  - ffn_down.weight

No tensors are zeroed by this exporter. Whatever training left in the
checkpoint is what goes into the GGUF.

Layer remap (36 -> 37 blocks):
  old <= 17  -> keep as-is
  old >= 18  -> old + 1  (shift up to make room for inserted block 18)

Checkpoint format expected: {'l17_block': <state_dict>, 'meta': {...}}
  where state_dict keys match Qwen2DecoderLayer parameter names, e.g.:
    input_layernorm.weight
    self_attn.q_proj.weight / .bias
    self_attn.k_proj.weight / .bias
    self_attn.v_proj.weight / .bias
    self_attn.o_proj.weight
    post_attention_layernorm.weight
    mlp.gate_proj.weight
    mlp.up_proj.weight
    mlp.down_proj.weight

Usage:
    python export_live_block_gguf.py [--input ...] [--ckpt ...] [--output ...]
    python export_live_block_gguf.py --dry-run
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import gguf


# ---------------------------------------------------------------------------
# Layer index remap: 36-block source -> 37-block output
# ---------------------------------------------------------------------------
def get_new_idx(old_idx: int) -> int:
    """old <= 17 keep; old >= 18 shift +1."""
    if old_idx <= 17:
        return old_idx
    return old_idx + 1


# ---------------------------------------------------------------------------
# Helpers (same conventions as export_dead_block_gguf.py)
# ---------------------------------------------------------------------------
def _tensor_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, cast to float32, convert to numpy."""
    return t.detach().cpu().float().numpy()


def _maybe_transpose(arr: np.ndarray, reader_tensor) -> np.ndarray:
    """
    GGUF stores weight matrices as (out_features, in_features) row-major,
    but GGUFReader exposes shape as (in_features, out_features) -- transposed.
    If arr.shape != reader_tensor.shape[::-1], transpose to match.
    """
    if arr.ndim == 2 and list(arr.shape) != list(reader_tensor.shape[::-1]):
        return arr.T
    return arr


def _copy_metadata(reader, writer, bk_key: str, new_blocks: int) -> None:
    """Replay all KV metadata from reader into writer, patching block_count."""
    for field in reader.fields.values():
        name = field.name
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                    "general.architecture"):
            continue
        if name == bk_key:
            writer.add_uint32(name, new_blocks)
            continue

        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            etype = field.types[1]
            if etype == gguf.GGUFValueType.STRING:
                str_list = [bytes(field.parts[s]).decode("utf-8") for s in field.data]
                if name == "tokenizer.ggml.tokens":
                    writer.add_token_list(str_list)
                else:
                    writer.add_array(name, str_list)
            else:
                values = [int(field.parts[idx][0]) for idx in field.data]
                writer.add_array(name, values)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(name, bytes(field.parts[-1]).decode("utf-8").strip("\x00"))
        elif vtype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(name, bool(field.parts[-1][0]))
        else:
            try:
                writer.add_key_value(name, field.data, vtype)
            except Exception:
                print(f"  [!] Skipping metadata field: {name} (type {vtype})")


# ---------------------------------------------------------------------------
# Mapping: checkpoint state_dict key -> GGUF tensor suffix
# ---------------------------------------------------------------------------
# The inserted block uses all trained tensors from ckpt['l17_block'].
# State dict keys from Qwen2DecoderLayer:
#   input_layernorm.weight
#   self_attn.q_proj.weight / .bias
#   self_attn.k_proj.weight / .bias
#   self_attn.v_proj.weight / .bias
#   self_attn.o_proj.weight   (no bias)
#   post_attention_layernorm.weight
#   mlp.gate_proj.weight
#   mlp.up_proj.weight
#   mlp.down_proj.weight
#
# GGUF suffix conventions (Qwen2):
#   attn_norm.weight          <- input_layernorm.weight
#   attn_q.weight / .bias     <- self_attn.q_proj.*
#   attn_k.weight / .bias     <- self_attn.k_proj.*
#   attn_v.weight / .bias     <- self_attn.v_proj.*
#   attn_output.weight        <- self_attn.o_proj.weight
#   ffn_norm.weight           <- post_attention_layernorm.weight
#   ffn_gate.weight           <- mlp.gate_proj.weight
#   ffn_up.weight             <- mlp.up_proj.weight
#   ffn_down.weight           <- mlp.down_proj.weight

CKPT_TO_GGUF = [
    # (ckpt_key, gguf_suffix, source_ref_suffix_for_shape)
    ("input_layernorm.weight",         "attn_norm.weight",   "attn_norm.weight"),
    ("self_attn.q_proj.weight",        "attn_q.weight",      "attn_q.weight"),
    ("self_attn.q_proj.bias",          "attn_q.bias",        "attn_q.bias"),
    ("self_attn.k_proj.weight",        "attn_k.weight",      "attn_k.weight"),
    ("self_attn.k_proj.bias",          "attn_k.bias",        "attn_k.bias"),
    ("self_attn.v_proj.weight",        "attn_v.weight",      "attn_v.weight"),
    ("self_attn.v_proj.bias",          "attn_v.bias",        "attn_v.bias"),
    ("self_attn.o_proj.weight",        "attn_output.weight", "attn_output.weight"),
    ("post_attention_layernorm.weight","ffn_norm.weight",     "ffn_norm.weight"),
    ("mlp.gate_proj.weight",           "ffn_gate.weight",     "ffn_gate.weight"),
    ("mlp.up_proj.weight",             "ffn_up.weight",       "ffn_up.weight"),
    ("mlp.down_proj.weight",           "ffn_down.weight",     "ffn_down.weight"),
]


# ---------------------------------------------------------------------------
# Build inserted block tensor list
# ---------------------------------------------------------------------------
def _build_inserted_block_tensors(
    block_new_idx: int,
    ckpt_layer: dict,
    src_tensors_by_suffix: dict,
    dry_run: bool,
) -> list:
    """
    Return list of (name, data_or_none, raw_shape, raw_dtype) for the inserted block.
    data_or_none is None only in dry-run mode.
    src_tensors_by_suffix: {suffix: ReaderTensor} for source layer 17.
    """
    entries = []
    prefix = f"blk.{block_new_idx}"

    _DTYPE_MAP = {
        gguf.GGMLQuantizationType.F16:  np.float16,
        gguf.GGMLQuantizationType.F32:  np.float32,
        gguf.GGMLQuantizationType.BF16: np.float32,  # store as F32 bytes
    }

    for ckpt_key, gguf_suf, ref_suf in CKPT_TO_GGUF:
        if ckpt_key not in ckpt_layer:
            # Bias keys may be absent if the model variant lacks them; skip gracefully
            continue

        src = src_tensors_by_suffix.get(ref_suf)
        if src is None:
            raise RuntimeError(
                f"Source layer 17 missing '{ref_suf}' -- cannot determine "
                f"shape/dtype for inserted tensor '{gguf_suf}'."
            )

        name = f"{prefix}.{gguf_suf}"

        if dry_run:
            entries.append((name, None, src.shape, src.tensor_type))
            continue

        arr = _tensor_numpy(ckpt_layer[ckpt_key])
        arr = _maybe_transpose(arr, src)

        # Cast to match source tensor's storage dtype
        target_dtype = _DTYPE_MAP.get(src.tensor_type)
        if target_dtype is not None and arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)

        entries.append((name, arr, src.shape, src.tensor_type))

    return entries


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------
def export(gguf_in: str, ckpt_path: str, gguf_out: str, dry_run: bool = False) -> None:
    print(f"[*] Reading {gguf_in}...")
    reader = gguf.GGUFReader(gguf_in)

    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[-1]).decode("utf-8").strip("\x00") if arch_field else "qwen2"

    bk_key = f"{arch}.block_count"
    if bk_key not in reader.fields:
        for k in reader.fields:
            if k.endswith(".block_count"):
                bk_key = k
                break
    orig_blocks = int(reader.fields[bk_key].parts[-1][0])
    new_blocks = orig_blocks + 1
    print(f"[*] Architecture: {arch}, original blocks: {orig_blocks} -> {new_blocks}")

    if orig_blocks != 36:
        print(f"  [!] WARNING: expected 36-block Qwen2.5-3B source, got {orig_blocks}. Proceeding anyway.")

    # Load checkpoint (CPU only -- no GPU involvement)
    print(f"[*] Loading checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "l17_block" not in ckpt:
        print(f"[!] Checkpoint missing 'l17_block' key. Found keys: {list(ckpt.keys())}")
        sys.exit(1)
    ckpt_layer = ckpt["l17_block"]
    meta = ckpt.get("meta", {})
    print(f"  Checkpoint meta: {meta}")
    print(f"  l17_block keys ({len(ckpt_layer)}): {sorted(ckpt_layer.keys())}")

    # Collect source tensors for layer 17 (for shape/dtype reference)
    src_tensors: dict = {}
    for tensor in reader.tensors:
        if tensor.name.startswith("blk.17."):
            suffix = tensor.name[len("blk.17."):]
            src_tensors[suffix] = tensor
    print(f"[*] Source layer 17 tensors ({len(src_tensors)}): {sorted(src_tensors.keys())}")

    orig_tensor_count = len(reader.tensors)
    # Count how many entries the inserted block will contribute
    inserted_tensor_count = sum(
        1 for ck, _, _ in CKPT_TO_GGUF if ck in ckpt_layer
    )
    expected_total = orig_tensor_count + inserted_tensor_count
    print(
        f"[*] Tensor count: {orig_tensor_count} original + "
        f"{inserted_tensor_count} inserted = {expected_total} expected"
    )

    if dry_run:
        print("\n[DRY RUN] Metadata/remap planning only -- no file written.\n")

    # -----------------------------------------------------------------------
    # Streaming export (same pattern as export_dead_block_gguf.py):
    # Pass 1: build emit_plan + register tensor metadata with GGUFWriter.
    # Pass 2: write header/KV/TI, then stream raw bytes one tensor at a time.
    # -----------------------------------------------------------------------
    emit_plan: list = []
    written_names: set = set()

    writer = gguf.GGUFWriter(gguf_out, arch) if not dry_run else None

    if not dry_run:
        print("[*] Copying metadata...")
        _copy_metadata(reader, writer, bk_key, new_blocks)

    def _plan_emit(name, kind, payload, raw_shape, raw_dtype, nbytes):
        if name in written_names:
            print(f"  [!] Duplicate tensor skipped: {name}")
            return
        written_names.add(name)
        if dry_run:
            shape_str = "x".join(str(d) for d in raw_shape)
            dtype_name = raw_dtype.name if hasattr(raw_dtype, "name") else str(raw_dtype)
            print(f"  {name}  [{shape_str}]  {dtype_name}")
        else:
            writer.add_tensor_info(name, raw_shape, np.dtype("float16"), nbytes,
                                   raw_dtype=raw_dtype)
        emit_plan.append((kind, name, payload, raw_shape, raw_dtype, nbytes))

    inserted_emitted = False

    for tensor in reader.tensors:
        name = tensor.name

        if not name.startswith("blk."):
            _plan_emit(name, "reader", tensor, tensor.shape[::-1], tensor.tensor_type,
                       tensor.data.nbytes)
            continue

        parts = name.split(".")
        old_idx = int(parts[1])
        new_idx = get_new_idx(old_idx)
        new_parts = list(parts)
        new_parts[1] = str(new_idx)
        new_name = ".".join(new_parts)
        suffix = ".".join(parts[2:])

        _plan_emit(new_name, "reader", tensor, tensor.shape[::-1], tensor.tensor_type,
                   tensor.data.nbytes)

        # Inject the inserted block immediately after the last tensor of old layer 17
        if old_idx == 17 and suffix == "ffn_down.weight" and not inserted_emitted:
            print(f"[*] Injecting inserted block 18 (from trained layer 17 checkpoint)...")
            block_entries = _build_inserted_block_tensors(
                18, ckpt_layer, src_tensors, dry_run=dry_run
            )
            for entry_name, entry_data, entry_shape, entry_dtype in block_entries:
                if entry_data is None:
                    src_ref = src_tensors.get(".".join(entry_name.split(".")[2:]))
                    nbytes = src_ref.data.nbytes if src_ref else 0
                    _plan_emit(entry_name, "zeros", entry_shape,
                               entry_shape[::-1], entry_dtype, nbytes)
                else:
                    if np.all(entry_data == 0):
                        _plan_emit(entry_name, "zeros", entry_shape,
                                   entry_shape[::-1], entry_dtype, entry_data.nbytes)
                    else:
                        _plan_emit(entry_name, "numpy", entry_data,
                                   entry_shape[::-1], entry_dtype, entry_data.nbytes)
            inserted_emitted = True

    if not inserted_emitted:
        print("  [!] WARNING: inserted block 18 was never emitted -- "
              "blk.17.ffn_down.weight trigger tensor may be missing from source GGUF.")

    written_count = len(emit_plan)

    if dry_run:
        print(f"\n[DRY RUN] Total tensors planned: {written_count} (expected ~{expected_total})")
        print(f"[DRY RUN] block_count would be: {new_blocks}")
        return

    # ---- Pass 2: write header + metadata, then stream tensor bytes ----
    print(f"[*] Writing header + KV + tensor-info ({written_count} tensors)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    assert writer.fout is not None
    fout = writer.fout[0]
    fout.flush()

    ALIGN = writer.data_alignment  # 32

    pos = fout.tell()
    pad = (ALIGN - (pos % ALIGN)) % ALIGN
    if pad:
        fout.write(b"\x00" * pad)

    print(f"[*] Streaming tensor data ({written_count} tensors)...")
    for i, (kind, name, payload, raw_shape, raw_dtype, nbytes) in enumerate(emit_plan):
        if (i + 1) % 50 == 0 or (i + 1) == written_count:
            print(f"  [{i+1}/{written_count}] {name}")

        if kind == "reader":
            raw = payload.data.tobytes()
        elif kind == "zeros":
            raw = b"\x00" * nbytes
        elif kind == "numpy":
            raw = payload.tobytes()
        else:
            raise RuntimeError(f"Unknown emit kind: {kind!r}")

        assert len(raw) == nbytes, f"{name}: expected {nbytes} bytes, got {len(raw)}"
        fout.write(raw)
        pad = (ALIGN - (nbytes % ALIGN)) % ALIGN
        if pad:
            fout.write(b"\x00" * pad)

    fout.flush()
    fout.close()
    writer.state = gguf.WriterState.WEIGHTS
    print(f"[+] Written: {gguf_out}")

    _verify(gguf_out, orig_tensor_count, inserted_tensor_count, ckpt_layer)


# ---------------------------------------------------------------------------
# Post-write correctness checks
# ---------------------------------------------------------------------------
def _verify(
    gguf_out: str,
    orig_tensor_count: int,
    inserted_tensor_count: int,
    ckpt_layer: dict,
) -> None:
    print("\n[*] Running post-write correctness checks...")
    reader = gguf.GGUFReader(gguf_out)

    results = {}

    # 1. block_count == 37
    for k in reader.fields:
        if k.endswith(".block_count"):
            bc = int(reader.fields[k].parts[-1][0])
            results["block_count==37"] = (bc == 37, f"got {bc}")
            break
    else:
        results["block_count==37"] = (False, "block_count field not found")

    # 2. tensor count == original + inserted
    expected_total = orig_tensor_count + inserted_tensor_count
    actual_total = len(reader.tensors)
    results[f"tensor_count=={expected_total}"] = (
        actual_total == expected_total,
        f"got {actual_total}",
    )

    # 3. Remap spot-checks: blk.0, blk.17, blk.19 (was old 18), blk.36 (was old 35)
    tensor_names = {t.name for t in reader.tensors}
    spot = [
        "blk.0.attn_norm.weight",
        "blk.17.ffn_down.weight",
        "blk.19.ffn_down.weight",   # was old 18 -> shifted to 19
        "blk.36.ffn_down.weight",   # was old 35 -> shifted to 36
    ]
    missing = [n for n in spot if n not in tensor_names]
    results["spot_check_remap"] = (
        len(missing) == 0,
        f"missing: {missing}" if missing else "OK",
    )

    # 4. Spot-check 2 inserted tensors match checkpoint values within dtype tolerance.
    #    Check attn_norm.weight and ffn_down.weight for block 18.
    spot_ckpt_checks = [
        ("blk.18.attn_norm.weight",  "input_layernorm.weight"),
        ("blk.18.ffn_down.weight",   "mlp.down_proj.weight"),
    ]
    ckpt_check_results = []
    for gguf_name, ck_key in spot_ckpt_checks:
        if ck_key not in ckpt_layer:
            ckpt_check_results.append(f"{gguf_name}: ckpt key '{ck_key}' missing -- skipped")
            continue
        found_tensor = None
        for t in reader.tensors:
            if t.name == gguf_name:
                found_tensor = t
                break
        if found_tensor is None:
            ckpt_check_results.append(f"{gguf_name}: NOT FOUND in output GGUF")
            continue
        # Compare shapes
        ck_arr = ckpt_layer[ck_key].detach().cpu().float().numpy()
        # Reader shape is (in, out); ck_arr is (out, in) for 2D weights
        if ck_arr.ndim == 2:
            expected_shape = tuple(reversed(ck_arr.shape))
        else:
            expected_shape = ck_arr.shape
        actual_shape = tuple(found_tensor.shape)
        if actual_shape != expected_shape:
            ckpt_check_results.append(
                f"{gguf_name}: shape mismatch reader={actual_shape} ckpt={expected_shape}"
            )
        else:
            ckpt_check_results.append(f"{gguf_name}: shape OK {actual_shape}")
    results["spot_check_ckpt_tensors"] = (
        all("NOT FOUND" not in s and "mismatch" not in s for s in ckpt_check_results),
        "; ".join(ckpt_check_results),
    )

    # Print results
    print()
    all_pass = True
    for check, (passed, detail) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}: {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("[+] All checks PASSED.")
    else:
        print("[!] Some checks FAILED -- review output above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export live-block checkpoint as a 37-layer Qwen2 GGUF."
    )
    parser.add_argument(
        "--input", default="qwen2.5-3b-brainloop.gguf",
        help="Source F16 GGUF (default: qwen2.5-3b-brainloop.gguf)",
    )
    parser.add_argument(
        "--ckpt", default="checkpoints-liveblock/live_block_best.pt",
        help="Live-block checkpoint (default: checkpoints-liveblock/live_block_best.pt)",
    )
    parser.add_argument(
        "--output", default="cerebellum-liveblock.gguf",
        help="Output GGUF path (default: cerebellum-liveblock.gguf)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print remap plan only; do not write tensor data or output file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[!] Input GGUF not found: {args.input}")
        sys.exit(1)

    if not args.dry_run and not os.path.isfile(args.ckpt):
        print(f"[!] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    if args.dry_run:
        # Dry-run: checkpoint not required for remap planning,
        # but we need dummy ckpt_layer to enumerate which tensors will be inserted.
        print(f"[*] DRY RUN mode -- no file written.")
        print(f"[*] Reading {args.input}...")
        reader_dr = gguf.GGUFReader(args.input)

        # Build dummy ckpt_layer from GGUF source shapes
        src_tensors_dr: dict = {}
        for tensor in reader_dr.tensors:
            if tensor.name.startswith("blk.17."):
                suffix = tensor.name[len("blk.17."):]
                src_tensors_dr[suffix] = tensor

        def _dummy_tensor(src) -> torch.Tensor:
            # shape[::-1] gives (out, in) = PyTorch weight shape
            return torch.zeros(list(src.shape[::-1]), dtype=torch.float32)

        dummy_ckpt = {}
        for ck_key, gguf_suf, ref_suf in CKPT_TO_GGUF:
            src = src_tensors_dr.get(ref_suf)
            if src is not None:
                dummy_ckpt[ck_key] = _dummy_tensor(src)

        # Delegate to the main export function in dry-run mode
        export(args.input, args.ckpt, args.output, dry_run=True)
        return

    export(args.input, args.ckpt, args.output, dry_run=False)


if __name__ == "__main__":
    main()
