"""
export_dead_block_gguf.py -- Export dead-block checkpoint as a 38-layer GGUF.

The two inserted blocks (18 and 32) are complete standard Qwen2 decoder blocks:
  - Attention weights/biases: all zeros (identity pass-through since attn contributes 0)
  - attn_norm.weight (input_layernorm): copied from the corresponding source layer
  - ffn_norm.weight  <- trained norm2.weight
  - ffn_gate.weight  <- trained gate_proj.weight
  - ffn_up.weight    <- trained up_proj.weight
  - ffn_down.weight  <- trained down_proj.weight (rows 0..1535 remain zero per subspace constraint)

Layer remap (same as surgery.py / unroll_vanilla_gguf.py):
  old <= 17  -> keep
  18..30     -> old + 1  (shift up to make room for inserted block 18)
  >= 31      -> old + 2  (shift up to make room for inserted block 32)
  new block 18 gets ckpt['l17'], new block 32 gets ckpt['l30']

Usage:
    python export_dead_block_gguf.py [--input ...] [--ckpt ...] [--output ...]
    python export_dead_block_gguf.py --dry-run   # metadata planning only, no tensor writes
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import gguf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_new_idx(old_idx: int) -> int:
    if old_idx <= 17:
        return old_idx
    if old_idx <= 30:
        return old_idx + 1
    return old_idx + 2


def _tensor_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, cast to float32, convert to numpy."""
    return t.detach().cpu().float().numpy()


def _maybe_transpose(arr: np.ndarray, reader_tensor) -> np.ndarray:
    """
    GGUF stores weight matrices as (out_features, in_features) in row-major,
    but GGUFReader exposes shape as (in_features, out_features) — i.e. transposed.
    Match surgery.py / unroll_vanilla_gguf.py: if the numpy shape doesn't equal
    tensor.shape[::-1], transpose.
    """
    if arr.ndim == 2 and list(arr.shape) != list(reader_tensor.shape[::-1]):
        return arr.T
    return arr


def _copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter,
                   bk_key: str, new_blocks: int) -> None:
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
                # field.data contains INDICES into field.parts (not the values themselves).
                # Extract actual scalar values via field.parts[idx][0] for each idx.
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
# Zero-block builder
# ---------------------------------------------------------------------------

# Map from GGUF tensor suffix -> role tag used below
ATTN_ZERO_SUFFIXES = {
    "attn_q.weight":      "attn_q_w",
    "attn_q.bias":        "attn_q_b",
    "attn_k.weight":      "attn_k_w",
    "attn_k.bias":        "attn_k_b",
    "attn_v.weight":      "attn_v_w",
    "attn_v.bias":        "attn_v_b",
    "attn_output.weight": "attn_o_w",
}

# Map from GGUF tensor suffix -> which checkpoint key provides the data
# Keys are the checkpoint state-dict keys in DeadBlockWrapper
CKPT_KEY_MAP = {
    "ffn_norm.weight":  "norm2.weight",
    "ffn_gate.weight":  "gate_proj.weight",
    "ffn_up.weight":    "up_proj.weight",
    "ffn_down.weight":  "down_proj.weight",
}


def _build_inserted_block_tensors(
    block_new_idx: int,
    source_old_idx: int,
    ckpt_layer: dict,
    reader_tensors_by_suffix: dict,
    dry_run: bool,
) -> list:
    """
    Return a list of (name, data_or_none, raw_shape, raw_dtype) for the inserted block.

    data_or_none is None only in dry-run mode (we still record shape/dtype).
    reader_tensors_by_suffix: {suffix: ReaderTensor} for source_old_idx
    """
    entries = []
    prefix = f"blk.{block_new_idx}"

    # ---- attention norm: copy from source layer ----
    suffix = "attn_norm.weight"
    src = reader_tensors_by_suffix.get(suffix)
    if src is None:
        raise RuntimeError(
            f"Source layer {source_old_idx} missing {suffix} — cannot build inserted block."
        )
    if dry_run:
        entries.append((f"{prefix}.{suffix}", None, src.shape, src.tensor_type))
    else:
        entries.append((f"{prefix}.{suffix}", src.data.copy(), src.shape, src.tensor_type))

    # ---- zero attention tensors ----
    for suf in ATTN_ZERO_SUFFIXES:
        src = reader_tensors_by_suffix.get(suf)
        if src is None:
            # bias tensors may not exist in some variants; skip gracefully
            continue
        zero_arr = np.zeros(src.data.shape, dtype=src.data.dtype)
        if dry_run:
            entries.append((f"{prefix}.{suf}", None, src.shape, src.tensor_type))
        else:
            entries.append((f"{prefix}.{suf}", zero_arr, src.shape, src.tensor_type))

    # ---- trained FFN tensors ----
    for gguf_suf, ckpt_key in CKPT_KEY_MAP.items():
        if ckpt_key not in ckpt_layer:
            raise KeyError(
                f"Checkpoint layer missing '{ckpt_key}'. "
                f"Available keys: {list(ckpt_layer.keys())}"
            )
        arr = _tensor_numpy(ckpt_layer[ckpt_key])

        # Reference source tensor for shape / dtype metadata
        # Use the corresponding source suffix (gate/up/down/ffn_norm)
        src_suf_map = {
            "ffn_norm.weight": "ffn_norm.weight",
            "ffn_gate.weight": "ffn_gate.weight",
            "ffn_up.weight":   "ffn_up.weight",
            "ffn_down.weight": "ffn_down.weight",
        }
        src = reader_tensors_by_suffix.get(src_suf_map[gguf_suf])
        if src is None:
            raise RuntimeError(
                f"Source layer {source_old_idx} missing '{gguf_suf}' — need it for shape/dtype."
            )

        arr = _maybe_transpose(arr, src)

        if dry_run:
            entries.append((f"{prefix}.{gguf_suf}", None, src.shape, src.tensor_type))
        else:
            entries.append((f"{prefix}.{gguf_suf}", arr, src.shape, src.tensor_type))

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
    new_blocks = orig_blocks + 2
    print(f"[*] Architecture: {arch}, original blocks: {orig_blocks} -> {new_blocks}")

    # Load checkpoint (CPU only)
    print(f"[*] Loading checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_l17 = {k: v for k, v in ckpt["l17"].items() if not k.startswith("base_layer.")}
    ckpt_l30 = {k: v for k, v in ckpt["l30"].items() if not k.startswith("base_layer.")}
    print(f"  l17 trainable keys: {sorted(ckpt_l17.keys())}")
    print(f"  l30 trainable keys: {sorted(ckpt_l30.keys())}")

    inserted = {
        18: (17, ckpt_l17),
        32: (31, ckpt_l30),
    }
    source_indices = {17, 31}

    src_tensors: dict[int, dict[str, object]] = {idx: {} for idx in source_indices}
    for tensor in reader.tensors:
        if not tensor.name.startswith("blk."):
            continue
        parts = tensor.name.split(".")
        old_idx = int(parts[1])
        if old_idx in source_indices:
            suffix = ".".join(parts[2:])
            src_tensors[old_idx][suffix] = tensor

    print(f"[*] Source layer 17 tensors ({len(src_tensors[17])}): {sorted(src_tensors[17].keys())}")
    print(f"[*] Source layer 31 tensors ({len(src_tensors[31])}): {sorted(src_tensors[31].keys())}")

    orig_tensor_count = len(reader.tensors)
    tensors_per_new_block = 1 + sum(
        1 for s in ATTN_ZERO_SUFFIXES if s in src_tensors[17]
    ) + len(CKPT_KEY_MAP)
    expected_total = orig_tensor_count + 2 * tensors_per_new_block
    print(f"[*] Tensor count: {orig_tensor_count} original + {2 * tensors_per_new_block} inserted = {expected_total} expected")

    if dry_run:
        print("\n[DRY RUN] Metadata/remap planning only — no file written.\n")
        print(f"{'NEW IDX':>8}  {'OLD IDX':>8}  {'TENSOR NAME'}")
        print("-" * 70)

    # -----------------------------------------------------------------------
    # Streaming export strategy (avoids buffering all tensor data in RAM):
    #
    # Pass 1 — collect the ordered emit list: each entry is either a
    #   ("reader", reader_tensor, new_name)  — copy from source GGUF
    #   ("zeros",  shape, dtype)             — zero attention tensor
    #   ("numpy",  np.ndarray, shape, dtype) — trained FFN weight (from ckpt)
    #
    # Pass 1 also calls add_tensor_info on the GGUFWriter so the header
    # records the correct tensor count, names, shapes, and offsets — but
    # no tensor bytes are buffered in Python memory.
    #
    # After write_header + write_kv + write_ti, we open the output file in
    # append mode and stream each tensor's raw bytes directly, one at a time.
    # Peak extra RAM = one tensor at a time (~600 MB for token_embd.weight).
    # -----------------------------------------------------------------------

    # Ordered list of (kind, ...) tuples — built during pass 1
    emit_plan: list = []
    written_names: set = set()

    writer = gguf.GGUFWriter(gguf_out, arch) if not dry_run else None

    if not dry_run:
        print("[*] Copying metadata...")
        _copy_metadata(reader, writer, bk_key, new_blocks)

    def _plan_emit(name, kind, payload, raw_shape, raw_dtype, nbytes):
        """Record one tensor in emit_plan and register metadata with writer."""
        if name in written_names:
            print(f"  [!] Duplicate tensor skipped: {name}")
            return
        written_names.add(name)
        if dry_run:
            shape_str = "x".join(str(d) for d in raw_shape)
            print(f"  {name}  [{shape_str}]  {raw_dtype.name if hasattr(raw_dtype, 'name') else raw_dtype}")
        else:
            # Register metadata only — no data copy
            writer.add_tensor_info(name, raw_shape, np.dtype("float16"), nbytes,
                                   raw_dtype=raw_dtype)
        emit_plan.append((kind, name, payload, raw_shape, raw_dtype, nbytes))

    inserted_emitted: set = set()

    for tensor in reader.tensors:
        name = tensor.name

        if not name.startswith("blk."):
            # raw_shape must be tensor.shape[::-1]: write_ti_data_to_file reverses
            # the shape before writing, so passing the already-reversed reader shape
            # causes a double-reversal. Pass the reversed form so it round-trips correctly.
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

        for new_block_idx, (source_old, ckpt_layer) in inserted.items():
            if old_idx == source_old and new_block_idx not in inserted_emitted:
                if suffix == "ffn_down.weight":
                    print(f"[*] Injecting inserted block {new_block_idx} (from source layer {source_old})...")
                    block_entries = _build_inserted_block_tensors(
                        new_block_idx, source_old, ckpt_layer,
                        src_tensors[source_old], dry_run=dry_run,
                    )
                    for entry_name, entry_data, entry_shape, entry_dtype in block_entries:
                        if entry_data is None:
                            # dry-run: no actual data
                            src_ref = src_tensors[source_old].get(
                                ".".join(entry_name.split(".")[2:]))
                            nbytes = src_ref.data.nbytes if src_ref else 0
                            # entry_shape comes from src.shape (reader shape); must reverse
                            # so write_ti_data_to_file round-trips it correctly.
                            _plan_emit(entry_name, "zeros", entry_shape,
                                       entry_shape[::-1], entry_dtype, nbytes)
                        else:
                            # Cast array to the dtype matching raw_dtype so that
                            # nbytes == actual bytes we will stream (F16 weights
                            # come out of _tensor_numpy as F32; zero arrays from
                            # np.zeros already match because they share the source
                            # tensor's dtype).
                            _DTYPE_MAP = {
                                gguf.GGMLQuantizationType.F16: np.float16,
                                gguf.GGMLQuantizationType.F32: np.float32,
                                gguf.GGMLQuantizationType.BF16: np.float32,  # store as F32 bytes
                            }
                            target_np_dtype = _DTYPE_MAP.get(entry_dtype)
                            if target_np_dtype is not None and entry_data.dtype != target_np_dtype:
                                entry_data = entry_data.astype(target_np_dtype)
                            if np.all(entry_data == 0):
                                _plan_emit(entry_name, "zeros", entry_shape,
                                           entry_shape[::-1], entry_dtype, entry_data.nbytes)
                            else:
                                _plan_emit(entry_name, "numpy", entry_data,
                                           entry_shape[::-1], entry_dtype, entry_data.nbytes)
                    inserted_emitted.add(new_block_idx)

    for nb in inserted:
        if nb not in inserted_emitted:
            print(f"  [!] WARNING: inserted block {nb} was never emitted — trigger tensor may be missing.")

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
    # Flush and get the current file position (end of tensor-info section)
    assert writer.fout is not None
    fout = writer.fout[0]
    fout.flush()

    ALIGN = writer.data_alignment  # 32

    # Write alignment padding between tensor-info and tensor data
    pos = fout.tell()
    pad = (ALIGN - (pos % ALIGN)) % ALIGN
    if pad:
        fout.write(b"\x00" * pad)

    print(f"[*] Streaming tensor data ({written_count} tensors)...")
    for i, (kind, name, payload, raw_shape, raw_dtype, nbytes) in enumerate(emit_plan):
        if (i + 1) % 50 == 0 or (i + 1) == written_count:
            print(f"  [{i+1}/{written_count}] {name}")

        if kind == "reader":
            # payload is a ReaderTensor; .data is a memory-mapped numpy array
            raw = payload.data.tobytes()
        elif kind == "zeros":
            raw = b"\x00" * nbytes
        elif kind == "numpy":
            raw = payload.tobytes()
        else:
            raise RuntimeError(f"Unknown emit kind: {kind!r}")

        assert len(raw) == nbytes, f"{name}: expected {nbytes} bytes, got {len(raw)}"
        fout.write(raw)
        # Pad to alignment boundary
        pad = (ALIGN - (nbytes % ALIGN)) % ALIGN
        if pad:
            fout.write(b"\x00" * pad)

    fout.flush()
    fout.close()
    # Mark writer closed without it trying to re-write tensors
    writer.state = gguf.WriterState.WEIGHTS
    print(f"[+] Written: {gguf_out}")

    # ---- Post-write correctness checks ----
    _verify(gguf_out, orig_tensor_count, tensors_per_new_block)


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def _verify(gguf_out: str, orig_tensor_count: int, tensors_per_new_block: int) -> None:
    print("\n[*] Running post-write correctness checks...")
    reader = gguf.GGUFReader(gguf_out)

    results = {}

    # 1. block_count == 38
    for k in reader.fields:
        if k.endswith(".block_count"):
            bc = int(reader.fields[k].parts[-1][0])
            results["block_count==38"] = (bc == 38, f"got {bc}")
            break
    else:
        results["block_count==38"] = (False, "block_count field not found")

    # 2. tensor count == original + 2 full blocks
    expected_total = orig_tensor_count + 2 * tensors_per_new_block
    actual_total = len(reader.tensors)
    results[f"tensor_count=={expected_total}"] = (
        actual_total == expected_total,
        f"got {actual_total}",
    )

    # 3. Zero tensors in inserted blocks are actually zero
    attn_zero_checks = []
    for blk_idx in (18, 32):
        for suf in ATTN_ZERO_SUFFIXES:
            tname = f"blk.{blk_idx}.{suf}"
            for t in reader.tensors:
                if t.name == tname:
                    max_abs = float(np.abs(t.data).max()) if t.data.size > 0 else 0.0
                    attn_zero_checks.append((tname, max_abs))
                    break
    all_zero = all(v == 0.0 for _, v in attn_zero_checks)
    nonzero_names = [n for n, v in attn_zero_checks if v != 0.0]
    results["zero_attn_tensors"] = (
        all_zero,
        "OK" if all_zero else f"non-zero: {nonzero_names}",
    )

    # 4. ffn_down rows 0..1535 are zero in inserted blocks
    subspace_ok = True
    subspace_details = []
    for blk_idx in (18, 32):
        tname = f"blk.{blk_idx}.ffn_down.weight"
        for t in reader.tensors:
            if t.name == tname:
                # GGUFReader always exposes arr in PyTorch layout (shape field is
                # GGUF-transposed; reader transposes back).  For down_proj:
                #   shape field = [intermediate, hidden]  (e.g. [11008, 2048])
                #   t.data.shape = (hidden, intermediate)  (e.g. (2048, 11008))
                # Frozen rows 0..1535 in PyTorch = arr[0:1536, :] (first 1536 rows).
                arr = t.data  # shape: (hidden_size, intermediate_size) or flat
                if arr.ndim == 2:
                    frozen_slice = arr[:1536, :]
                else:
                    # flat; skip structured check
                    frozen_slice = None

                if frozen_slice is not None:
                    max_frozen = float(np.abs(frozen_slice).max()) if frozen_slice.size > 0 else 0.0
                    ok = max_frozen == 0.0
                    subspace_ok = subspace_ok and ok
                    subspace_details.append(f"blk.{blk_idx}: frozen_rows_max={max_frozen}")
                break
    results["ffn_down_frozen_rows_zero"] = (
        subspace_ok,
        "; ".join(subspace_details) if subspace_details else "no ffn_down tensors found",
    )

    # 5. All non-inserted tensors survived remap (spot-check: blk.0, blk.17, blk.19, blk.37)
    tensor_names = {t.name for t in reader.tensors}
    spot = [
        "blk.0.attn_norm.weight",
        "blk.17.ffn_down.weight",
        "blk.19.ffn_down.weight",   # was old 18 -> shifted to 19
        "blk.37.ffn_down.weight",   # was old 35 -> shifted to 37
    ]
    missing = [n for n in spot if n not in tensor_names]
    results["spot_check_remap"] = (len(missing) == 0, f"missing: {missing}" if missing else "OK")

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
        print("[!] Some checks FAILED — review output above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export dead-block checkpoint as a 38-layer Qwen2 GGUF."
    )
    parser.add_argument(
        "--input", default="qwen2.5-3b-brainloop.gguf",
        help="Source F16 GGUF (default: qwen2.5-3b-brainloop.gguf)",
    )
    parser.add_argument(
        "--ckpt", default="checkpoints-deadblock-13k/dead_blocks.pt",
        help="Dead-block checkpoint (default: checkpoints-deadblock-13k/dead_blocks.pt)",
    )
    parser.add_argument(
        "--output", default="cerebellum-deadblock-python.gguf",
        help="Output GGUF path (default: cerebellum-deadblock-python.gguf)",
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
        # Dry-run: no checkpoint needed — use dummy ckpt_layer filled with zeros
        # to allow shape planning from the GGUF alone.
        print(f"[*] DRY RUN mode — reading GGUF metadata only, no tensor data written.")
        print(f"[*] Reading {args.input}...")
        reader = gguf.GGUFReader(args.input)

        arch_field = reader.fields.get("general.architecture")
        arch = bytes(arch_field.parts[-1]).decode("utf-8").strip("\x00") if arch_field else "qwen2"
        bk_key = f"{arch}.block_count"
        if bk_key not in reader.fields:
            for k in reader.fields:
                if k.endswith(".block_count"):
                    bk_key = k
                    break
        orig_blocks = int(reader.fields[bk_key].parts[-1][0])
        new_blocks = orig_blocks + 2
        print(f"[*] arch={arch}, orig_blocks={orig_blocks} -> new_blocks={new_blocks}")

        # Collect source tensors
        source_indices = {17, 31}
        src_tensors: dict[int, dict] = {idx: {} for idx in source_indices}
        for tensor in reader.tensors:
            if not tensor.name.startswith("blk."):
                continue
            parts = tensor.name.split(".")
            old_idx = int(parts[1])
            if old_idx in source_indices:
                suffix = ".".join(parts[2:])
                src_tensors[old_idx][suffix] = tensor

        # Build dummy ckpt layers (zero tensors matching source shapes)
        def _dummy_ckpt(src_layer_tensors: dict) -> dict:
            d = {}
            shape_map = {
                "norm2.weight":    src_layer_tensors.get("ffn_norm.weight"),
                "gate_proj.weight": src_layer_tensors.get("ffn_gate.weight"),
                "up_proj.weight":   src_layer_tensors.get("ffn_up.weight"),
                "down_proj.weight": src_layer_tensors.get("ffn_down.weight"),
            }
            for ck, src in shape_map.items():
                if src is not None:
                    # shape[::-1] is (out, in) = PyTorch weight shape
                    d[ck] = torch.zeros(list(src.shape[::-1]), dtype=torch.float32)
            return d

        dummy_l17 = _dummy_ckpt(src_tensors[17])
        dummy_l30 = _dummy_ckpt(src_tensors[31])

        inserted = {18: (17, dummy_l17), 32: (31, dummy_l30)}

        # Print remap plan
        print(f"\n{'':=<70}")
        print(f"REMAP PLAN  ({orig_blocks} original tensors -> {new_blocks} blocks)")
        print(f"{'':=<70}")
        print(f"{'NEW NAME':<45}  {'SOURCE'}")
        print(f"{'-'*45}  {'-'*24}")

        inserted_emitted = set()
        plan_count = 0

        for tensor in reader.tensors:
            name = tensor.name
            if not name.startswith("blk."):
                shape_str = "x".join(str(d) for d in tensor.shape)
                print(f"  {name:<43}  [kept, {shape_str}]")
                plan_count += 1
                continue

            parts = name.split(".")
            old_idx = int(parts[1])
            new_idx = get_new_idx(old_idx)
            new_parts = list(parts)
            new_parts[1] = str(new_idx)
            new_name = ".".join(new_parts)
            suffix = ".".join(parts[2:])
            shape_str = "x".join(str(d) for d in tensor.shape)
            print(f"  {new_name:<43}  [blk.{old_idx}.{suffix}, {shape_str}]")
            plan_count += 1

            for new_block_idx, (source_old, ckpt_layer) in inserted.items():
                if old_idx == source_old and suffix == "ffn_down.weight" and new_block_idx not in inserted_emitted:
                    block_entries = _build_inserted_block_tensors(
                        new_block_idx, source_old, ckpt_layer,
                        src_tensors[source_old], dry_run=True,
                    )
                    for entry_name, _, entry_shape, entry_dtype in block_entries:
                        shape_str2 = "x".join(str(d) for d in entry_shape)
                        role = "zero-attn" if any(s in entry_name for s in ("attn_q", "attn_k", "attn_v", "attn_output")) else \
                               "copy-norm" if "attn_norm" in entry_name else "trained"
                        print(f"  {entry_name:<43}  [INSERTED blk.{new_block_idx}, {role}, {shape_str2}]")
                        plan_count += 1
                    inserted_emitted.add(new_block_idx)

        print(f"\n[DRY RUN] Total tensors planned: {plan_count}")
        print(f"[DRY RUN] block_count: {new_blocks}")
        print(f"[DRY RUN] Output would be written to: {args.output}")
        return

    export(args.input, args.ckpt, args.output)


if __name__ == "__main__":
    main()
