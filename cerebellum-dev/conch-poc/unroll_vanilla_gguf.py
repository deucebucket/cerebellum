import gguf
import torch
import os
import numpy as np

def unroll_cloned_gguf(gguf_in, ckpt_path, gguf_out):
    print(f"[*] Reading {gguf_in}...")
    reader = gguf.GGUFReader(gguf_in)
    
    ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[*] Loading trained parameters from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Get architecture
    arch_field = reader.fields.get("general.architecture")
    arch = bytes(arch_field.parts[-1]).decode('utf-8').strip('\x00')
    
    bk_key = "qwen2.block_count" if "qwen2.block_count" in reader.fields else "llama.block_count"
    orig_blocks = int(reader.fields[bk_key].parts[-1][0])
    new_blocks = orig_blocks + 2
    print(f"[*] Blocks: {orig_blocks} -> {new_blocks}")
    
    writer = gguf.GGUFWriter(gguf_out, arch)
    
    # 1. Defensively Copy Metadata
    print("[*] Cloning metadata...")
    for name, field in reader.fields.items():
        if name in ["GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"]:
            continue
        
        if name == bk_key:
            writer.add_uint32(name, new_blocks)
            continue

        try:
            # Re-add based on type
            vtype = field.types[0]
            val_parts = field.parts[-1]
            
            if vtype == gguf.GGUFValueType.UINT32:
                writer.add_uint32(name, int(val_parts[0]))
            elif vtype == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(name, float(val_parts[0]))
            elif vtype == gguf.GGUFValueType.STRING:
                writer.add_string(name, bytes(val_parts).decode('utf-8').strip('\x00'))
            elif vtype == gguf.GGUFValueType.BOOL:
                writer.add_bool(name, bool(val_parts[0]))
            elif vtype == gguf.GGUFValueType.ARRAY:
                # Array handling is the trickiest
                # We'll use the field.data if it's available and valid
                if name == "tokenizer.ggml.tokens":
                    tokens = [bytes(s).decode('utf-8').strip('\x00') for s in field.data]
                    writer.add_token_list(tokens)
                elif name == "tokenizer.ggml.token_type":
                    writer.add_array(name, field.data.tolist() if hasattr(field.data, 'tolist') else field.data)
                elif name == "tokenizer.ggml.merges":
                    merges = [bytes(s).decode('utf-8').strip('\x00') for s in field.data]
                    writer.add_array(name, merges)
                else:
                    writer.add_array(name, field.data)
            else:
                # Try raw copy as fallback
                writer.add_key_value(name, field.types, field.data)
        except Exception as e:
            print(f"  [!] Failed to copy KV '{name}': {e}")

    # 2. Layer Remapping & Physical Cloning
    print("[*] Remapping and CLONING tensors...")
    base_tensors = {t.name: t for t in reader.tensors}
    
    def get_new_idx(old_idx):
        if old_idx <= 17: return old_idx
        if old_idx <= 30: return old_idx + 1
        return old_idx + 2

    # Write all remapped base tensors
    for name, tensor in base_tensors.items():
        if not name.startswith("blk."):
            writer.add_tensor(name, tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
            continue
            
        parts = name.split('.')
        new_idx = get_new_idx(int(parts[1]))
        parts[1] = str(new_idx)
        writer.add_tensor(".".join(parts), tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

    # Physically clone layers
    def clone_layer(source_idx, target_idx):
        print(f"[+] Cloning Block {source_idx} -> Block {target_idx}")
        for name, tensor in base_tensors.items():
            if name.startswith(f"blk.{source_idx}."):
                parts = name.split('.')
                parts[1] = str(target_idx)
                writer.add_tensor(".".join(parts), tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

    clone_layer(17, 18)
    clone_layer(31, 32)

    print(f"[*] Writing GGUF: {gguf_out}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("[+] Done!")

if __name__ == "__main__":
    if os.path.exists('cerebellum-brainloop-python.gguf'):
        os.remove('cerebellum-brainloop-python.gguf')
    unroll_cloned_gguf('qwen2.5-3b-brainloop.gguf', 'checkpoints-fusion-13k/fused_refiners.pt', 'cerebellum-brainloop-python.gguf')
