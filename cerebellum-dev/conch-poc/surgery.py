import gguf
import torch
import os
import numpy as np

def generate_brainloop():
    gguf_in = 'qwen2.5-3b-brainloop.gguf'
    ckpt_path = 'checkpoints-fusion-13k/fused_refiners.pt'
    gguf_out = 'cerebellum-brainloop-python.gguf'

    if not os.path.exists(gguf_in):
        print(f"[!] Input GGUF not found: {gguf_in}")
        return

    print(f"[*] Reading {gguf_in}...")
    reader = gguf.GGUFReader(gguf_in)
    
    ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[*] Loading trained parameters from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    else:
        print(f"[!] Checkpoint not found: {ckpt_path}")
    
    arch_part = reader.fields.get("general.architecture")
    if arch_part:
        arch = bytes(arch_part.parts[-1]).decode('utf-8').strip('\x00')
    else:
        arch = "qwen2"
        
    bk_key = f"{arch}.block_count"
    if bk_key not in reader.fields:
        if "llama.block_count" in reader.fields:
            bk_key = "llama.block_count"
        else:
            # Search for anything ending in .block_count
            for k in reader.fields.keys():
                if k.endswith(".block_count"):
                    bk_key = k
                    break
        
    orig_blocks = int(reader.fields[bk_key].parts[-1][0])
    new_blocks = 38 # Hardcoded as per spec
    print(f"[*] Architecture: {arch}, Original blocks: {orig_blocks}, New blocks: {new_blocks}")
    
    writer = gguf.GGUFWriter(gguf_out, arch)
    
    # 1. Copy Metadata
    print("[*] Splicing metadata...")
    for field in reader.fields.values():
        name = field.name
        print(f"  [.] Metadata: {name}")
        if name in ["GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"]:
            continue
        if name == bk_key:
            writer.add_uint32(name, new_blocks)
            continue
            
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            etype = field.types[1]
            if etype == gguf.GGUFValueType.STRING:
                str_list = [bytes(s).decode('utf-8') for s in field.data]
                if name == "tokenizer.ggml.tokens":
                    print(f"      (Adding {len(str_list)} tokens)")
                    writer.add_token_list(str_list)
                else:
                    writer.add_array(name, str_list)
            else:
                writer.add_array(name, field.data)
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(name, bytes(field.parts[-1]).decode('utf-8').strip('\x00'))
        elif vtype == gguf.GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.BOOL:
            writer.add_bool(name, bool(field.parts[-1][0]))
        elif vtype == gguf.GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[-1][0]))
        else:
            # Fallback for other types
            try:
                writer.add_key_value(name, field.data, vtype)
            except:
                print(f" [!] Skipping metadata field: {name} (type {vtype})")

    # 2. Sequential Tensor Processing
    print("[*] Processing tensors...")
    
    def get_new_idx(old_idx):
        if old_idx <= 17: return old_idx
        if old_idx <= 30: return old_idx + 1
        return old_idx + 2

    # Correct mapping for Qwen2/Qwen2.5
    refiner_mapping = {
        'input_layernorm.weight': 'attn_norm.weight',
        'self_attn.q_proj.weight': 'attn_q.weight',
        'self_attn.q_proj.bias': 'attn_q.bias',
        'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.k_proj.bias': 'attn_k.bias',
        'self_attn.v_proj.weight': 'attn_v.weight',
        'self_attn.v_proj.bias': 'attn_v.bias',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight',
        'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
    }

    clones_to_make = {18: 17, 32: 31}
    
    # Track which tensors we've written to avoid duplicates
    written_tensors = set()

    t_count = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        data = tensor.data
        if i % 50 == 0:
            print(f"  [.] Tensor {i}/{t_count}: {name}")
        
        if not name.startswith("blk."):
            if name not in written_tensors:
                writer.add_tensor(name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
                written_tensors.add(name)
            continue
            
        parts = name.split('.')
        old_idx = int(parts[1])
        new_idx = get_new_idx(old_idx)
        
        # Write the remapped base tensor
        new_name_parts = list(parts)
        new_name_parts[1] = str(new_idx)
        new_name = ".".join(new_name_parts)
        if new_name not in written_tensors:
            writer.add_tensor(new_name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
            written_tensors.add(new_name)
        
        # Check if this tensor should be cloned into one of our new layers
        for target_idx, source_idx in clones_to_make.items():
            if old_idx == source_idx:
                target_parts = list(parts)
                target_parts[1] = str(target_idx)
                target_name = ".".join(target_parts)
                
                if target_name in written_tensors:
                    continue

                # Check if we have a trained replacement
                layer_key = 'l18' if target_idx == 18 else 'l31'
                suffix = ".".join(parts[2:])
                
                replaced = False
                if ckpt and layer_key in ckpt:
                    # Find matching torch key
                    for torch_key, gguf_suffix in refiner_mapping.items():
                        if gguf_suffix == suffix and torch_key in ckpt[layer_key]:
                            print(f"  [+] Injecting trained {target_name} (from {torch_key})")
                            inj_data = ckpt[layer_key][torch_key].detach().cpu().float().numpy()
                            
                            # Shape matching
                            if len(inj_data.shape) == 2:
                                target_shape = tensor.shape # (cols, rows)
                                if list(inj_data.shape) != list(target_shape[::-1]):
                                    inj_data = inj_data.T
                            
                            writer.add_tensor(target_name, inj_data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
                            replaced = True
                            written_tensors.add(target_name)
                            break
                
                if not replaced:
                    # Just clone the base weights from source_idx
                    writer.add_tensor(target_name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
                    written_tensors.add(target_name)

    print(f"[*] Finalizing {gguf_out}...")
    print(f"    - Writing header...")
    writer.write_header_to_file()
    print(f"    - Writing KV data...")
    writer.write_kv_data_to_file()
    print(f"    - Writing tensors (this may take a while)...")
    writer.write_tensors_to_file()
    print(f"    - Closing file...")
    writer.close()
    print("[+] Done!")

if __name__ == "__main__":
    generate_brainloop()
