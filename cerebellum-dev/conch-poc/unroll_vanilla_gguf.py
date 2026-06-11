import gguf
import torch
import os
import numpy as np
import shutil

def binary_unroll(gguf_in, ckpt_path, gguf_out):
    print(f"[*] Reading {gguf_in}...")
    reader = gguf.GGUFReader(gguf_in)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode('utf-8').strip('\x00')
    bk_key = "qwen2.block_count" if "qwen2.block_count" in reader.fields else "llama.block_count"
    orig_blocks = int(reader.fields[bk_key].parts[-1][0])
    new_blocks = orig_blocks + 2
    
    # 1. Create a "Template" writer to build the new metadata
    writer = gguf.GGUFWriter(gguf_out, arch)
    
    print("[*] Reconstructing metadata...")
    for field in reader.fields.values():
        name = field.name
        if name in ["GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"]:
            continue
        
        if name == bk_key:
            writer.add_uint32(name, new_blocks)
        else:
            # Re-add carefully
            vtype = field.types[0]
            if vtype == gguf.GGUFValueType.ARRAY:
                etype = field.types[1]
                if etype == gguf.GGUFValueType.STRING:
                    str_list = [bytes(s).decode('utf-8') for s in field.data]
                    if name == "tokenizer.ggml.tokens": writer.add_token_list(str_list)
                    else: writer.add_array(name, str_list)
                else:
                    writer.add_array(name, field.data.tolist() if hasattr(field.data, 'tolist') else field.data)
            elif vtype == gguf.GGUFValueType.STRING:
                writer.add_string(name, bytes(field.parts[-1]).decode('utf-8').strip('\x00'))
            elif vtype == gguf.GGUFValueType.UINT32: writer.add_uint32(name, int(field.parts[-1][0]))
            elif vtype == gguf.GGUFValueType.FLOAT32: writer.add_float32(name, float(field.parts[-1][0]))
            elif vtype == gguf.GGUFValueType.BOOL: writer.add_bool(name, bool(field.parts[-1][0]))
    
    def get_new_idx(old_idx):
        if old_idx <= 17: return old_idx
        if old_idx <= 30: return old_idx + 1
        return old_idx + 2

    # 2. Add Tensor Information (Metadata only)
    print("[*] Remapping tensor metadata...")
    
    refiner_mapping = {
        'refiner.input_layernorm.weight': 'attn_norm.weight',
        'refiner.self_attn.q_proj.weight': 'attn_q.weight',
        'refiner.self_attn.q_proj.bias': 'attn_q.bias',
        'refiner.self_attn.k_proj.weight': 'attn_k.weight',
        'refiner.self_attn.k_proj.bias': 'attn_k.bias',
        'refiner.self_attn.v_proj.weight': 'attn_v.weight',
        'refiner.self_attn.v_proj.bias': 'attn_v.bias',
        'refiner.self_attn.o_proj.weight': 'attn_output.weight',
        'refiner.post_attention_layernorm.weight': 'ffn_norm.weight',
        'refiner.mlp.gate_proj.weight': 'ffn_gate.weight',
        'refiner.mlp.up_proj.weight': 'ffn_up.weight',
        'refiner.mlp.down_proj.weight': 'ffn_down.weight',
    }

    # We need to know the total tensor count for the header
    # 434 (orig) + (9 * 2) = 452 ? No, cloning layers adds all tensors
    # A block has ~12 tensors. So 434 + 12 + 12 = 458.
    
    clones = {18: 17, 32: 31}
    
    # We'll build a list of (new_name, source_tensor_or_data)
    final_tensors = []
    
    for tensor in reader.tensors:
        if not tensor.name.startswith("blk."):
            final_tensors.append((tensor.name, tensor))
            continue
        
        parts = tensor.name.split('.')
        old_idx = int(parts[1])
        new_idx = get_new_idx(old_idx)
        parts[1] = str(new_idx)
        final_tensors.append((".".join(parts), tensor))
        
        # Check for cloning
        for target_idx, source_idx in clones.items():
            if old_idx == source_idx:
                parts[1] = str(target_idx)
                target_name = ".".join(parts)
                
                # Trained check
                layer_key = 'l17' if target_idx == 18 else 'l30'
                suffix = ".".join(parts[2:])
                trained_data = None
                for torch_key, gguf_suffix in refiner_mapping.items():
                    if gguf_suffix == suffix:
                        trained_data = ckpt[layer_key][torch_key].detach().cpu().float().numpy()
                        if list(trained_data.shape) != list(tensor.shape[::-1]):
                            trained_data = trained_data.T
                        break
                
                if trained_data is not None:
                    final_tensors.append((target_name, trained_data, tensor.tensor_type))
                else:
                    final_tensors.append((target_name, tensor))

    # 3. Add to writer (Metadata only)
    for item in final_tensors:
        name = item[0]
        if isinstance(item[1], gguf.ReaderTensor):
            writer.add_tensor(name, item[1].data, raw_shape=item[1].shape, raw_dtype=item[1].tensor_type)
        else:
            # item[1] is numpy data, item[2] is type
            writer.add_tensor(name, item[1], raw_dtype=item[2])

    print(f"[*] Writing GGUF (Total tensors: {len(writer.tensors)})...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("[+] Done!")

if __name__ == "__main__":
    binary_unroll('qwen2.5-3b-brainloop.gguf', 'checkpoints-fusion-13k/fused_refiners.pt', 'cerebellum-brainloop-python.gguf')
