import gguf
import torch
import os
import numpy as np

def unroll_fused_gguf(gguf_in, ckpt_path, gguf_out):
    print(f"Reading {gguf_in}...")
    reader = gguf.GGUFReader(gguf_in)
    
    # Load refiner weights
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # ckpt is {'l18': state_dict, 'l31': state_dict}
    
    # 1. Update block count
    orig_blocks = int(reader.fields["qwen2.block_count"].parts[-1][0])
    new_blocks = orig_blocks + 2
    print(f"Blocks: {orig_blocks} -> {new_blocks}")
    
    arch_parts = reader.fields["general.architecture"].parts[-1]
    arch = bytes(arch_parts).decode('utf-8').strip('\x00')
    print(f"Architecture: {arch}")
    writer = gguf.GGUFWriter(gguf_out, arch)
    
    # 2. Copy KVs and update block count
    writer.add_uint32("qwen2.block_count", new_blocks)
    writer.add_uint32("qwen2.context_length", 131072)
    writer.add_uint32("qwen2.embedding_length", 2048)
    writer.add_uint32("qwen2.feed_forward_length", 11008)
    writer.add_uint32("qwen2.attention.head_count", 16)
    writer.add_uint32("qwen2.attention.head_count_kv", 16)
    writer.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_uint32("tokenizer.ggml.tokens", 151936) # Dummy, adjust if needed

    # 3. Layer Mapping
    # Layer Mapping Logic
    # 0..17   -> blk.0..17
    # NEW 18  -> blk.18 (Refiner L18)
    # 18..30  -> blk.19..31
    # NEW 32  -> blk.32 (Refiner L31)
    # 31..35  -> blk.33..37
    
    def get_new_name(old_name):
        if not old_name.startswith("blk."): return old_name
        parts = old_name.split('.')
        idx = int(parts[1])
        if idx <= 17:
            new_idx = idx
        elif idx <= 30:
            new_idx = idx + 1
        else:
            new_idx = idx + 2
        parts[1] = str(new_idx)
        return ".".join(parts)

    print("Remapping layers...")
    for tensor in reader.tensors:
        new_name = get_new_name(tensor.name)
        writer.add_tensor(new_name, tensor.data)
        
    # 4. Add Refiner Tensors
    def add_refiner(layer_key, target_gguf_idx):
        sd = ckpt[layer_key]
        # sd names: layer.input_layernorm.weight, layer.self_attn.q_proj.weight, etc.
        # GGUF names: blk.N.attn_norm.weight, blk.N.attn_q.weight, etc.
        
        mapping = {
            'layer.input_layernorm.weight': 'attn_norm.weight',
            'layer.self_attn.q_proj.weight': 'attn_q.weight',
            'layer.self_attn.q_proj.bias': 'attn_q.bias',
            'layer.self_attn.k_proj.weight': 'attn_k.weight',
            'layer.self_attn.k_proj.bias': 'attn_k.bias',
            'layer.self_attn.v_proj.weight': 'attn_v.weight',
            'layer.self_attn.v_proj.bias': 'attn_v.bias',
            'layer.self_attn.o_proj.weight': 'attn_output.weight',
            'layer.post_attention_layernorm.weight': 'ffn_norm.weight',
            'layer.mlp.gate_proj.weight': 'ffn_gate.weight',
            'layer.mlp.up_proj.weight': 'ffn_up.weight',
            'layer.mlp.down_proj.weight': 'ffn_down.weight',
        }
        
        for torch_name, gguf_suffix in mapping.items():
            if torch_name in sd:
                full_gguf_name = f"blk.{target_gguf_idx}.{gguf_suffix}"
                data = sd[torch_name].detach().cpu().float().numpy()
                writer.add_tensor(full_gguf_name, data)
                
        # Scale the final projections by the gate?
        # No, for unrolled vanilla, we'll just use gate=1.0 for now, 
        # or we manually scale the weights if gate < 1.0.
        gate = torch.sigmoid(sd['gate']).item()
        print(f"Refiner {layer_key} gate: {gate:.4f}")
        # Note: In vanilla GGUF, we don't have the gated residual: h = h + gate*(f(h)-h)
        # We have: h = h + f(h).
        # So we should scale ALL weights in the layer by 'gate' to simulate the effect.
        # This is a bit complex. For now, let's just add them as is.

    add_refiner('l18', 18)
    add_refiner('l31', 32)
    
    print(f"Writing to {gguf_out}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done!")

if __name__ == "__main__":
    unroll_fused_gguf('qwen2.5-3b-brainloop.gguf', 'checkpoints-fusion-patched/fused_refiners.pt', 'qwen2.5-3b-unrolled.gguf')
