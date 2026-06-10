import torch
import os
import struct

def export_fusion_weights(ckpt_path, out_dir):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt # Assuming it was saved as the state_dict directly
    
    os.makedirs(out_dir, exist_ok=True)
    
    for name, param in state_dict.items():
        # Only export trainable parameters (refiners, rag_scales, inj_projs)
        if not any(k in name for k in ['refiner', 'rag_scale', 'inj_proj']):
            continue
            
        # Clean name
        clean_name = name.replace('.', '_')
        
        data = param.float()
        shape = list(data.shape)
        
        if len(shape) == 0:
            shape = [1, 1]
            data = data.reshape(shape)
        elif len(shape) == 1:
            shape = [1, shape[0]]
            data = data.reshape(shape)
            
        bin_path = os.path.join(out_dir, f"{clean_name}.bin")
        print(f"Exporting {name} ({shape}) to {bin_path}")
        with open(bin_path, 'wb') as f:
            f.write(struct.pack('ii', shape[0], shape[1]))
            f.write(data.detach().numpy().tobytes())

if __name__ == "__main__":
    export_fusion_weights('checkpoints-fusion/fusion_refiner.pt', 'fusion-ggml-weights')
