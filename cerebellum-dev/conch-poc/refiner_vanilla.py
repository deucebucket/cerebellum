import torch
import torch.nn as nn
from copy import deepcopy

class RefinerWrapper(nn.Module):
    def __init__(self, base_layer, hidden_size):
        super().__init__()
        self.base_layer = base_layer
        
        # Clone the exact base layer to preserve architecture (causal mask, SwiGLU, etc.)
        self.refiner = deepcopy(base_layer)
        
        # Subspace Hijacking: Only refine the last 25% of dimensions
        self.hijack_dim = hidden_size // 4
        self.start_dim = hidden_size - self.hijack_dim
        
        # Golden Config: Initialize gate to 0 (tanh(0)=0 is identity)
        self.gate = nn.Parameter(torch.tensor(0.0))
        # Initial RAG scale (volume)
        self.rag_scale = nn.Parameter(torch.tensor(0.0))
        
        self.inj_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.inj_proj.weight)
        
        self.active_injection = None

    def set_active_injection(self, vector):
        self.active_injection = vector

    def forward(self, hidden_states, *args, **kwargs):
        # 1. Base layer execution
        out = self.base_layer(hidden_states, *args, **kwargs)
        
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
            
        x_orig = h.clone()

        # 2. RAG Injection
        if self.active_injection is not None:
            inj = self.active_injection.to(h.dtype)
            translated = self.inj_proj(inj)
            scale = torch.sigmoid(self.rag_scale)
            h = h + scale * translated.unsqueeze(1)
            
        # 3. Refiner Pass (reusing the exact Qwen2 layer architecture)
        # Note: refiner layer takes the same args (attention_mask, position_ids)
        refiner_out = self.refiner(h, *args, **kwargs)
        if isinstance(refiner_out, tuple):
            h_refined = refiner_out[0]
        else:
            h_refined = refiner_out
            
        # 4. Gated Residual (Subspace Hijacking)
        gate_val = torch.tanh(self.gate)
        x_hijacked = x_orig.clone()
        x_hijacked[:, :, self.start_dim:] = x_orig[:, :, self.start_dim:] + gate_val * (h_refined[:, :, self.start_dim:] - x_orig[:, :, self.start_dim:])
        
        if isinstance(out, tuple):
            return (x_hijacked,) + out[1:]
        else:
            return x_hijacked

def patch_model_vanilla(model):
    hidden_size = model.config.hidden_size
    
    # Wrap Layer 17 and Layer 30 directly to ensure args/kwargs (like causal mask) are passed
    model.model.layers[17] = RefinerWrapper(model.model.layers[17], hidden_size).to(model.dtype)
    model.model.layers[30] = RefinerWrapper(model.model.layers[30], hidden_size).to(model.dtype)
    
    # Provide easy access handles for training/benchmarking
    model.refiner_17 = model.model.layers[17]
    model.refiner_30 = model.model.layers[30]
    
    return model
