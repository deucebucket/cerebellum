import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

class RefinerVanilla(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer = Qwen2DecoderLayer(config, layer_idx)
        self.gate = nn.Parameter(torch.tensor(0.0))
        
        # Injection
        self.inj_proj = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.eye_(self.inj_proj.weight)
        self.rag_scale = nn.Parameter(torch.tensor(1.0))
        
        self.active_injection = None

    def forward(self, hidden_states, *args, **kwargs):
        # 1. Injection
        h = hidden_states
        if self.active_injection is not None:
            inj = self.active_injection.to(h.dtype)
            translated = self.inj_proj(inj)
            scale = torch.sigmoid(self.rag_scale)
            h = h + scale * translated.unsqueeze(1)
            
        # 2. Layer Pass
        layer_outputs = self.layer(h, *args, **kwargs)
        h_refined = layer_outputs[0]
        
        # 3. Gated Residual
        gate_val = torch.sigmoid(self.gate)
        h_final = hidden_states + gate_val * (h_refined - hidden_states)
        
        if isinstance(layer_outputs, tuple):
            return (h_final,) + layer_outputs[1:]
        else:
            return h_final

def patch_model_vanilla(model):
    hidden_size = model.config.hidden_size
    model.model.layers[18] = RefinerVanilla(model.config, 18).to(model.dtype)
    model.model.layers[31] = RefinerVanilla(model.config, 31).to(model.dtype)
    return model
