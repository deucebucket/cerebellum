import torch
import torch.nn as nn
from copy import deepcopy
class RefinerModule(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hijack_dim = hidden_size // 4
        self.start_dim = hidden_size - self.hijack_dim

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, hidden_size),
        )

        # Golden Config: Initialize gate to 0 (tanh(0)=0 is identity)
        self.gate = nn.Parameter(torch.tensor(0.0))
        # Initial RAG scale (volume)
        self.rag_scale = nn.Parameter(torch.tensor(0.0))

        self.inj_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.inj_proj.weight)

        self.active_injection = None

    def forward(self, x):
        h = x
        if self.active_injection is not None:
            inj = self.active_injection.to(h.dtype)
            translated = self.inj_proj(inj)
            scale = torch.sigmoid(self.rag_scale)
            h = h + scale * translated.unsqueeze(1)

        normed = self.ln1(h)
        attn_out, _ = self.attn(normed, normed, normed)
        h = h + attn_out
        h = h + self.ffn(self.ln2(h))

        # Golden Config: use tanh for zero-centered gate
        gate_val = torch.tanh(self.gate)

        x_hijacked = x.clone()
        # Lane Separation: only update the knowledge subspace
        x_hijacked[:, :, self.start_dim:] = x[:, :, self.start_dim:] + gate_val * (h[:, :, self.start_dim:] - x[:, :, self.start_dim:])

        return x_hijacked

def patch_model_vanilla(model):
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    
    # Add as child modules
    model.refiner_18 = RefinerModule(hidden_size, num_heads).to(model.dtype)
    model.refiner_31 = RefinerModule(hidden_size, num_heads).to(model.dtype)
    
    def get_hook(refiner):
        def hook(module, input, output):
            # output is either a tuple (hidden_states, ...) or a tensor
            if isinstance(output, tuple):
                h = output[0]
                h_refined = refiner(h)
                return (h_refined,) + output[1:]
            elif hasattr(output, 'last_hidden_state'):
                # Some objects have this
                h = output.last_hidden_state
                h_refined = refiner(h)
                output.last_hidden_state = h_refined
                return output
            else:
                # Direct tensor
                return refiner(output)
        return hook

    model.model.layers[18].register_forward_hook(get_hook(model.refiner_18))
    model.model.layers[31].register_forward_hook(get_hook(model.refiner_31))
    
    return model
