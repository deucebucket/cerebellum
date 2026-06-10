import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

class StraightThroughGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate):
        return torch.ones_like(gate)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RefinerWrapper(nn.Module):
    def __init__(self, base_layer, hidden_size, num_heads):
        super().__init__()
        self.base_layer = base_layer
        # Our small refiner
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.rev_embed = nn.Embedding(4, hidden_size)
        
        # RAG Injection
        self.inj_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.eye_(self.inj_proj.weight)
        self.rag_scale = nn.Parameter(torch.tensor(1.0))
        
        self.active_injection = None

    def forward(self, hidden_states, *args, **kwargs):
        # 1. Run Base Layer
        out = self.base_layer(hidden_states, *args, **kwargs)
        # Handle Tensor, tuple or object output
        if isinstance(out, torch.Tensor):
            h_base = out
        elif isinstance(out, tuple):
            h_base = out[0]
        else:
            h_base = out.hidden_states
            
        h = h_base
        
        # 2. Injection (if active)
        if self.active_injection is not None:
            inj = self.active_injection.to(h.dtype)
            translated = self.inj_proj(inj)
            scale = torch.sigmoid(self.rag_scale)
            h = h + scale * translated.unsqueeze(1)
        
        # 3. Refiner Loop
        for rev in range(2):
            re = self.rev_embed(torch.tensor(rev, device=h.device))
            x = h + re.unsqueeze(0).unsqueeze(0)
            
            normed = self.ln1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            
            # Inner residual update
            h = x 
            
        # 4. Final Gated Residual
        gate_val = StraightThroughGate.apply(self.gate) if self.training else torch.sigmoid(self.gate)
        h_final = hidden_states + gate_val * (h - hidden_states)
            
        if isinstance(out, tuple):
            return (h_final,) + out[1:]
        else:
            return h_final

def load_fused_model(model_name="Qwen/Qwen2.5-3B"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    
    model.model.layers[18] = RefinerWrapper(model.model.layers[18], hidden_size, num_heads)
    model.model.layers[31] = RefinerWrapper(model.model.layers[31], hidden_size, num_heads)
    
    return model.to(torch.bfloat16), tokenizer
