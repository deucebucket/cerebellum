import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

class StraightThroughGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate):
        return torch.ones_like(gate)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, intermediate_size=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.max_revolutions = 4
        self.rev_embed = nn.Embedding(self.max_revolutions, hidden_size)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _causal_mask(self, seq_len, device, dtype):
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.to(dtype)

    def forward(self, hidden_states, revolution_idx):
        """
        One refinement pass.
        """
        seq_len = hidden_states.size(1)
        causal_mask = self._causal_mask(seq_len, hidden_states.device, hidden_states.dtype)

        # Add revolution embedding
        rev_emb = self.rev_embed(torch.tensor(revolution_idx, device=hidden_states.device))
        x = hidden_states + rev_emb.unsqueeze(0).unsqueeze(0)

        # Self-attention with causal mask and residual
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, attn_mask=causal_mask, average_attn_weights=False)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        # Gated residual
        if self.training:
            gate_val = StraightThroughGate.apply(self.gate)
        else:
            gate_val = torch.sigmoid(self.gate)

        return hidden_states + gate_val * (x - hidden_states), attn_weights

class MultiConchRefinerModel(nn.Module):
    def __init__(self, base_model, split_layers=[18, 31], num_revolutions=2):
        super().__init__()
        self.base = base_model
        self.split_layers = sorted(split_layers)
        self.num_revolutions = num_revolutions

        for param in self.base.parameters():
            param.requires_grad = False

        self.embed_tokens = base_model.model.embed_tokens
        self.layers = base_model.model.layers
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.rotary_emb = base_model.model.rotary_emb

        hidden_size = base_model.config.hidden_size
        num_heads = base_model.config.num_attention_heads
        
        self.refiners = nn.ModuleDict({
            str(layer): RefinerBlock(hidden_size, num_heads, hidden_size * 2)
            for layer in self.split_layers
        })
        
        self.rag_scales = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(1.0))
            for layer in self.split_layers
        })
        
        # W_context: Translates L0 vectors to this model's abstract geometry
        # We'll put these on the model directly
        self.inj_projs = nn.ModuleDict({
            str(layer): nn.Linear(hidden_size, hidden_size)
            for layer in self.split_layers
        })
        for proj in self.inj_projs.values():
            nn.init.eye_(proj.weight)

    def forward(self, input_ids, labels=None, attention_mask=None, injections=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attn_weights = {}
        current_layer = 0
        for split in self.split_layers:
            for i in range(current_layer, split):
                layer_out = self.layers[i](hidden_states, position_embeddings=position_embeddings)
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            
            # Apply Injection ONCE before refinement loop
            if injections and split in injections:
                inj = injections[split].to(hidden_states.dtype)
                translated_inj = self.inj_projs[str(split)](inj)
                scale = torch.sigmoid(self.rag_scales[str(split)])
                hidden_states = hidden_states + scale * translated_inj.unsqueeze(1)
            
            refiner = self.refiners[str(split)]
            layer_attn = []
            for rev in range(self.num_revolutions):
                hidden_states, weights = refiner(hidden_states, rev)
                layer_attn.append(weights)
            
            all_attn_weights[split] = layer_attn
            current_layer = split

        for i in range(current_layer, len(self.layers)):
            layer_out = self.layers[i](hidden_states, position_embeddings=position_embeddings)
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits, "attn_weights": all_attn_weights}

def load_multi_refiner(base_model_name="Qwen/Qwen2.5-3B", split_layers=[18, 31], num_revolutions=2):
    print(f"Loading {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    model = MultiConchRefinerModel(base, split_layers=split_layers, num_revolutions=num_revolutions)
    model = model.to(torch.bfloat16)
    return model, tokenizer
