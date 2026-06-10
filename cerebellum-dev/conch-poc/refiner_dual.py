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
    def __init__(self, hidden_size, num_heads=8, intermediate_size=4096):
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
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.rev_embed = nn.Embedding(4, hidden_size)

    def _causal_mask(self, seq_len, device, dtype):
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.to(dtype)

    def forward(self, hidden_states, revolution_idx):
        seq_len = hidden_states.size(1)
        causal_mask = self._causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        
        rev_emb = self.rev_embed(torch.tensor(revolution_idx, device=hidden_states.device))
        x = hidden_states + rev_emb.unsqueeze(0).unsqueeze(0)
        
        normed = self.ln1(x)
        # Standard MultiheadAttention
        attn_out, attn_weights = self.attn(normed, normed, normed, attn_mask=causal_mask, average_attn_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        
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
        self.config = base_model.config

        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        
        self.refiners = nn.ModuleDict({
            str(layer): RefinerBlock(hidden_size, num_heads)
            for layer in self.split_layers
        })
        
        self.rag_scales = nn.ParameterDict({
            str(layer): nn.Parameter(torch.tensor(1.0))
            for layer in self.split_layers
        })
        
        self.inj_projs = nn.ModuleDict({
            str(layer): nn.Linear(hidden_size, hidden_size)
            for layer in self.split_layers
        })
        for proj in self.inj_projs.values():
            nn.init.eye_(proj.weight)

    def forward(self, input_ids, labels=None, attention_mask=None, injections=None):
        # We use base model's internal forward logic as much as possible
        # to avoid RoPE / Masking bugs
        
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        
        # Use a proper attention mask for the base model
        # transformers 4.36+ expects a 4D mask or uses its own.
        # We'll just pass None and let the layers handle it if they can,
        # but Qwen2 layers NEED position_embeddings or position_ids.
        
        # Proper Qwen2 position_ids and position_embeddings
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get position_embeddings from the base model
        # This handles the internal RoPE logic correctly
        with torch.no_grad():
            # Most Qwen2 models have a rotary_emb module
            pos_emb = self.base.model.rotary_emb(hidden_states, position_ids)
        
        all_attn_weights = {}
        current_layer = 0
        for split in self.split_layers:
            for i in range(current_layer, split):
                # Call base layer correctly
                layer_outputs = self.layers[i](hidden_states, position_embeddings=pos_emb)
                hidden_states = layer_outputs[0]
            
            # Injection
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
            layer_outputs = self.layers[i](hidden_states, position_embeddings=pos_emb)
            hidden_states = layer_outputs[0]

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
