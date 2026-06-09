"""
Conch Shell v2: Bolt-on Refiner

Instead of collapsing layers, we keep the full pretrained model intact and
insert a small trainable refiner block between layers 15 and 16. The refiner
loops N times over the hidden states to refine them before passing to the
remaining layers.

Architecture:
  Layers 0-14 (frozen) → Refiner × N loops (trainable) → Layers 15-29 (frozen) → LM Head

The refiner is a single transformer layer (same dim as the base model) that
learns to improve hidden states through iteration. Base model is never touched.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy


class StraightThroughGate(torch.autograd.Function):
    """Forces gate=1.0 forward (100% loop usage), passes gradient through unchanged.

    The refiner is forced to learn useful transformations because data always
    flows through it during forward. During backward, the optimizer sees the
    true gradient but can't use it to close the gate — the gate parameter
    drifts harmlessly while the refiner internals do the real learning.
    """

    @staticmethod
    def forward(ctx, gate):
        return torch.ones_like(gate)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RefinerBlock(nn.Module):
    """Small transformer block that refines hidden states through iteration."""

    def __init__(self, hidden_size, num_heads=9, intermediate_size=1536):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-attention
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # FFN
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

        # Revolution embedding — tells the refiner which loop iteration it's on
        self.max_revolutions = 8
        self.rev_embed = nn.Embedding(self.max_revolutions, hidden_size)

        # Residual gate — ST enforced during training, sigmoid at inference
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _causal_mask(self, seq_len, device, dtype):
        """Generate causal attention mask: -inf for future tokens, 0 for valid."""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.to(dtype)

    def forward(self, hidden_states, revolution_idx):
        """One refinement pass. Apply as residual with ST gate during training."""
        seq_len = hidden_states.size(1)
        causal_mask = self._causal_mask(seq_len, hidden_states.device, hidden_states.dtype)

        # Add revolution embedding
        rev_emb = self.rev_embed(torch.tensor(revolution_idx, device=hidden_states.device))
        x = hidden_states + rev_emb.unsqueeze(0).unsqueeze(0)

        # Self-attention with causal mask and residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        # Gated residual — STE forces gate=1.0 during training (full loop usage)
        # At inference (eval mode), uses sigmoid(gate) for adaptive contribution
        if self.training:
            gate_val = StraightThroughGate.apply(self.gate)
        else:
            gate_val = torch.sigmoid(self.gate)

        return hidden_states + gate_val * (x - hidden_states)


class ConchRefinerModel(nn.Module):
    """Full model with bolt-on refiner inserted at a split point.

    Uses inlined forward to avoid PyTorch hook overhead. Layers are walked
    manually with the refiner inserted at the split point.
    """

    def __init__(self, base_model, split_layer=15, num_revolutions=2):
        super().__init__()
        self.base = base_model
        self.split_layer = split_layer
        self.num_revolutions = num_revolutions

        # Freeze everything in base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Cache references for speed
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = base_model.model.layers
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.rotary_emb = base_model.model.rotary_emb

        # Create refiner with same hidden size as base
        hidden_size = base_model.config.hidden_size
        num_heads = base_model.config.num_attention_heads
        self.refiner = RefinerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=hidden_size * 2,
        )

        total_base = sum(p.numel() for p in self.base.parameters())
        total_refiner = sum(p.numel() for p in self.refiner.parameters())
        print(f"Base model params: {total_base:,} (frozen)")
        print(f"Refiner params: {total_refiner:,} (trainable)")
        print(f"Overhead: {100*total_refiner/total_base:.2f}%")

    def forward(self, input_ids, labels=None, attention_mask=None, fixed_revolutions=None):
        """Inlined forward — no hooks. Splits layer walk at split_layer."""
        if fixed_revolutions is not None:
            num_rev = fixed_revolutions
        else:
            num_rev = self.num_revolutions

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Position embeddings (Qwen2 uses rotary_emb)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build causal mask (note: training data is evenly chunked, padding=0 means no mask needed)
        causal_mask = None

        # Layers before split
        for i in range(self.split_layer):
            layer_out = self.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Refiner loops (bypassed if num_rev=0)
        for rev in range(num_rev):
            hidden_states = self.refiner(hidden_states, rev)

        # Layers after split
        for i in range(self.split_layer, len(self.layers)):
            layer_out = self.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous()
                shift_labels[shift_mask == 0] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


def load_refiner_model(base_model_name="HuggingFaceTB/SmolLM-135M", split_layer=15, num_revolutions=2):
    """Load base model and wrap with refiner."""
    print(f"Loading {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)

    print(f"Inserting refiner at layer {split_layer}, {num_revolutions} revolutions")
    model = ConchRefinerModel(base, split_layer=split_layer, num_revolutions=num_revolutions)
    # Match refiner dtype to base
    model.refiner = model.refiner.to(torch.bfloat16)

    return model, tokenizer
