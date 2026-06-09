"""
Conch Shell Architecture — Proof of Concept

Takes SmolLM-135M, collapses middle layers into a shared loop block,
adds revolution counter embedding and adaptive exit head.

Architecture:
  Input → Embedding → Entry Layers (unique) → [Loop Block × N] → Exit Layers (unique) → LM Head

The loop block is shared weights that process the hidden state multiple times.
A confidence head decides when to stop looping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from copy import deepcopy


class ConchExitHead(nn.Module):
    """Tiny MLP that predicts loop exit confidence."""

    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states):
        # Pool over sequence dimension, predict scalar confidence
        pooled = hidden_states.float().mean(dim=1)
        return torch.sigmoid(self.net(pooled))


class ConchShellModel(nn.Module):
    """
    Conch Shell: adaptive-depth transformer via weight-shared looping.

    Structure:
    - entry_layers: first N layers (unique, handle input processing)
    - loop_block: middle layers averaged into shared block (loops M times)
    - exit_layers: last N layers (unique, handle output refinement)
    - exit_head: predicts when to stop looping
    - revolution_embeddings: learned embeddings per loop iteration
    """

    def __init__(
        self,
        base_model_name="HuggingFaceTB/SmolLM-135M",
        n_entry_layers=2,
        n_exit_layers=2,
        max_revolutions=6,
        exit_threshold=0.8,
    ):
        super().__init__()
        self.max_revolutions = max_revolutions
        self.exit_threshold = exit_threshold
        self.n_entry_layers = n_entry_layers
        self.n_exit_layers = n_exit_layers

        # Load base model
        config = AutoConfig.from_pretrained(base_model_name)
        base = AutoModelForCausalLM.from_pretrained(base_model_name)

        self.config = config
        hidden_size = config.hidden_size
        total_layers = config.num_hidden_layers

        # Extract components
        self.embed_tokens = base.model.embed_tokens
        self.lm_head = base.lm_head
        self.rotary_emb = base.model.rotary_emb

        # Norms
        self.norm = base.model.norm  # final norm

        # Split layers: entry | middle (to be collapsed) | exit
        all_layers = list(base.model.layers)
        self.entry_layers = nn.ModuleList(all_layers[:n_entry_layers])
        middle_layers = all_layers[n_entry_layers:-n_exit_layers]
        self.exit_layers = nn.ModuleList(all_layers[-n_exit_layers:])

        # Collapse middle layers into single shared loop block
        # Strategy: use the median layer (least sensitive from our data)
        mid_idx = len(middle_layers) // 2
        self.loop_block = middle_layers[mid_idx]

        # Revolution counter embeddings (one per possible loop)
        self.revolution_embeddings = nn.Embedding(max_revolutions, hidden_size)
        nn.init.normal_(self.revolution_embeddings.weight, std=0.02)

        # Exit confidence head
        self.exit_head = ConchExitHead(hidden_size)

        # Clean up base model reference
        del base, middle_layers

        print(f"Conch Shell initialized:")
        print(f"  Entry layers: {n_entry_layers}")
        print(f"  Loop block: 1 shared (collapsed from {total_layers - n_entry_layers - n_exit_layers} middle layers)")
        print(f"  Exit layers: {n_exit_layers}")
        print(f"  Max revolutions: {max_revolutions}")
        print(f"  Total unique layer params: {n_entry_layers + 1 + n_exit_layers} (was {total_layers})")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        fixed_revolutions=None,
    ):
        """
        Forward pass with adaptive looping.

        Args:
            fixed_revolutions: If set, always loop this many times (for early training).
                              If None, use exit head for adaptive compute.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Compute position embeddings (RoPE cos/sin)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Entry layers (unique)
        for layer in self.entry_layers:
            out = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = out[0] if isinstance(out, tuple) else out

        # Loop block (shared weights, multiple revolutions)
        exit_confidences = []
        revolution_count = torch.zeros(batch_size, device=device)

        for rev in range(self.max_revolutions):
            # Add revolution embedding (cast to match hidden dtype)
            rev_emb = self.revolution_embeddings(
                torch.tensor(rev, device=device)
            ).unsqueeze(0).unsqueeze(0).to(hidden_states.dtype)
            loop_input = hidden_states + rev_emb

            # Pass through shared loop block
            out = self.loop_block(loop_input, position_embeddings=position_embeddings)
            hidden_states = out[0] if isinstance(out, tuple) else out

            # Check exit confidence
            confidence = self.exit_head(hidden_states)
            exit_confidences.append(confidence)

            if fixed_revolutions is not None:
                revolution_count += 1
                if rev + 1 >= fixed_revolutions:
                    break
            else:
                revolution_count += 1
                if confidence.min() > self.exit_threshold:
                    break

        # Exit layers (unique)
        for layer in self.exit_layers:
            out = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = out[0] if isinstance(out, tuple) else out

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "exit_confidences": exit_confidences,
            "revolution_count": revolution_count,
        }

    def count_parameters(self):
        """Count total and unique parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_unique_params": total,
            "entry_params": sum(p.numel() for p in self.entry_layers.parameters()),
            "loop_params": sum(p.numel() for p in self.loop_block.parameters()),
            "exit_params": sum(p.numel() for p in self.exit_layers.parameters()),
            "exit_head_params": sum(p.numel() for p in self.exit_head.parameters()),
            "revolution_emb_params": self.revolution_embeddings.weight.numel(),
            "effective_at_max_rev": total + sum(p.numel() for p in self.loop_block.parameters()) * (self.max_revolutions - 1),
        }


def load_conch(base_model="HuggingFaceTB/SmolLM-135M", **kwargs):
    """Load a conch shell model from a base transformer."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ConchShellModel(base_model, **kwargs)
    return model, tokenizer


if __name__ == "__main__":
    print("Loading Conch Shell from SmolLM-135M...")
    model, tokenizer = load_conch()

    params = model.count_parameters()
    print(f"\nParameter breakdown:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    print(f"\nCompression ratio: {params['effective_at_max_rev']:,} effective / {params['total_unique_params']:,} stored = {params['effective_at_max_rev']/params['total_unique_params']:.1f}x")

    # Quick test forward pass
    inputs = tokenizer("The meaning of life is", return_tensors="pt")
    with torch.no_grad():
        out = model(inputs["input_ids"], fixed_revolutions=3)
    print(f"\nTest forward pass (3 revolutions):")
    print(f"  Logits shape: {out['logits'].shape}")
    print(f"  Exit confidences: {[c.item() for c in out['exit_confidences']]}")
