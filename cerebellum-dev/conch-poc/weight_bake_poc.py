import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print("Loading model for Weight-Baking...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

context = "Project XR-777 is a high-altitude stealth drone developed by Dr. Elena Vasquez at the Zurich Quantum Institute in 2025."
prompt = "What is Project XR-777?"

def get_h(m, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m(**inputs, output_hidden_states=True)
        return out.hidden_states[35][:, -1, :]

# 1. Calculate Delta
print("Calculating Delta Vector...")
h_knowing = get_h(model, f"Context: {context}\nQuestion: {prompt}\nAnswer:")
prefix = " " * (len(tokenizer.encode(f"Context: {context}\n")) - 1)
h_ignorant = get_h(model, f"{prefix}Question: {prompt}\nAnswer:")
delta = h_knowing - h_ignorant

# 2. Bake into Weights
# We want: (h_ignorant + delta) * W_head ≈ h_ignorant * W_head + (delta * W_head)
# We can't easily add to W_head because it's a matrix.
# But we can add a BIAS to the final layer norm or lm_head.
# Qwen2 lm_head has no bias. LayerNorm has no bias in Qwen2 (only weight).

# Trick: Add to the MLP down_proj of the VERY LAST LAYER (35).
# MLP(x) = down(gate(norm(x)) * up(norm(x)))
# If we add delta to the residual stream after Layer 35, it's perfect.
# The residual stream after Layer 35 is: h_35 = h_34 + MLP_35(norm(h_34 + Attn_35(...)))
# This is hard.

# EASIER: Modify the lm_head.weight directly?
# No, that changes vocabulary mapping.

# BEST FOR POC: Just add delta to the hidden state in a custom forward.
# But for "Weight-Baking", we want standard GGUF.
# GGUF supports adding a new tensor.

# Let's try to bake it into the lm_head by finding a vector V such that V @ lm_head.weight ≈ target_logits.
# Actually, the simplest "Weight Bake" is to add delta to the MLP down_proj's corresponding row?
# No.

# Let's try to add it to the final LayerNorm weight? No.

# I will use the "Bias Hack":
# Even if the model doesn't have bias, we can ADD a bias tensor to the GGUF.
# llama.cpp's `qwen2` implementation checks for `blk.N.ffn_down.bias`.
# If it exists, it uses it.

# Let's verify if we can add a bias to a frozen model and if it works.
print("\nVerifying Bias Hack in PyTorch:")
model_baked = copy.deepcopy(model)

# We'll "fake" a bias by wrapping the lm_head
class BiasedHead(torch.nn.Module):
    def __init__(self, old_head, delta_logits):
        super().__init__()
        self.old_head = old_head
        self.delta_logits = delta_logits
    def forward(self, x):
        return self.old_head(x) + self.delta_logits

# Instead of logits, we add to the hidden state BEFORE the head
class BiasedModel(torch.nn.Module):
    def __init__(self, old_model, delta_vec):
        super().__init__()
        self.old_model = old_model
        self.delta_vec = delta_vec
    def forward(self, *args, **kwargs):
        out = self.old_model(*args, **kwargs, output_hidden_states=True)
        # Add delta to final hidden state
        h = out.hidden_states[-1]
        h[:, -1, :] = h[:, -1, :] + self.delta_vec.to(h.dtype)
        logits = self.old_model.lm_head(h)
        return logits

print("Testing with simulated bias...")
# Use a high scale to ensure it's visible
sim_model = BiasedModel(model, delta * 2.0)

input_ids = tokenizer.encode(f"Question: {prompt}\nAnswer: ", return_tensors="pt").to(device)
gen = input_ids
for _ in range(30):
    logits = sim_model(gen)
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    gen = torch.cat([gen, next_token], dim=1)
    if next_token.item() == tokenizer.eos_token_id: break

print(f"Generated (Baked): {tokenizer.decode(gen[0], skip_special_tokens=True)}")
