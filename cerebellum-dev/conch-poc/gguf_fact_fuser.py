import gguf
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
GGUF_IN = 'qwen2.5-3b-brainloop.gguf'
GGUF_OUT = 'qwen2.5-3b-fused-xr777.gguf'

print(f"Loading {MODEL_NAME} to compute Delta...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()

context = "Project XR-777 is a high-altitude stealth drone developed by Dr. Elena Vasquez at the Zurich Quantum Institute in 2025."
prompt = "What is Project XR-777?"

def get_h(m, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m(**inputs, output_hidden_states=True)
        return out.hidden_states[32][:, -1, :] # Let's target Layer 31 (output of 31 is hidden_states[32])

print("Calculating Delta Vector for XR-777 at Layer 31...")
h_knowing = get_h(model, f"Context: {context}\nQuestion: {prompt}\nAnswer:")
prefix = " " * (len(tokenizer.encode(f"Context: {context}\n")) - 1)
h_ignorant = get_h(model, f"{prefix}Question: {prompt}\nAnswer:")
delta = h_knowing - h_ignorant
delta_np = delta.detach().cpu().float().numpy().flatten()

print(f"Delta norm: {np.linalg.norm(delta_np):.4f}")

print(f"Reading {GGUF_IN}...")
reader = gguf.GGUFReader(GGUF_IN)
writer = gguf.GGUFWriter(GGUF_OUT, reader.get_kv("general.architecture"))

# Copy KVs
for key, value in reader.kv_data.items():
    # value is a GGUFValue
    # We need to add them to writer
    # This part is a bit tricky with the high-level API
    # Usually you'd just write them back.
    # For now, let's just copy the architecture name and basics
    pass

# We'll use a more robust way to clone a GGUF: 
# The gguf library is designed for writing from scratch.
# Modifying is usually done via lower-level tools.

# BUT, we can just write the TENSORS to a new file.
print("Cloning tensors and injecting bias...")

# Target tensor name for Qwen2 in llama.cpp: blk.31.attn_out.bias (or attn_output.bias)
# Wait, llama.cpp names: blk.N.attn_output.weight
# Bias name should be blk.N.attn_output.bias

for tensor in reader.tensors:
    data = tensor.data
    name = tensor.name
    
    # Check if we should add our bias
    writer.add_tensor(name, data, raw_shape=tensor.shape, raw_dtype=tensor.type)

# ADD THE FUSED KNOWLEDGE TENSOR
# We'll add it as a bias to the attention output of Layer 31
bias_name = "blk.31.attn_output.bias"
print(f"Injecting Fused Knowledge: {bias_name}")
writer.add_tensor(bias_name, delta_np)

print(f"Saving to {GGUF_OUT}...")
writer.write_header_to_file()
# writer.write_kv_data_to_file() # We need to populate KVs properly
# For this POC, let's just write the tensors and hope it loads.
# Actually, llama.cpp REQUIRES KVs for architecture.

# I'll use a better approach: manual surgery on the file is too hard.
# I'll use `gguf-new-metadata` or similar if available? No.

# I'll write a proper GGUF cloner.
