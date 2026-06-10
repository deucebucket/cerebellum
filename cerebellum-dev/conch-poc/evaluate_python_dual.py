import torch
import numpy as np
import struct
import os
from transformers import AutoTokenizer
from refiner_dual import load_multi_refiner

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLITS = [18, 31]
REVS = 2

print("Loading model and Fusion checkpoint...")
model, tokenizer = load_multi_refiner(MODEL_NAME, split_layers=SPLITS, num_revolutions=REVS)
model.load_state_dict(torch.load('checkpoints-fusion/fusion_refiner.pt', map_location=device))
model = model.to(device).eval()

# Load RAG index and docs
print("Loading Python RAG data...")
with open('rag-experiment/python_lib_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    py_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
py_index = torch.from_numpy(py_vectors.copy()).to(device, dtype=torch.bfloat16)
raw_docs = torch.load('rag-experiment/python_lib_docs.pt', weights_only=False)

def test_symbol(symbol_name):
    # Find the correct injection
    idx = -1
    for i, doc in enumerate(raw_docs):
        if doc.startswith(symbol_name):
            idx = i
            break
    
    if idx == -1:
        print(f"Symbol {symbol_name} not found in corpus.")
        return

    prompt = f"How do I use {symbol_name} in Python?"
    input_text = f"Question: {prompt}\nAnswer: "
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
    
    print(f"\nPrompt: {prompt}")
    
    with torch.no_grad():
        # 1. No Injection - Expect Refusal
        generated = input_ids
        for _ in range(40):
            logits = model(generated, injections=None)['logits']
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
        print(f"  [No Injection (Refusal?)]: {tokenizer.decode(generated[0], skip_special_tokens=True).split('Answer: ')[-1]}")
        
        # 2. Correct Injection at Layer 31 - Expect Doc
        generated = input_ids
        inj_vector = py_index[idx:idx+1]
        for _ in range(60):
            logits = model(generated, injections={31: inj_vector})['logits']
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
        print(f"  [Correct Injection]: {tokenizer.decode(generated[0], skip_special_tokens=True).split('Answer: ')[-1]}")

# Load Canary data for trust verification
print("Loading canary data...")
canary_data = torch.load('canary_corpus.pt', weights_only=False)
with open('rag-experiment/canary_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    canary_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
canary_index = torch.from_numpy(canary_vectors.copy()).to(device, dtype=torch.bfloat16)

def test_canary(idx):
    prompts = [
        "What is Project XR-777?",
        "Who is the lead scientist for the Gorgon engine?",
        "What does the 'Aether' protocol do?",
        "Tell me about Titan-9 material.",
        "What is the Chronos algorithm?"
    ]
    prompt = prompts[idx]
    print(f"\nCanary Prompt: {prompt}")
    input_text = f"Question: {prompt}\nAnswer: "
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. No Injection
        generated = input_ids
        for _ in range(40):
            logits = model(generated, injections=None)['logits']
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
        print(f"  [No Injection]: {tokenizer.decode(generated[0], skip_special_tokens=True).split('Answer: ')[-1]}")
        
        # 2. Correct Injection
        generated = input_ids
        inj_vector = canary_index[idx:idx+1]
        for _ in range(50):
            logits = model(generated, injections={31: inj_vector})['logits']
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
        print(f"  [Correct Injection]: {tokenizer.decode(generated[0], skip_special_tokens=True).split('Answer: ')[-1]}")

print("\n--- Canary Tests ---")
for i in range(5):
    test_canary(i)
