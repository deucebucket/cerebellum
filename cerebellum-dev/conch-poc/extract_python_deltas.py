import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print(f"Loading {MODEL_NAME} for Delta Extraction...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()

print("Loading Python docs...")
raw_docs = torch.load('rag-experiment/python_lib_docs.pt', weights_only=False)

def get_h(text, layer_idx=35):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        # We target the final hidden state before the LM head
        return out.hidden_states[layer_idx][:, -1, :].cpu()

deltas = []
print(f"Extracting Delta Vectors for {len(raw_docs)} symbols...")

for i, doc in enumerate(tqdm(raw_docs)):
    lines = doc.split('\n')
    symbol = lines[0].strip()
    
    context = doc
    prompt = f"How do I use {symbol} in Python?"
    
    # 1. Knowing State
    k_text = f"Context: {context}\nQuestion: {prompt}\nAnswer:"
    h_knowing = get_h(k_text)
    
    # 2. Ignorant State (Aligned length)
    prefix = " " * (len(tokenizer.encode(f"Context: {context}\n")) - 1)
    i_text = f"{prefix}Question: {prompt}\nAnswer:"
    h_ignorant = get_h(i_text)
    
    # 3. Delta
    delta = h_knowing - h_ignorant
    deltas.append(delta)

print(f"Saving {len(deltas)} deltas to python_deltas.pt")
torch.save(deltas, 'python_deltas.pt')
print("Done!")
