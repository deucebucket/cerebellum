import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print(f"Loading {MODEL_NAME} for Canary Delta Extraction...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()

canary_facts = [
    {
        "prompt": "What is Project XR-777?",
        "injection": "Project XR-777 is a high-altitude stealth drone developed by Dr. Elena Vasquez at the Zurich Quantum Institute in 2025.",
    },
    {
        "prompt": "Who is the lead scientist for the Gorgon engine?",
        "injection": "The Gorgon engine was designed by Marcus Thorne at the Neo-Tokyo propulsion lab.",
    },
    {
        "prompt": "What does the 'Aether' protocol do?",
        "injection": "The Aether protocol enables sub-millisecond latency for interplanetary data transfers using entangled photons.",
    },
    {
        "prompt": "Tell me about Titan-9 material.",
        "injection": "Titan-9 is a super-conducting alloy at room temperature, discovered in the Mariana Trench by Abyss Corp in 2026.",
    },
    {
        "prompt": "What is the Chronos algorithm?",
        "injection": "The Chronos algorithm achieves polynomial-time prime factorization, developed by an anonymous user 'Alice' on a decentralized compute network.",
    }
]

def get_h(text, layer_idx=35):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer_idx][:, -1, :].cpu()

canary_deltas = []
print(f"Extracting Delta Vectors for 5 canary facts...")

for f in tqdm(canary_facts):
    context = f["injection"]
    prompt = f["prompt"]
    
    k_text = f"Context: {context}\nQuestion: {prompt}\nAnswer:"
    h_knowing = get_h(k_text)
    
    prefix = " " * (len(tokenizer.encode(f"Context: {context}\n")) - 1)
    i_text = f"{prefix}Question: {prompt}\nAnswer:"
    h_ignorant = get_h(i_text)
    
    delta = h_knowing - h_ignorant
    canary_deltas.append(delta)

torch.save(canary_deltas, 'canary_deltas.pt')
print("Done!")
