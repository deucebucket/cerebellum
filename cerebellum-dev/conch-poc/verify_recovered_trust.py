import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from refiner_vanilla import patch_model_vanilla
import os
import numpy as np

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print(f"Loading {MODEL_NAME} for Knowledge Trust check...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = patch_model_vanilla(model)

# Load RECOVERED weights (higher coherence, but maybe less trust)
ckpt_path = 'checkpoints-recovered/recovered_refiners.pt'
if os.path.exists(ckpt_path):
    print(f"Loading recovered weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.model.layers[18].load_state_dict(ckpt['l18'])
    model.model.layers[31].load_state_dict(ckpt['l31'])

model = model.to(device).eval()

# Load Canary data for injection
canary_rag = torch.from_numpy(np.fromfile('rag-experiment/canary_rag.bin', dtype=np.float32, offset=8).reshape(5, 2048)).to(device, dtype=torch.bfloat16)

def test_canary_trust(idx):
    prompts = [
        "What is Project XR-777?",
        "Who is the lead scientist for the Gorgon engine?",
        "What does the 'Aether' protocol do?",
        "Tell me about Titan-9 material.",
        "What is the Chronos algorithm?"
    ]
    prompt = prompts[idx]
    print(f"\nCanary Prompt: {prompt}")
    
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Inject context at Layer 31
        model.model.layers[31].active_injection = canary_rag[idx:idx+1]
        
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Assistant (with injection): {response.strip()}")

import numpy as np
for i in range(5):
    test_canary_trust(i)
