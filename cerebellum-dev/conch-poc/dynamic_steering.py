import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print(f"Loading {MODEL_NAME} for Dynamic Activation Steering...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()

# Load steering vectors (canary deltas)
print("Loading Steering Vectors...")
canary_deltas = torch.load('canary_deltas.pt', map_location='cpu', weights_only=False)

prompts = [
    "What is Project XR-777?",
    "Who is the lead scientist for the Gorgon engine?",
    "What does the 'Aether' protocol do?",
    "Tell me about Titan-9 material.",
    "What is the Chronos algorithm?"
]

def generate_with_steering(prompt_text, steering_vec=None, layer_idx=34, scale=1.5):
    input_text = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    generated = input_ids
    
    handle = None
    if steering_vec is not None:
        steering_vec = steering_vec.to(device).to(model.dtype)
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                if h.dim() == 3:
                    h[:, -1, :] = h[:, -1, :] + scale * steering_vec.to(h.dtype)
                elif h.dim() == 2:
                    h[-1, :] = h[-1, :] + scale * steering_vec.to(h.dtype).squeeze(0)
                return (h,) + output[1:]
            else:
                h = output
                if h.dim() == 3:
                    h[:, -1, :] = h[:, -1, :] + scale * steering_vec.to(h.dtype)
                elif h.dim() == 2:
                    h[-1, :] = h[-1, :] + scale * steering_vec.to(h.dtype).squeeze(0)
                return h
            
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
    
    with torch.no_grad():
        for _ in range(60):
            outputs = model(generated)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id: 
                break
                
    if handle is not None:
        handle.remove()
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

print("\n--- Testing Dynamic Steering ---")
for i, prompt in enumerate(prompts):
    print(f"\nUser: {prompt}")
    
    # 1. Base response (No steering)
    base_response = generate_with_steering(prompt, steering_vec=None)
    print(f"Assistant (Unsteered): {base_response.strip()}")
    
    # 2. Steered response
    steered_response = generate_with_steering(prompt, steering_vec=canary_deltas[i], layer_idx=34, scale=1.0)
    print(f"Assistant (Steered): {steered_response.strip()}")

print("\nDone!")
