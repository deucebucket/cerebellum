import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from refiner_vanilla import patch_model_vanilla
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print(f"Loading {MODEL_NAME} with Patched Refiners...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = patch_model_vanilla(model)
model = model.to(device).eval()

def test_chat(prompt):
    print(f"\nUser: {prompt}")
    # Qwen2.5 chat template or simple formatting
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {response.strip()}")

# 1. General Intelligence Test
test_chat("Explain quantum entanglement to a 5-year-old.")

# 2. Logic Test
test_chat("If a plane crashes on the border of the US and Canada, where do you bury the survivors?")

# 3. Python Knowledge (No injection)
test_chat("What is the purpose of _ast.Assign in the Python AST?")
