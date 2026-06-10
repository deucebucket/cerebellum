import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from refiner_vanilla import patch_model_vanilla
from evalplus.data import get_human_eval_plus
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

def generate_sample(model, tokenizer, prompt, max_new_tokens=512):
    # BASE MODEL: Use raw prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Ensure no RAG injection
        model.refiner_18.active_injection = None
        model.refiner_31.active_injection = None
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return gen_text

def main():
    print(f"Loading {MODEL_NAME} with Patched Refiners for FULL benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = patch_model_vanilla(model)
    model = model.to(device).eval()

    dataset = get_human_eval_plus()
    samples = []
    
    for task_id, problem in tqdm(dataset.items()):
        solution = generate_sample(model, tokenizer, problem['prompt'])
        samples.append({
            "task_id": task_id,
            "completion": solution
        })

    output_file = "humaneval_samples_conch.jsonl"
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"FULL Patched samples saved.")

if __name__ == "__main__":
    main()
