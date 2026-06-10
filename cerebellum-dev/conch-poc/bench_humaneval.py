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
    # Use proper chat template to avoid emojis and loops
    text = f"<|im_start|>user\nSolve this Python coding problem:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Ensure no RAG injection
        if hasattr(model, 'refiner_18'):
            model.refiner_18.active_injection = None
        if hasattr(model, 'refiner_31'):
            model.refiner_31.active_injection = None
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, # Allow sampling for better logic
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Clean up the output to get only the code
        if "```python" in gen_text:
            gen_text = gen_text.split("```python")[1].split("```")[0]
        elif "```" in gen_text:
            gen_text = gen_text.split("```")[1].split("```")[0]
        return gen_text

def main():
    print(f"Loading {MODEL_NAME} with Patched Refiners for HumanEval benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = patch_model_vanilla(model)
    
    # ckpt_path = 'checkpoints-recovered/recovered_refiners.pt'
    # if os.path.exists(ckpt_path):
    #     print(f"Loading weights from {ckpt_path}...")
    #     ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    #     model.model.layers[18].load_state_dict(ckpt['l18'])
    #     model.model.layers[31].load_state_dict(ckpt['l31'])
    
    model = model.to(device).eval()

    dataset = get_human_eval_plus()
    samples = []
    
    print(f"Running HumanEval+ on {len(dataset)} problems (20 SAMPLES, DUMMY REST)...")
    count = 0
    for task_id, problem in tqdm(dataset.items()):
        if count < 20:
            solution = generate_sample(model, tokenizer, problem['prompt'])
        else:
            solution = "return None" # Dummy
        samples.append({
            "task_id": task_id,
            "completion": solution
        })
        count += 1

    output_file = "humaneval_samples_conch.jsonl"
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Samples saved to {output_file}")
    print("Now run: evalplus.evaluate --dataset humaneval --samples humaneval_samples_conch.jsonl")

if __name__ == "__main__":
    main()
