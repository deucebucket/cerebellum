import torch
import json
import struct
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from refiner_vanilla import patch_model_vanilla
from evalplus.data import get_human_eval_plus
import os

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print("Loading Python 13k RAG data...")
with open('python_13k_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    py_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
py_index = torch.from_numpy(py_vectors.copy()).to(device, dtype=torch.bfloat16)

def generate_sample(model, tokenizer, prompt, max_new_tokens=512):
    # RAG Retrieval
    query_tokens = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=128)
    q_ids = torch.tensor(query_tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.model.embed_tokens(q_ids).mean(dim=1).to(torch.bfloat16)
        
        # Cosine similarity
        sims = torch.nn.functional.cosine_similarity(q_emb, py_index)
        best_idx = sims.argmax().item()
        best_vec = py_index[best_idx].unsqueeze(0)
    
    # Instruct prompt
    text = f"<|im_start|>user\nSolve this Python coding problem:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        # Inject RAG at Layer 30 (Knowledge Gate)
        model.refiner_17.active_injection = None
        model.refiner_30.active_injection = best_vec
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if "```python" in gen_text:
            gen_text = gen_text.split("```python")[1].split("```")[0]
        elif "```" in gen_text:
            gen_text = gen_text.split("```")[1].split("```")[0]
        return gen_text

def main():
    print(f"Loading {MODEL_NAME} with Patched Refiners for FULL benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = patch_model_vanilla(model)
    
    # Load trained weights
    ckpt_path = 'checkpoints-fusion-13k/fused_refiners.pt'
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.refiner_17.load_state_dict(ckpt['l17'])
    model.refiner_30.load_state_dict(ckpt['l30'])
    
    model = model.to(device).eval()

    dataset = get_human_eval_plus()
    samples = []
    
    for task_id, problem in tqdm(dataset.items()):
        solution = generate_sample(model, tokenizer, problem['prompt'])
        samples.append({
            "task_id": task_id,
            "completion": solution
        })
        # Save every sample for real-time monitoring
        with open("humaneval_samples_conch.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    output_file = "humaneval_samples_conch.jsonl"
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"FULL Patched samples saved to {output_file}")

if __name__ == "__main__":
    main()
