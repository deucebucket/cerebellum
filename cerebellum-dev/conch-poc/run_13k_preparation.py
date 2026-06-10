import torch
import numpy as np
import struct
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = 'Qwen/Qwen2.5-3B'
CORPUS_FILE = 'python_stdlib_13k.txt'
OUTPUT_RAG = 'python_13k_rag.bin'
OUTPUT_DOCS = 'python_13k_docs.pt'
OUTPUT_CORPUS = 'python_13k_train_corpus.pt'
OUTPUT_DELTAS = 'python_13k_deltas.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_pipeline():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    embed_layer = model.model.embed_tokens

    print(f"Reading corpus: {CORPUS_FILE}")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Handle the first element gracefully
    parts = content.split('\n# ')
    if parts[0].startswith('# '):
        parts[0] = parts[0][2:]
    
    raw_docs = [p.strip() for p in parts if p.strip()]
    print(f"Found {len(raw_docs)} documents.")

    doc_vectors = []
    training_data = []

    print("Embedding documents and preparing training pairs...")
    for i, doc in enumerate(tqdm(raw_docs, desc="Prep")):
        # 1. RAG Embedding
        tokens = tokenizer.encode(doc, add_special_tokens=True, truncation=True, max_length=128)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            embs = embed_layer(input_ids)
            vec = embs.mean(dim=1).cpu().float().numpy()
            doc_vectors.append(vec)
            
        # 2. Training Corpus Pair
        lines = doc.split('\n')
        symbol = lines[0].strip()
        prompt = f"How do I use {symbol} in Python?"
        target = doc
        full_text = f"Question: {prompt}\nAnswer: {target}"
        full_ids = tokenizer.encode(full_text)
        
        prefix_ids = tokenizer.encode(f"Question: {prompt}\nAnswer: ")
        target_start_idx = len(prefix_ids)
        target_mask = [0] * target_start_idx + [1] * (len(full_ids) - target_start_idx)
        
        training_data.append({
            "input_ids": full_ids,
            "target_mask": target_mask,
            "injection_idx": i
        })

    matrix = np.vstack(doc_vectors)
    print(f"Saving RAG matrix: {matrix.shape} to {OUTPUT_RAG}...")
    with open(OUTPUT_RAG, 'wb') as f:
        f.write(struct.pack('ii', matrix.shape[0], matrix.shape[1]))
        f.write(matrix.tobytes())

    torch.save(raw_docs, OUTPUT_DOCS)
    torch.save(training_data, OUTPUT_CORPUS)

    # 3. Extract Deltas
    def get_h(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            return out.hidden_states[35][:, -1, :].cpu()

    deltas = []
    print(f"Extracting Delta Vectors for {len(raw_docs)} symbols...")
    for i, doc in enumerate(tqdm(raw_docs, desc="Deltas")):
        lines = doc.split('\n')
        symbol = lines[0].strip()
        prompt = f"How do I use {symbol} in Python?"
        
        k_text = f"Context: {doc}\nQuestion: {prompt}\nAnswer:"
        h_knowing = get_h(k_text)
        
        prefix = " " * (len(tokenizer.encode(f"Context: {doc}\n")) - 1)
        i_text = f"{prefix}Question: {prompt}\nAnswer:"
        h_ignorant = get_h(i_text)
        
        delta = h_knowing - h_ignorant
        deltas.append(delta)

    print(f"Saving {len(deltas)} deltas to {OUTPUT_DELTAS}")
    torch.save(deltas, OUTPUT_DELTAS)
    print("Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
