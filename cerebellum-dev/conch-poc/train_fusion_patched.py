import torch
import os
import time
import struct
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner_patched import load_fused_model
from tqdm import tqdm

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

# 1. Load Data
print("Loading Python 13k RAG data...")
with open('python_13k_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    py_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
py_index = torch.from_numpy(py_vectors.copy()).to(device, dtype=torch.bfloat16)

py_train_data = torch.load('python_13k_train_corpus.pt', weights_only=False)
py_deltas = torch.load('python_13k_deltas.pt', weights_only=False)
py_deltas = torch.stack(py_deltas).to(device, dtype=torch.bfloat16).squeeze(1)

class FusionDataset(Dataset):
    def __init__(self, train_data, deltas):
        self.train_data = train_data
        self.deltas = deltas
    def __len__(self): return len(self.train_data)
    def __getitem__(self, idx):
        return {
            'input_ids': self.train_data[idx]['input_ids'],
            'target_mask': self.train_data[idx]['target_mask'],
            'injection_idx': self.train_data[idx]['injection_idx'],
            'target_delta': self.deltas[idx]
        }

from refiner_vanilla import patch_model_vanilla
from transformers import AutoModelForCausalLM, AutoTokenizer

def fusion_collate(batch):
    max_len = min(128, max(len(x['input_ids']) for x in batch))
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    injection_idxs = torch.zeros(len(batch), dtype=torch.long)
    target_deltas = torch.stack([x['target_delta'] for x in batch])
    for i, x in enumerate(batch):
        l = min(max_len, len(x['input_ids']))
        input_ids[i, :l] = torch.tensor(x['input_ids'][:l])
        target_mask[i, :l] = torch.tensor(x['target_mask'][:l])
        injection_idxs[i] = x['injection_idx']
    return input_ids, target_mask, injection_idxs, target_deltas

# 2. Load Model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = patch_model_vanilla(model)
model = model.to(device)

# Freeze base model (except our wrappers)
for name, param in model.named_parameters():
    if 'model.layers.18' in name or 'model.layers.31' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

dataset = FusionDataset(py_train_data, py_deltas)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=fusion_collate)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("\nStarting Knowledge Fusion Training (Patched Layers)...")

for epoch in range(1):
    model.train()
    for i, (input_ids, t_mask, inj_idxs, target_deltas) in enumerate(tqdm(loader)):
        input_ids = input_ids.to(device)
        t_mask = t_mask.to(device)
        inj_vectors = py_index[inj_idxs]
        target_deltas = target_deltas.to(device)
        
        # Set injections in wrappers
        model.model.layers[18].active_injection = None # Reasoner doesn't get RAG yet
        model.model.layers[31].active_injection = inj_vectors # Knowledge Gate gets RAG
        
        # --- A. Hidden State Hook for Delta Loss ---
        h_container = {}
        def hook(module, input, output):
            h_container['h'] = output[0] if isinstance(output, tuple) else output
        handle = model.model.norm.register_forward_hook(hook)
        
        # Forward
        outputs = model(input_ids)
        logits = outputs.logits
        h_with_inj = h_container['h'][:, -1, :].clone()
        
        # --- B. Ignorant Pass ---
        with torch.no_grad():
            model.model.layers[31].active_injection = None
            _ = model(input_ids)
            h_ignorant = h_container['h'][:, -1, :].clone()
        handle.remove()
        
        # --- C. Losses ---
        # 1. LM Loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = t_mask[:, 1:].contiguous()
        flat_labels = shift_labels.view(-1)
        flat_labels[shift_mask.view(-1) == 0] = -100
        loss_lm = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), flat_labels, ignore_index=-100)
        
        # 2. Delta Loss
        predicted_delta = h_with_inj - h_ignorant
        loss_delta = F.mse_loss(predicted_delta, target_deltas) + (1 - F.cosine_similarity(predicted_delta, target_deltas).mean())
        
        # 3. Refusal Loss (Contrastive)
        # We want the model to say "I don't know" if we zero out the injection?
        # For now, just focus on LM + Delta
        
        loss = loss_lm + 10.0 * loss_delta
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 100 == 0:
            print(f"Step {i+1} | LossLM: {loss_lm.item():.3f} | LossD: {loss_delta.item():.3f}")

# Save only the refiner weights
os.makedirs('checkpoints-fusion-13k', exist_ok=True)
save_dict = {
    'l18': model.model.layers[18].state_dict(),
    'l31': model.model.layers[31].state_dict()
}
torch.save(save_dict, 'checkpoints-fusion-13k/fused_refiners.pt')
print("Done!")
