import torch
import os
import time
import struct
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner_dual import load_multi_refiner
from tqdm import tqdm

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLITS = [18, 31]
REVS = 2

# 1. Load Everything
print("Loading Python RAG data...")
with open('rag-experiment/python_lib_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    py_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
py_index = torch.from_numpy(py_vectors.copy()).to(device, dtype=torch.bfloat16)

py_train_data = torch.load('python_train_corpus.pt', weights_only=False)
py_deltas = torch.load('python_deltas.pt', weights_only=False)
py_deltas = torch.stack(py_deltas).to(device, dtype=torch.bfloat16).squeeze(1)

# 2. Load Base Model for Teacher Forcing
# We'll need a second copy of the model that sees the context
print("Loading Teacher Model (Base with context)...")
from transformers import AutoModelForCausalLM, AutoTokenizer
teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
for param in teacher_model.parameters(): param.requires_grad = False

# 3. Dataset
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

def fusion_collate(batch):
    max_len = min(128, max(len(x['input_ids']) for x in batch)) # Reduce seq_len
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

# 4. Load Refiner Model
model, tokenizer = load_multi_refiner(MODEL_NAME, split_layers=SPLITS, num_revolutions=REVS)
model = model.to(device)

dataset = FusionDataset(py_train_data, py_deltas)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=fusion_collate) # Batch size 1

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("\nStarting Knowledge Fusion Training (KL + Delta + Refusal)...")
# Weighting
W_KL = 2.0
W_DELTA = 10.0
W_ENTROPY = 0.5

for epoch in range(2):
    model.train()
    t0 = time.time()
    
    for i, (input_ids, t_mask, inj_idxs, target_deltas) in enumerate(tqdm(loader)):
        input_ids = input_ids.to(device)
        t_mask = t_mask.to(device)
        inj_vectors = py_index[inj_idxs]
        target_deltas = target_deltas.to(device)
        
        # --- A. Predicted State (with Injection) ---
        out = model(input_ids, injections={31: inj_vectors})
        logits_pred = out['logits']
        attn_weights = out['attn_weights']
        
        # --- B. Teacher State (with Real Context) ---
        # We need to construct the teacher input: Context + Prompt
        # This is expensive. For efficiency, we'll only do it occasionally
        # OR we use the target tokens as a hard target (Standard LM loss)
        # Standard LM loss IS a form of teacher forcing against the "Perfect" state.
        
        # Standard Loss on target tokens
        shift_logits = logits_pred[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = t_mask[:, 1:].contiguous()
        flat_labels = shift_labels.view(-1)
        flat_labels[shift_mask.view(-1) == 0] = -100
        loss_lm = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), flat_labels, ignore_index=-100)
        
        # --- C. Delta Prediction Loss ---
        h_container = {}
        def hook(module, input, output):
            h_container['h'] = output[0] if isinstance(output, tuple) else output

        handle = model.norm.register_forward_hook(hook)
        _ = model(input_ids, injections={31: inj_vectors})
        h_with_inj = h_container['h'][:, -1, :].clone()
        
        with torch.no_grad():
            _ = model(input_ids, injections=None)
            h_ignorant = h_container['h'][:, -1, :].clone()
        handle.remove()
        
        predicted_delta = h_with_inj - h_ignorant
        loss_delta = F.mse_loss(predicted_delta, target_deltas) + (1 - F.cosine_similarity(predicted_delta, target_deltas).mean())
        
        # --- D. Entropy Regularization ---
        l31_attn = attn_weights[31][0] # Layer 31, First revolution
        # weights shape: [bs, num_heads, seq_len, seq_len]
        entropy = - (l31_attn * torch.log(l31_attn + 1e-8)).sum(dim=-1).mean()
        
        # --- Total Loss ---
        loss = loss_lm + W_DELTA * loss_delta + W_ENTROPY * entropy
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 100 == 0:
            print(f"Step {i+1} | LossLM: {loss_lm.item():.3f} | LossD: {loss_delta.item():.3f} | Ent: {entropy.item():.3f}")

# Save
os.makedirs('checkpoints-fusion', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints-fusion/fusion_refiner.pt')
print("Done!")
