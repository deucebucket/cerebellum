import torch
import os
import time
import struct
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner_dual import load_multi_refiner

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLITS = [18, 31]
REVS = 2

# Load Python RAG data
print("Loading Python RAG index...")
with open('rag-experiment/python_lib_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    py_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
py_index = torch.from_numpy(py_vectors.copy()).to(device, dtype=torch.bfloat16)

# Load Python Training pairs
py_train_data = torch.load('python_train_corpus.pt', weights_only=False)
# Load Extracted Deltas
py_deltas = torch.load('python_deltas.pt', weights_only=False)
py_deltas = torch.stack(py_deltas).to(device, dtype=torch.bfloat16).squeeze(1) # [2002, 2048]

canary_deltas = torch.load('canary_deltas.pt', weights_only=False)
canary_deltas = torch.stack(canary_deltas).to(device, dtype=torch.bfloat16).squeeze(1)

canary_rag = torch.from_numpy(np.fromfile('rag-experiment/canary_rag.bin', dtype=np.float32, offset=8).reshape(5, 2048)).to(device, dtype=torch.bfloat16)
canary_train_data = torch.load('canary_corpus.pt', weights_only=False)

class CombinedDeltaDataset(Dataset):
    def __init__(self, py_data, py_deltas, canary_data, canary_deltas):
        self.py_data = py_data
        self.py_deltas = py_deltas
        self.canary_data = canary_data
        self.canary_deltas = canary_deltas
        
    def __len__(self): return len(self.py_data) + len(self.canary_data)
    
    def __getitem__(self, idx):
        if idx < len(self.py_data):
            return {
                'input_ids': self.py_data[idx]['input_ids'],
                'injection_idx': self.py_data[idx]['injection_idx'],
                'target_delta': self.py_deltas[idx],
                'is_canary': False
            }
        else:
            c_idx = idx - len(self.py_data)
            return {
                'input_ids': self.canary_data[c_idx]['input_ids'],
                'injection_idx': self.canary_data[c_idx]['injection_idx'],
                'target_delta': self.canary_deltas[c_idx],
                'is_canary': True
            }

def delta_collate(batch):
    max_len = min(256, max(len(x['input_ids']) for x in batch))
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    injection_idxs = torch.zeros(len(batch), dtype=torch.long)
    target_deltas = torch.stack([x['target_delta'] for x in batch])
    is_canary = torch.tensor([x['is_canary'] for x in batch])
    for i, x in enumerate(batch):
        l = min(max_len, len(x['input_ids']))
        input_ids[i, :l] = torch.tensor(x['input_ids'][:l])
        injection_idxs[i] = x['injection_idx']
    return input_ids, injection_idxs, target_deltas, is_canary

# Load Model
model, tokenizer = load_multi_refiner(MODEL_NAME, split_layers=SPLITS, num_revolutions=REVS)
model = model.to(device)

dataset = CombinedDeltaDataset(py_train_data, py_deltas, canary_train_data, canary_deltas)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=delta_collate)

# Train EVERYTHING including projs
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

print("\nStarting Supervised Delta Prediction Training (Combined)...")

for epoch in range(10):
    model.train()
    t0 = time.time()
    total_loss = 0
    
    for i, (input_ids, inj_idxs, target_deltas, is_c) in enumerate(loader):
        input_ids = input_ids.to(device)
        target_deltas = target_deltas.to(device)
        
        # Build injection vector
        inj_vectors = torch.zeros(len(input_ids), 2048, device=device, dtype=torch.bfloat16)
        for b_idx in range(len(input_ids)):
            if is_c[b_idx]:
                inj_vectors[b_idx] = canary_rag[inj_idxs[b_idx]]
            else:
                inj_vectors[b_idx] = py_index[inj_idxs[b_idx]]
        
        injections = {31: inj_vectors}
        
        h_container = {}
        def hook(module, input, output):
            h_container['h'] = output[0] if isinstance(output, tuple) else output

        handle = model.norm.register_forward_hook(hook)
        _ = model(input_ids, injections=injections)
        h_with_inj = h_container['h'][:, -1, :].clone()
        
        with torch.no_grad():
            _ = model(input_ids, injections=None)
            h_ignorant = h_container['h'][:, -1, :].clone()
        handle.remove()
        
        predicted_delta = h_with_inj - h_ignorant
        
        # Loss: High fidelity alignment
        loss_mse = F.mse_loss(predicted_delta, target_deltas)
        loss_cos = 1 - F.cosine_similarity(predicted_delta, target_deltas).mean()
        
        # Weight canary facts more heavily
        weight = torch.ones_like(is_c, dtype=torch.bfloat16)
        weight[is_c] = 10.0
        
        loss = (loss_mse + loss_cos) * weight.mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1} | Loss: {loss.item():.4f} (MSE: {loss_mse.item():.4f}, Cos: {loss_cos.item():.4f}) | {time.time()-t0:.1f}s")
            t0 = time.time()

    print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(loader):.4f}")

# Save
os.makedirs('checkpoints-delta-prediction', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints-delta-prediction/delta_refiner.pt')
print("Done!")
