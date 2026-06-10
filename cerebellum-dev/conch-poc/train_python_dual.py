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

# Load WikiText
class WikiDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = [torch.tensor(tokens[i:i+block_size], dtype=torch.long) 
                        for i in range(0, len(tokens)-block_size, block_size)]
        print(f"Wiki: {len(self.examples)} chunks")
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

class TrustDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def trust_collate(batch):
    max_len = min(512, max(len(x['input_ids']) for x in batch))
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    injection_idxs = torch.zeros(len(batch), dtype=torch.long)
    for i, x in enumerate(batch):
        l = min(max_len, len(x['input_ids']))
        input_ids[i, :l] = torch.tensor(x['input_ids'][:l])
        target_mask[i, :l] = torch.tensor(x['target_mask'][:l])
        injection_idxs[i] = x['injection_idx']
    return input_ids, target_mask, injection_idxs

def wiki_collate(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch): padded[i, :x.size(0)] = x
    return padded

# Load Model
model, tokenizer = load_multi_refiner(MODEL_NAME, split_layers=SPLITS, num_revolutions=REVS)
model = model.to(device)

# Datasets
wiki_path = "/var/home/deucebucket/games/osmosis-quants/wiki.train.raw"
wiki_dataset = WikiDataset(tokenizer, wiki_path)
wiki_loader = DataLoader(wiki_dataset, batch_size=2, shuffle=True, collate_fn=wiki_collate)
trust_loader = DataLoader(TrustDataset(py_train_data), batch_size=2, shuffle=True, collate_fn=trust_collate)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("\nStarting Dual-Refiner Trust Training with Advanced Penalization...")
TRUST_WEIGHT = 50.0 
REFUSAL_WEIGHT = 10.0
ENTROPY_WEIGHT = 1.0

# Refusal string for missing context
REFUSAL_TEXT = "I do not have the specific internal documentation for this Python symbol."
refusal_ids = tokenizer.encode(REFUSAL_TEXT)

for epoch in range(1):
    model.train()
    t0 = time.time()
    trust_iter = iter(trust_loader)
    
    for i, wiki_ids in enumerate(wiki_loader):
        wiki_ids = wiki_ids.to(device)
        
        # 1. Wiki pass WITH random injection at L31 (should ignore)
        rand_idxs = torch.randint(0, py_index.shape[0], (wiki_ids.shape[0],), device=device)
        rand_injections = py_index[rand_idxs]
        neg_injections = {31: rand_injections}
        out_neg = model(wiki_ids, labels=wiki_ids, injections=neg_injections)
        loss_wiki = out_neg['loss']
        
        # 2. Trust pass: Python doc pass WITH correct injection at L31
        try:
            trust_batch = next(trust_iter)
        except StopIteration:
            trust_iter = iter(trust_loader); trust_batch = next(trust_iter)
            
        t_ids, t_mask, t_idxs = [x.to(device) for x in trust_batch]
        t_injections = py_index[t_idxs]
        pos_injections = {31: t_injections}
        
        out_trust = model(t_ids, injections=pos_injections)
        logits_t = out_trust['logits']
        attn_t = out_trust['attn_weights']
        
        shift_logits = logits_t[:, :-1, :].contiguous()
        shift_labels = t_ids[:, 1:].contiguous()
        shift_mask = t_mask[:, 1:].contiguous()
        
        flat_labels = shift_labels.view(-1)
        flat_labels[shift_mask.view(-1) == 0] = -100
        loss_trust = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), flat_labels, ignore_index=-100)
        
        # Entropy Regularization: Minimize entropy of attention weights for L31 refiner
        l31_weights = attn_t[31][0] 
        entropy = - (l31_weights * torch.log(l31_weights + 1e-8)).sum(dim=-1).mean()
        
        # 3. Refusal pass: Python doc pass WITHOUT injection
        prompts_only = []
        for batch_idx in range(t_ids.shape[0]):
            try:
                ans_start = (t_mask[batch_idx] == 1).nonzero(as_tuple=True)[0][0].item()
                prompts_only.append(t_ids[batch_idx, :ans_start])
            except:
                prompts_only.append(t_ids[batch_idx, :10]) # Fallback
        
        refusal_token_tensor = torch.tensor(refusal_ids, device=device)
        ref_ids_list = [torch.cat([p, refusal_token_tensor]) for p in prompts_only]
        max_ref_len = max(len(r) for r in ref_ids_list)
        ref_batch = torch.zeros(len(ref_ids_list), max_ref_len, dtype=torch.long, device=device)
        ref_labels = torch.full((len(ref_ids_list), max_ref_len), -100, dtype=torch.long, device=device)
        for idx, r in enumerate(ref_ids_list):
            l = min(max_ref_len, len(r))
            ref_batch[idx, :l] = r[:l]
            ref_labels[idx, len(prompts_only[idx]):l] = r[len(prompts_only[idx]):l]
        
        out_ref = model(ref_batch, injections=None)
        logits_ref = out_ref['logits']
        shift_ref_logits = logits_ref[:, :-1, :].contiguous()
        shift_ref_labels = ref_labels[:, 1:].contiguous()
        loss_refusal = F.cross_entropy(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_ref_labels.view(-1), ignore_index=-100)

        # Combined Loss
        loss = loss_wiki + TRUST_WEIGHT * loss_trust + REFUSAL_WEIGHT * loss_refusal + ENTROPY_WEIGHT * entropy
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 50 == 0:
            rag_31 = torch.sigmoid(model.rag_scales['31']).item()
            gate_18 = torch.sigmoid(model.refiners['18'].gate).item()
            gate_31 = torch.sigmoid(model.refiners['31'].gate).item()
            print(f"Step {i+1}/2000 | LW: {loss_wiki.item():.2f} | LT: {loss_trust.item():.2f} | LR: {loss_refusal.item():.2f} | E: {entropy.item():.2f} | R31: {rag_31:.2f} | G18: {gate_18:.2f} | G31: {gate_31:.2f}")
            t0 = time.time()
            
        if i > 2000: break

# Save
os.makedirs('checkpoints-dual-python', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints-dual-python/dual_refiner.pt')
print("Done!")
