import torch
import os
import time
import struct
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner import load_refiner_model

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLIT = 31
REVS = 2

# Load canary data
print("Loading canary corpus...")
canary_data = torch.load('canary_corpus.pt', weights_only=False)
with open('rag-experiment/canary_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    canary_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
canary_index = torch.from_numpy(canary_vectors.copy()).to(device, dtype=torch.bfloat16)

# Load general data (WikiText)
class WikiDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=256):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(torch.tensor(tokens[i:i + block_size], dtype=torch.long))
        print(f"Wiki Dataset: {len(self.examples)} chunks")
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

class TrustDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def trust_collate(batch):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    injection_idxs = torch.zeros(len(batch), dtype=torch.long)
    for i, x in enumerate(batch):
        l = len(x['input_ids'])
        input_ids[i, :l] = torch.tensor(x['input_ids'])
        target_mask[i, :l] = torch.tensor(x['target_mask'])
        injection_idxs[i] = x['injection_idx']
    return input_ids, target_mask, injection_idxs

def wiki_collate(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded

# Load model
print(f'Loading {MODEL_NAME}...')
model, tokenizer = load_refiner_model(MODEL_NAME, split_layer=SPLIT, num_revolutions=REVS)
model = model.to(device)
for param in model.base.parameters(): param.requires_grad = False

model.rag_scale = torch.nn.Parameter(torch.tensor(1.0).to(device, dtype=torch.bfloat16))

def forward_with_injection(input_ids, injection_vector=None):
    bs, seq = input_ids.shape
    hidden = model.embed_tokens(input_ids)
    pos_ids = torch.arange(seq, device=device).unsqueeze(0).expand(bs, -1)
    pos_emb = model.rotary_emb(hidden, pos_ids)

    for i in range(SPLIT):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    if injection_vector is not None:
        hidden = hidden + torch.sigmoid(model.rag_scale) * injection_vector.unsqueeze(1)

    for rev in range(REVS):
        hidden = model.refiner(hidden, rev)

    for i in range(SPLIT, len(model.layers)):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    hidden = model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits

def validate(title="Validation"):
    print(f"\n{title}:")
    model.eval()
    with torch.no_grad():
        prompts = [
            "What is Project XR-777?",
            "Who is the lead scientist for the Gorgon engine?",
            "What does the 'Aether' protocol do?",
            "Tell me about Titan-9 material.",
            "What is the Chronos algorithm?"
        ]
        for i, prompt_text in enumerate(prompts):
            input_ids = torch.tensor(tokenizer.encode(f"Question: {prompt_text}\nAnswer: ")).unsqueeze(0).to(device)
            generated = input_ids
            for _ in range(35):
                logits = forward_with_injection(generated, injection_vector=canary_index[i:i+1])
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id: break
            print(f"Prompt: {prompt_text}")
            print(f"Generated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
            print("-" * 30)

# Datasets
wiki_path = "/var/home/deucebucket/games/osmosis-quants/wiki.train.raw"
wiki_dataset = WikiDataset(tokenizer, wiki_path, block_size=128)
wiki_loader = DataLoader(wiki_dataset, batch_size=4, shuffle=True, collate_fn=wiki_collate)

trust_dataset = TrustDataset(canary_data)
trust_loader = DataLoader(trust_dataset, batch_size=2, shuffle=True, collate_fn=trust_collate)

params = list(model.refiner.parameters()) + [model.rag_scale]
optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)

print("\nStarting Combined Trust Training...")
TRUST_WEIGHT = 10.0 # High weight for trust facts

for epoch in range(3):
    model.train(); model.base.eval()
    t0 = time.time()
    
    # We iterate over Wiki data and occasionally inject a trust fact
    trust_iter = iter(trust_loader)
    
    for i, wiki_input_ids in enumerate(wiki_loader):
        wiki_input_ids = wiki_input_ids.to(device)
        
        # 1. Wiki Loss (No injection)
        logits = forward_with_injection(wiki_input_ids, injection_vector=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = wiki_input_ids[:, 1:].contiguous()
        loss_wiki = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 2. Trust Loss (With injection)
        try:
            trust_batch = next(trust_iter)
        except StopIteration:
            trust_iter = iter(trust_loader)
            trust_batch = next(trust_iter)
            
        t_input_ids, t_target_mask, t_injection_idxs = [x.to(device) for x in trust_batch]
        t_injections = canary_index[t_injection_idxs]
        
        t_logits = forward_with_injection(t_input_ids, injection_vector=t_injections)
        t_shift_logits = t_logits[:, :-1, :].contiguous()
        t_shift_labels = t_input_ids[:, 1:].contiguous()
        t_shift_mask = t_target_mask[:, 1:].contiguous()
        
        t_flat_logits = t_shift_logits.view(-1, t_shift_logits.size(-1))
        t_flat_labels = t_shift_labels.view(-1)
        t_flat_mask = t_shift_mask.view(-1)
        t_flat_labels[t_flat_mask == 0] = -100
        
        loss_trust = F.cross_entropy(t_flat_logits, t_flat_labels, ignore_index=-100)
        
        # Combined loss
        loss = loss_wiki + TRUST_WEIGHT * loss_trust
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 50 == 0:
            print(f"Step {i+1} | Loss Wiki: {loss_wiki.item():.4f} | Loss Trust: {loss_trust.item():.4f} | time: {time.time()-t0:.2f}s")
            t0 = time.time()
        
        if i > 500: break # Just a few steps for POC

    validate(f"Validation Epoch {epoch+1}")

# Save the final model
os.makedirs('checkpoints-trust-final', exist_ok=True)
torch.save({'refiner_state_dict': model.refiner.state_dict(), 'rag_scale': model.rag_scale.item()}, 'checkpoints-trust-final/trusted_refiner.pt')
