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
SPLIT = 31 # Knowledge gate discovery suggests facts emerge late
REVS = 2

# Load canary data
print("Loading canary corpus...")
canary_data = torch.load('canary_corpus.pt', weights_only=False)
with open('rag-experiment/canary_rag.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    canary_vectors = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
canary_index = torch.from_numpy(canary_vectors.copy()).to(device, dtype=torch.bfloat16)

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

# Load model
print(f'Loading {MODEL_NAME}...')
model, tokenizer = load_refiner_model(MODEL_NAME, split_layer=SPLIT, num_revolutions=REVS)
model = model.to(device)
for param in model.base.parameters(): param.requires_grad = False

model.rag_scale = torch.nn.Parameter(torch.tensor(1.0).to(device, dtype=torch.bfloat16)) # Start with stronger scale

def trust_forward(input_ids, target_mask, injection_idxs):
    bs, seq = input_ids.shape
    hidden = model.embed_tokens(input_ids)
    pos_ids = torch.arange(seq, device=device).unsqueeze(0).expand(bs, -1)
    pos_emb = model.rotary_emb(hidden, pos_ids)

    for i in range(SPLIT):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    # Inject specific canary vectors
    injections = canary_index[injection_idxs] # [bs, 2048]
    hidden = hidden + torch.sigmoid(model.rag_scale) * injections.unsqueeze(1)

    for rev in range(REVS):
        hidden = model.refiner(hidden, rev)

    for i in range(SPLIT, len(model.layers)):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    hidden = model.norm(hidden)
    logits = model.lm_head(hidden)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = target_mask[:, 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)
    
    flat_labels[flat_mask == 0] = -100
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    return loss

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
                bs, seq = generated.shape
                h = model.embed_tokens(generated)
                p_ids = torch.arange(seq, device=device).unsqueeze(0).expand(bs, -1)
                p_emb = model.rotary_emb(h, p_ids)
                for li in range(SPLIT):
                    out = model.layers[li](h, position_embeddings=p_emb)
                    h = out[0] if isinstance(out, tuple) else out
                
                inj = canary_index[i:i+1]
                h = h + torch.sigmoid(model.rag_scale) * inj.unsqueeze(1)
                
                for rev in range(REVS):
                    h = model.refiner(h, rev)
                for li in range(SPLIT, len(model.layers)):
                    out = model.layers[li](h, position_embeddings=p_emb)
                    h = out[0] if isinstance(out, tuple) else out
                h = model.norm(h)
                logits = model.lm_head(h)
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id: break
                
            print(f"Prompt: {prompt_text}")
            print(f"Generated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
            print("-" * 30)

# Baseline
validate("Baseline (Before Training)")

dataset = TrustDataset(canary_data)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=trust_collate)

# Train refiner + rag_scale
params = list(model.refiner.parameters()) + [model.rag_scale]
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01) # Higher LR for fast trust

print("\nStarting Trust Training (Canary Facts)...")
for epoch in range(100):
    model.train(); model.base.eval()
    epoch_loss = 0; steps = 0; t0 = time.time()
    for input_ids, target_mask, injection_idxs in loader:
        input_ids = input_ids.to(device)
        target_mask = target_mask.to(device)
        injection_idxs = injection_idxs.to(device)
        
        loss = trust_forward(input_ids, target_mask, injection_idxs)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        epoch_loss += loss.item(); steps += 1
    
    avg_loss = epoch_loss / steps
    if (epoch + 1) % 10 == 0 or avg_loss < 0.05:
        gate = torch.sigmoid(model.refiner.gate).item()
        rag = torch.sigmoid(model.rag_scale).item()
        print(f"Epoch {epoch+1} | loss: {avg_loss:.4f} | gate: {gate:.4f} | rag: {rag:.4f} | time: {time.time()-t0:.2f}s")
    if avg_loss < 0.001:
        print("Loss threshold reached!")
        break

# Final Validation
validate("Final (After Training)")

# Save the "trusted" refiner
os.makedirs('checkpoints-trust', exist_ok=True)
torch.save({'refiner_state_dict': model.refiner.state_dict(), 'rag_scale': model.rag_scale.item()}, 'checkpoints-trust/trusted_refiner.pt')
