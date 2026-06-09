"""Retrain refiner with frozen norm (uses model's attn_norm, not learnable).
Produces weights compatible with C++ port that uses model's GPU norm."""
import torch, sys, os, time, struct, numpy as np
sys.stdout.reconfigure(line_buffering=True)
torch.set_float32_matmul_precision('high')

from refiner import load_refiner_model, RefinerBlock, ConchRefinerModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import autocast

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(torch.tensor(tokens[i:i+block_size], dtype=torch.long))
        print(f'Dataset: {len(self.examples)} chunks of {block_size} tokens')

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i,:x.size(0)] = x
        masks[i,:x.size(0)] = 1
    return padded, masks

def compute_ppl(model, tokenizer, text_path, device, max_chunks=32):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    total_loss = 0; n = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens)-512, 512):
            chunk = torch.tensor(tokens[i:i+512], dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk)
            total_loss += out['loss'].item(); n += 1
            if n >= max_chunks: break
    return torch.exp(torch.tensor(total_loss/n)).item()

print(f'Loading {MODEL_NAME}...', flush=True)
model, tokenizer = load_refiner_model(MODEL_NAME, split_layer=18, num_revolutions=2)
model = model.to(device)

# FREEZE the refiner norm weights (use model's attn_norm at split layer instead)
# Copy model's attn_norm values to refiner's ln1
with torch.no_grad():
    base_ln = model.base.model.layers[18].self_attn.v_proj  # wrong tensor, find attn_norm
    # Actually get the norm from the base model
    for name, param in model.base.model.layers[18].named_parameters():
        if 'input_layernorm' in name or 'attn_norm' in name:
            print(f'  Found base norm: {name}, shape={param.shape}')
            # Copy to refiner ln1
            model.refiner.ln1.weight.data.copy_(param.data)
            break
    
    # Also freeze ln1 and ln2
    for p in model.refiner.ln1.parameters():
        p.requires_grad = False
    for p in model.refiner.ln2.parameters():
        p.requires_grad = False

# torch.compile for speed
print('Compiling...', flush=True)
try:
    model = torch.compile(model, mode='reduce-overhead')
    print('  compiled OK', flush=True)
except Exception as e:
    print(f'  compile failed: {e}', flush=True)

# Baseline PPL
print('Baseline PPL...', flush=True)
ppl_base = compute_ppl(model, tokenizer, '/var/home/deucebucket/games/osmosis-quants/wiki.test.raw', device)
print(f'  PPL = {ppl_base:.4f}', flush=True)

# Dataset
data_path = '/var/home/deucebucket/games/osmosis-quants/wiki.train.raw'
dataset = TextDataset(tokenizer, data_path, block_size=512)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)

optimizer = AdamW([p for p in model.refiner.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.1)
os.makedirs('./checkpoints-refiner-v5-modelnorm', exist_ok=True)
best_ppl = ppl_base

print(f'Training ({sum(p.numel() for p in model.refiner.parameters() if p.requires_grad):,} trainable params)...', flush=True)
for epoch in range(3):
    model.train(); model.base.eval()
    epoch_loss = 0; steps = 0; t0 = time.time()
    
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out['loss']; loss.backward()
        optimizer.step(); optimizer.zero_grad()
        epoch_loss += loss.item(); steps += 1
    
    avg_loss = epoch_loss/steps
    ppl = compute_ppl(model, tokenizer, '/var/home/deucebucket/games/osmosis-quants/wiki.test.raw', device)
    gate = torch.sigmoid(model.refiner.gate).item()
    print(f'Epoch {epoch+1}/3 | loss={avg_loss:.4f} | PPL={ppl:.4f} | gate={gate:.4f} | {time.time()-t0:.0f}s', flush=True)
    
    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({'refiner_state_dict': model.refiner.state_dict(), 'epoch': epoch, 'ppl': ppl},
            './checkpoints-refiner-v5-modelnorm/best_refiner.pt')
        print(f'  NEW BEST: {ppl:.4f} (delta={100*(ppl-ppl_base)/ppl_base:+.2f}%)', flush=True)

print(f'\nBest PPL: {best_ppl:.4f} vs baseline {ppl_base:.4f} ({100*(best_ppl-ppl_base)/ppl_base:+.2f}%)', flush=True)

# Export to .bin files for C++
ckpt = torch.load('./checkpoints-refiner-v5-modelnorm/best_refiner.pt', map_location='cpu')
state = ckpt['refiner_state_dict']
out_dir = './brainloop-ggml-weights-v5'
os.makedirs(out_dir, exist_ok=True)

# Split QKV from in_proj_weight
embed_dim = 2048
iw = state['attn.in_proj_weight'].float()
ib = state['attn.in_proj_bias'].float()
weights = {
    'refiner_attn_q_weight': iw[:embed_dim],
    'refiner_attn_k_weight': iw[embed_dim:2*embed_dim],
    'refiner_attn_v_weight': iw[2*embed_dim:],
    'refiner_attn_q_bias': ib[:embed_dim],
    'refiner_attn_k_bias': ib[embed_dim:2*embed_dim],
    'refiner_attn_v_bias': ib[2*embed_dim:],
    'refiner_attn_output_weight': state['attn.out_proj.weight'].float(),
    'refiner_attn_output_bias': state['attn.out_proj.bias'].float(),
    'refiner_ffn_up_weight': state['ffn.0.weight'].float(),
    'refiner_ffn_up_bias': state['ffn.0.bias'].float(),
    'refiner_ffn_down_weight': state['ffn.2.weight'].float(),
    'refiner_ffn_down_bias': state['ffn.2.bias'].float(),
    'refiner_ln1_weight': state['ln1.weight'].float(),
    'refiner_ln2_weight': state['ln2.weight'].float(),
    'refiner_gate': state['gate'].float(),
}

for name, data in weights.items():
    shape = list(data.shape)
    if len(shape) == 0: shape = [1, 1]; data = data.reshape(1,1)
    elif len(shape) == 1: shape = [1, shape[0]]; data = data.reshape(shape)
    fname = os.path.join(out_dir, name.replace('.', '_') + '.bin')
    with open(fname, 'wb') as f:
        f.write(struct.pack('ii', shape[0], shape[1]))
        f.write(data.numpy().tobytes())
    print(f'  {name}: {shape}')

print(f'Weights exported to {out_dir}/')
