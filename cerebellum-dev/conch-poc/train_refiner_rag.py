"""Train refiner with inline RAG index. The refiner learns to use injected context."""
import torch, os, time, struct, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner import load_refiner_model

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLIT = 18
REVS = 2
RAG_SCALE = 0.5

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

def compute_ppl(model, tokenizer, text_path, max_chunks=32):
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

# Load RAG coding index
print('Loading RAG index...', flush=True)
with open('rag-experiment/rag_docs_real.bin', 'rb') as f:
    hdr = f.read(8)
    rows, cols = struct.unpack('ii', hdr)
    rag_data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
rag_index = torch.from_numpy(rag_data.copy()).to(device, dtype=torch.bfloat16)
print(f'RAG index: {rag_index.shape[0]} docs x {rag_index.shape[1]} dim', flush=True)

# Load model with refiner
print(f'Loading {MODEL_NAME}...', flush=True)
model, tokenizer = load_refiner_model(MODEL_NAME, split_layer=SPLIT, num_revolutions=REVS)
model = model.to(device)

# Freeze base model
for param in model.base.parameters():
    param.requires_grad = False

# Make rag_scale trainable — attach to model
model.rag_scale = torch.nn.Parameter(torch.tensor(RAG_SCALE).to(device, dtype=torch.bfloat16))

# Store original forward
_original_forward = model.forward

def rag_forward(input_ids, labels=None, attention_mask=None, fixed_revolutions=None):
    """Forward with RAG injection at split point."""
    if fixed_revolutions is not None:
        nrev = fixed_revolutions
    else:
        nrev = model.num_revolutions

    batch_size, seq_len = input_ids.shape
    hidden = model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    pos_emb = model.rotary_emb(hidden, position_ids)

    # Layers before split (no checkpointing, batch=1 is small enough)
    for i in range(SPLIT):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    # RAG injection at split point
    if model.training:
        query = hidden.mean(dim=1)
        sim = torch.matmul(rag_index, query.T)
        top1_idx = sim.argmax(dim=0)
        top1_docs = rag_index[top1_idx]
        hidden = hidden + torch.sigmoid(model.rag_scale) * top1_docs.unsqueeze(1)

    # Refiner
    for rev in range(nrev):
        hidden = model.refiner(hidden, rev)

    # Remaining layers
    for i in range(SPLIT, len(model.layers)):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    hidden = model.norm(hidden)
    logits = model.lm_head(hidden)

    loss = None
    if labels is not None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            shift_labels[shift_mask == 0] = -100
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)

    return {"loss": loss, "logits": logits}

model.forward = rag_forward

# Baseline PPL
ppl_base = compute_ppl(model, tokenizer, '/var/home/deucebucket/games/osmosis-quants/wiki.test.raw', max_chunks=32)
print(f'Baseline PPL: {ppl_base:.4f}', flush=True)

# Dataset
data_path = '/var/home/deucebucket/ai-drive/cerebellum/cerebellum-dev/conch-poc/tool_train.txt'
dataset = TextDataset(tokenizer, data_path, block_size=512)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)

# Optimizer: refiner params + rag_scale
params = list(model.refiner.parameters()) + [model.rag_scale]
optimizer = AdamW(params, lr=1e-4, weight_decay=0.1)

os.makedirs('./checkpoints-refiner-rag', exist_ok=True)
best_ppl = ppl_base

print(f'Training refiner with RAG ({sum(p.numel() for p in params):,} trainable params)...', flush=True)
for epoch in range(3):
    model.train(); model.base.eval()
    epoch_loss = 0; steps = 0; t0 = time.time()

    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
        
        # Use patched forward with RAG injection
        out = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
        loss = out['loss']
        
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        epoch_loss += loss.item(); steps += 1

    avg_loss = epoch_loss/steps
    ppl = compute_ppl(model, tokenizer, '/var/home/deucebucket/games/osmosis-quants/wiki.test.raw', max_chunks=32)
    gate = torch.sigmoid(model.refiner.gate).item()
    rag = torch.sigmoid(model.rag_scale).item()
    print(f'Epoch {epoch+1}/3 | loss={avg_loss:.4f} | PPL={ppl:.4f} | gate={gate:.4f} | rag={rag:.4f} | {time.time()-t0:.0f}s', flush=True)

    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({
            'refiner_state_dict': model.refiner.state_dict(),
            'rag_scale': model.rag_scale.item(),
            'epoch': epoch, 'ppl': ppl,
        }, './checkpoints-refiner-rag/best_refiner.pt')
        print(f'  NEW BEST: {ppl:.4f} (delta={100*(ppl-ppl_base)/ppl_base:+.2f}%)', flush=True)

print(f'\nBest PPL: {best_ppl:.4f} vs baseline {ppl_base:.4f} ({100*(best_ppl-ppl_base)/ppl_base:+.2f}%)', flush=True)

# Export to .bin for C++
import struct as _struct
ckpt = torch.load('./checkpoints-refiner-rag/best_refiner.pt', map_location='cpu', weights_only=True)
state = ckpt['refiner_state_dict']
out_dir = './brainloop-ggml-weights-rag'
os.makedirs(out_dir, exist_ok=True)

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
    'refiner_ln1_bias': state['ln1.bias'].float(),
    'refiner_ln2_weight': state['ln2.weight'].float(),
    'refiner_ln2_bias': state['ln2.bias'].float(),
    'refiner_gate': state['gate'].float().reshape(1),
    'refiner_rev_emb': state['rev_embed.weight'].float(),
}

for name, data in weights.items():
    shape = list(data.shape)
    if len(shape) == 0: shape = [1,1]; data = data.reshape(1,1)
    elif len(shape) == 1: shape = [1, shape[0]]; data = data.reshape(shape)
    fname = os.path.join(out_dir, name + '.bin')
    with open(fname, 'wb') as f:
        f.write(_struct.pack('ii', shape[0], shape[1]))
        f.write(data.numpy().tobytes())
    print(f'  {name}: {shape}')

print(f'\nRAG scale: {ckpt["rag_scale"]:.4f} (sigmoid: {1/(1+torch.exp(-torch.tensor(ckpt["rag_scale"]))).item():.4f})')
print(f'Weights exported to {out_dir}/')
