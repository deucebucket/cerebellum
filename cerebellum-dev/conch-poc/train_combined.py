"""Train refiner with BOTH RAG index and Cartridge V injection simultaneously."""
import torch, os, time, struct, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from refiner import load_refiner_model

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'
SPLIT = 18
REVS = 2
BATCH = 4

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path) as f: text = f.read()
        tokens = tokenizer.encode(text)
        self.examples = [torch.tensor(tokens[i:i+block_size], dtype=torch.long) for i in range(0, len(tokens)-block_size, block_size)]
        print('Dataset: %d chunks' % len(self.examples))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def collate_fn(batch):
    max_len = max(x.size(0) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i,:x.size(0)] = x; masks[i,:x.size(0)] = 1
    return padded, masks

# Load RAG index
with open('rag-experiment/rag_docs_real.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    rag_data = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
rag_index = torch.from_numpy(rag_data.copy()).to(device, dtype=torch.bfloat16)
print('RAG index: %d docs' % rag_index.shape[0])

# Load Cartridge V
with open('rag-experiment/cartridge_v.bin', 'rb') as f:
    hdr = f.read(8); r, c = struct.unpack('ii', hdr)
    cart_data = np.frombuffer(f.read(), dtype=np.float32).reshape(r, c)
cart_v = torch.from_numpy(cart_data.copy()).to(device, dtype=torch.bfloat16)
print('Cartridge V: %d layers x %d dim' % (cart_v.shape[0], cart_v.shape[1]))
n_kv_head = 2
n_head = 16
n_embd_head = 128

# Load model
print('Loading %s...' % MODEL_NAME)
model, tokenizer = load_refiner_model(MODEL_NAME, split_layer=SPLIT, num_revolutions=REVS)
model = model.to(device)
for param in model.base.parameters(): param.requires_grad = False

model.rag_scale = torch.nn.Parameter(torch.tensor(0.5).to(device, dtype=torch.bfloat16))
model.cart_scale = torch.nn.Parameter(torch.tensor(0.5).to(device, dtype=torch.bfloat16))

def combined_forward(input_ids, labels=None, attention_mask=None, fixed_revolutions=None):
    nrev = fixed_revolutions if fixed_revolutions is not None else REVS
    bs, seq = input_ids.shape
    hidden = model.embed_tokens(input_ids)
    pos_ids = torch.arange(seq, device=device).unsqueeze(0).expand(bs, -1)
    pos_emb = model.rotary_emb(hidden, pos_ids)

    for i in range(SPLIT):
        out = model.layers[i](hidden, position_embeddings=pos_emb)
        hidden = out[0] if isinstance(out, tuple) else out

    # RAG + Cartridge at refiner input (layer 17)
    if model.training:
        query = hidden.mean(dim=1)
        sim = torch.matmul(rag_index, query.T)
        top1_idx = sim.argmax(dim=0)
        top1_docs = rag_index[top1_idx]
        hidden = hidden + torch.sigmoid(model.rag_scale) * top1_docs.unsqueeze(1)
        # Cartridge at knowledge gate: inject V vectors directly
        cv = cart_v[30:34].mean(dim=0)  # avg of layers 30-33
        cv = cv.view(2, 128).transpose(0,1).repeat_interleave(8, dim=1).reshape(2048)
        cv = cv.unsqueeze(0).unsqueeze(0).expand(bs, seq, -1)
        hidden = hidden + torch.sigmoid(model.cart_scale) * cv

    # Refiner loops
    for rev in range(nrev):
        hidden = model.refiner(hidden, rev)

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
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
    return {"loss": loss, "logits": logits}

model.forward = combined_forward

# Quick PPL
def qppl(n=8):
    model.eval()
    with torch.no_grad():
        with open('/var/home/deucebucket/games/osmosis-quants/wiki.test.raw') as f: text = f.read()
        tokens = tokenizer.encode(text)
        total, cnt = 0, 0
        for i in range(0, min(len(tokens)-512, 512*n), 512):
            chunk = torch.tensor(tokens[i:i+512], dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk)
            total += out['loss'].item(); cnt += 1
        return torch.exp(torch.tensor(total/cnt)).item()

ppl_base = qppl()
print('Baseline PPL: %.4f' % ppl_base)

# Training data: code corpus
data_path = '/var/home/deucebucket/games/osmosis-quants/wiki.train.raw'
dataset = TextDataset(tokenizer, data_path, block_size=512)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, drop_last=True)

params = list(model.refiner.parameters()) + [model.rag_scale, model.cart_scale]
optimizer = AdamW(params, lr=1e-4, weight_decay=0.1)
os.makedirs('./checkpoints-refiner-combined', exist_ok=True)
best_ppl = ppl_base

print('Training combined refiner (%d params)...' % sum(p.numel() for p in params))
for epoch in range(3):
    model.train(); model.base.eval()
    epoch_loss, steps = 0, 0; t0 = time.time()
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
        out = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
        loss = out['loss']; loss.backward()
        optimizer.step(); optimizer.zero_grad()
        epoch_loss += loss.item(); steps += 1
    avg_loss = epoch_loss/steps
    ppl = qppl()
    gate = torch.sigmoid(model.refiner.gate).item()
    rag = torch.sigmoid(model.rag_scale).item()
    cart = torch.sigmoid(model.cart_scale).item()
    print('Epoch %d/3 | loss=%.4f | PPL=%.4f | gate=%.4f | rag=%.4f | cart=%.4f | %ds' % (epoch+1, avg_loss, ppl, gate, rag, cart, int(time.time()-t0)))
    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({'refiner_state_dict': model.refiner.state_dict(), 'rag_scale': model.rag_scale.item(), 'cart_scale': model.cart_scale.item(), 'epoch': epoch, 'ppl': ppl}, './checkpoints-refiner-combined/best_refiner.pt')
        print('  NEW BEST: %.4f' % ppl)

print('\nBest PPL: %.4f vs baseline %.4f (%.2f%%)' % (best_ppl, ppl_base, 100*(best_ppl-ppl_base)/ppl_base))

# Export weights
ckpt = torch.load('./checkpoints-refiner-combined/best_refiner.pt', map_location='cpu', weights_only=True)
state = ckpt['refiner_state_dict']
out_dir = './brainloop-ggml-weights-combined'
os.makedirs(out_dir, exist_ok=True)
d = 2048
iw = state['attn.in_proj_weight'].float(); ib = state['attn.in_proj_bias'].float()
weights = {
    'refiner_attn_q_weight':iw[:d],'refiner_attn_k_weight':iw[d:2*d],'refiner_attn_v_weight':iw[2*d:],
    'refiner_attn_q_bias':ib[:d],'refiner_attn_k_bias':ib[d:2*d],'refiner_attn_v_bias':ib[2*d:],
    'refiner_attn_output_weight':state['attn.out_proj.weight'].float(),'refiner_attn_output_bias':state['attn.out_proj.bias'].float(),
    'refiner_ffn_up_weight':state['ffn.0.weight'].float(),'refiner_ffn_up_bias':state['ffn.0.bias'].float(),
    'refiner_ffn_down_weight':state['ffn.2.weight'].float(),'refiner_ffn_down_bias':state['ffn.2.bias'].float(),
    'refiner_ln1_weight':state['ln1.weight'].float(),'refiner_ln1_bias':state['ln1.bias'].float(),
    'refiner_ln2_weight':state['ln2.weight'].float(),'refiner_ln2_bias':state['ln2.bias'].float(),
    'refiner_gate':state['gate'].float().reshape(1),'refiner_rev_emb':state['rev_embed.weight'].float(),
}
for name, data in weights.items():
    shape = list(data.shape)
    if len(shape)==0: shape=[1,1]; data=data.reshape(1,1)
    elif len(shape)==1: shape=[1,shape[0]]; data=data.reshape(shape)
    with open(os.path.join(out_dir, name+'.bin'), 'wb') as f:
        f.write(struct.pack('ii', shape[0], shape[1]))
        f.write(data.numpy().tobytes())
print('RAG scale: %.4f (sigmoid: %.4f)' % (ckpt['rag_scale'], 1/(1+np.exp(-ckpt['rag_scale']))))
print('Cart scale: %.4f (sigmoid: %.4f)' % (ckpt['cart_scale'], 1/(1+np.exp(-ckpt['cart_scale']))))
print('Weights exported to %s/' % out_dir)
