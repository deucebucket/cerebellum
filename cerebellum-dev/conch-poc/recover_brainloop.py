import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from refiner_vanilla import patch_model_vanilla
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm

device = torch.device('cuda')
MODEL_NAME = 'Qwen/Qwen2.5-3B'

print("Loading Model for Brainloop Recovery...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = patch_model_vanilla(model)

# Load current (lobotomized) weights
ckpt_path = 'checkpoints-fusion-patched/fused_refiners.pt'
if os.path.exists(ckpt_path):
    print("Loading fused weights...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.model.layers[18].load_state_dict(ckpt['l18'])
    model.model.layers[31].load_state_dict(ckpt['l31'])

model = model.to(device)

# 1. BRAINLOOP RECOVERY: Identity Prior
# Freeze everything except GATES
for name, param in model.named_parameters():
    if 'gate' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Dataset: WikiText
wiki_path = "/var/home/deucebucket/games/osmosis-quants/wiki.train.raw"
with open(wiki_path, "r", encoding="utf-8") as f:
    text = f.read()
tokens = tokenizer.encode(text[:500000]) # Smaller subset for speed
examples = [torch.tensor(tokens[i:i+128], dtype=torch.long) for i in range(0, len(tokens)-128, 128)]
loader = DataLoader(examples, batch_size=4, shuffle=True)

optimizer = AdamW([model.model.layers[18].gate, model.model.layers[31].gate], lr=5e-3)

print("\nStarting Recovery (Gate Closure for Identity Prior)...")
for epoch in range(1):
    model.train()
    for i, input_ids in enumerate(tqdm(loader)):
        input_ids = input_ids.to(device)
        
        # Ensure no injection
        model.model.layers[18].active_injection = None
        model.model.layers[31].active_injection = None
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i+1) % 50 == 0:
            g18 = torch.sigmoid(model.model.layers[18].gate).item()
            g31 = torch.sigmoid(model.model.layers[31].gate).item()
            print(f"Step {i+1} | Loss: {loss.item():.4f} | G18: {g18:.4f} | G31: {g31:.4f}")
        
        if i > 200: break

# 2. SAVE RECOVERY CHECKPOINT
os.makedirs('checkpoints-recovered', exist_ok=True)
save_dict = {
    'l18': model.model.layers[18].state_dict(),
    'l31': model.model.layers[31].state_dict()
}
torch.save(save_dict, 'checkpoints-recovered/recovered_refiners.pt')
print("Done!")
