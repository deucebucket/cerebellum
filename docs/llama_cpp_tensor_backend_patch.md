# llama.cpp Per-Tensor Backend Override Patch

## Summary
Add `--tensor-backend-file` flag that assigns individual tensors to specific
compute backends (CUDA, CPU) independently of their layer's `-ngl` assignment.

## Motivation
Current `-ngl` assigns ALL tensors in a layer to the same backend. Our ablation
data shows that within the same layer, tensors have wildly different sensitivity:
- `blk.63.ffn_down.weight` — sacred (+0.138 PPL delta when crushed)
- `blk.63.ffn_gate.weight` — neutral at same layer

With per-tensor override:
- Offload a layer to CPU but keep its sacred ffn_down on GPU
- Keep a layer on GPU but demote its safe tensors to CPU (save VRAM)

## File Format
```
# tensor-backend-file: one override per line
# format: tensor_name=backend[,priority]
# backend: cuda, cpu
# priority (optional): pin (never evict), normal (default), low (evict first)
blk.63.ffn_down.weight=cuda,pin
blk.63.attn_output.weight=cuda
blk.0.ffn_gate.weight=cpu
blk.2.ffn_gate.weight=cpu
```

## Implementation

### 1. Parse tensor-backend-file (new file: `src/llama-tensor-backend.h/cpp`)

```cpp
#include <string>
#include <unordered_map>

struct tensor_backend_override {
    std::string backend;  // "cuda" or "cpu"
    std::string priority; // "pin", "normal", "low"
};

using tensor_backend_map = std::unordered_map<std::string, tensor_backend_override>;

tensor_backend_map parse_tensor_backend_file(const std::string & path);
```

Parser: same pattern as existing `parse_tensor_type_file()` in `llama-quant.cpp`.

### 2. Thread into model params (modify `llama.h`)

```cpp
struct llama_model_params {
    // ... existing fields ...
    const char * tensor_backend_file; // NEW: per-tensor backend override
};
```

### 3. Override in create_tensor (modify `src/llama-model.cpp` ~line 2727)

Current code:
```cpp
auto create_tensor = [&](const LLM_TN_IMPL & tn, 
                          const std::initializer_list<int64_t> & ne, 
                          int flags) -> ggml_tensor * {
    const buft_list_t * buft_list_layer = tn.bid == -1 
        ? nullptr 
        : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, 
        pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, 
        buft_list_layer, tn, ne, flags);
};
```

Modified code:
```cpp
auto create_tensor = [&](const LLM_TN_IMPL & tn,
                          const std::initializer_list<int64_t> & ne,
                          int flags) -> ggml_tensor * {
    const buft_list_t * buft_list_layer;
    
    // Check per-tensor backend override
    auto it = tensor_backend_overrides.find(tn.str());
    if (it != tensor_backend_overrides.end()) {
        if (it->second.backend == "cuda" && !pimpl->gpu_buft_list.empty()) {
            buft_list_layer = &pimpl->gpu_buft_list.begin()->second;
        } else {
            buft_list_layer = &pimpl->cpu_buft_list;
        }
    } else {
        // Default: use layer assignment from -ngl
        buft_list_layer = tn.bid == -1 
            ? nullptr 
            : pimpl->dev_layer.at(tn.bid).buft_list;
    }
    
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list,
        pimpl->dev_input.buft_list, pimpl->dev_output.buft_list,
        buft_list_layer, tn, ne, flags);
};
```

### 4. CLI flag (modify `common/arg.cpp`)

Add alongside existing `--tensor-type`:
```cpp
add_opt(llama_arg(
    {"--tensor-backend-file"}, "FNAME",
    "per-tensor backend assignment file",
    [](common_params & params, const std::string & value) {
        params.tensor_backend_file = value;
    }
));
```

### 5. Memory accounting

Update the memory breakdown print to show per-backend totals accounting
for overrides. Currently it just reports layer-level GPU/CPU split.

## Files to modify
1. `include/llama.h` — add `tensor_backend_file` to `llama_model_params`
2. `src/llama-model.cpp` — override logic in `create_tensor`, parse file in `load_tensors`
3. `common/arg.cpp` — CLI flag
4. `common/common.h` — add field to `common_params`
5. NEW `src/llama-tensor-backend.h/cpp` — parser

## Estimated scope
~200-300 lines of new C++ code. No changes to ggml backend infrastructure.
The override works at the buffer-type selection level, which already supports
per-tensor assignment — it's just never been exposed.

## Testing
1. Verify that `--tensor-backend-file` with all tensors set to `cuda` matches
   behavior of `-ngl 99`
2. Verify that setting one tensor to `cpu` while its layer is on GPU actually
   places it on CPU (check memory breakdown)
3. Perplexity comparison: same model with/without overrides should be identical
   (backends don't affect math, only placement)

## Future: Double-buffer Prefetch (Phase 2)
Once per-tensor backend works, add prefetch scheduling:
- Allocate 2 VRAM scratch buffers sized for largest CPU tensor
- Before computing layer N, DMA layer N+1's CPU tensors into scratch buffer
- GPU reads from scratch buffer instead of waiting for PCIe transfer
- This is the "piano" — tensors flow through VRAM like notes through time

This requires deeper ggml_backend_sched changes and is a separate PR.
