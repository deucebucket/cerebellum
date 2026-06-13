"""
locate_coding_delta.py

Two-mode delta-vector analysis for HumanEval failures.

Mode 1 --mode scan:
  Forward both knowing/ignorant texts through all 37 hidden_states indices,
  record per-layer delta L2 norm, cosine vs layer-30 delta, and normalised delta.
  Output: locate_results_scan.json + printed per-layer table.

Mode 2 --mode patch:
  Hook model.model.layers[L] to inject scale * delta at last-token position
  during greedy generation, write patched completions to jsonl files.
  Output: locate_results_patch.json + humaneval_samples_patched_L{layer}.jsonl

Device: cuda (bf16). CPU-safe to import; model loads only when script runs.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
EVAL_RESULTS = "humaneval_samples_baseline_eval_results.json"
NUM_HIDDEN = 37  # 36 layers + embedding = indices 0..36

CONTROL_NONE = "none"
CONTROL_RANDOM = "random"
CONTROL_SHUFFLED = "shuffled"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_results(path: str):
    """Return (failed_ids, passed_ids) lists from evalplus detailed JSON.

    Schema observed:
      {
        "date": "...",
        "hash": "...",
        "eval": {
          "HumanEval/N": [
            {
              "task_id": "HumanEval/N",
              "solution": "...",
              "base_status": "pass"|"fail",
              "plus_status": "pass"|"fail",
              "base_fail_tests": [...],
              "plus_fail_tests": [...]
            }
          ],
          ...
        }
      }
    Pass = base_status == "pass" for the first entry in the list.
    """
    with open(path) as f:
        data = json.load(f)
    eval_data = data["eval"]
    failed, passed = [], []
    for task_id, entries in eval_data.items():
        status = entries[0]["base_status"]
        if status == "pass":
            passed.append(task_id)
        else:
            failed.append(task_id)
    return failed, passed


def build_texts(prompt: str, canonical_solution: str, tokenizer):
    """Return (knowing_text, ignorant_text) with space-padded ignorant prefix.

    Technique from run_13k_preparation.py lines 89-93:
    the ignorant text has the reference block replaced by spaces of equal
    token length so absolute positions stay aligned.
    """
    reference_block = (
        f"Here is a correct reference solution:\n```python\n{prompt}{canonical_solution}```\n\n"
    )
    knowing_text = reference_block + f"Solve this Python coding problem:\n{prompt}\n"

    # Token-length-match: encode reference block, replace with that many spaces
    ref_tokens = tokenizer.encode(reference_block)
    prefix = " " * (len(ref_tokens) - 1)
    ignorant_text = prefix + f"Solve this Python coding problem:\n{prompt}\n"

    return knowing_text, ignorant_text


def get_all_hidden_states(text: str, model, tokenizer, device):
    """Return tensor [37, hidden] - last-token hidden state at every layer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # out.hidden_states is a tuple of (batch, seq, hidden) tensors, length 37
    last_tok = torch.stack([h[0, -1, :] for h in out.hidden_states])  # [37, H]
    return last_tok.cpu()


def extract_code(text: str) -> str:
    """Extract python code block, mirroring bench_humaneval.py."""
    marker = "```python"
    if marker in text:
        return text.split(marker)[1].split("```")[0]
    if "```" in text:
        return text.split("```")[1].split("```")[0]
    return text


# ---------------------------------------------------------------------------
# Mode 1: scan
# ---------------------------------------------------------------------------

def run_scan(args, model, tokenizer, device, dataset, failed_ids, passed_ids):
    n_fail = min(args.n, len(failed_ids))
    n_pass = 5
    selected_fail = failed_ids[:n_fail]
    selected_pass = passed_ids[:n_pass]

    results = {"failed": {}, "passed": {}, "per_layer_summary": {}}

    # Accumulate per-layer stats
    fail_norms   = [[] for _ in range(NUM_HIDDEN)]
    pass_norms   = [[] for _ in range(NUM_HIDDEN)]
    fail_normed  = [[] for _ in range(NUM_HIDDEN)]
    pass_normed  = [[] for _ in range(NUM_HIDDEN)]
    fail_cosine  = [[] for _ in range(NUM_HIDDEN)]
    pass_cosine  = [[] for _ in range(NUM_HIDDEN)]

    def process_group(task_ids, group_name, norm_lists, normed_lists, cosine_lists):
        for task_id in task_ids:
            problem = dataset[task_id]
            prompt = problem["prompt"]
            canonical = problem["canonical_solution"]

            print(f"  [{group_name}] {task_id} ...", flush=True)
            knowing_text, ignorant_text = build_texts(prompt, canonical, tokenizer)

            h_know = get_all_hidden_states(knowing_text, model, tokenizer, device)
            h_ignor = get_all_hidden_states(ignorant_text, model, tokenizer, device)

            deltas = h_know - h_ignor  # [37, H]

            # Reference: layer-30 delta for cosine comparison
            ref_delta = deltas[30]
            ref_norm = ref_delta.norm().item()

            task_entry = {}
            for li in range(NUM_HIDDEN):
                d = deltas[li]
                l2 = d.norm().item()
                h_l2 = h_know[li].norm().item()
                normalized = l2 / (h_l2 + 1e-9)

                if ref_norm > 1e-9 and l2 > 1e-9:
                    cos = torch.nn.functional.cosine_similarity(
                        d.unsqueeze(0), ref_delta.unsqueeze(0)
                    ).item()
                else:
                    cos = 0.0

                norm_lists[li].append(l2)
                normed_lists[li].append(normalized)
                cosine_lists[li].append(cos)
                task_entry[li] = {
                    "delta_l2": round(l2, 5),
                    "delta_cos_vs_l30": round(cos, 5),
                    "normalized_delta": round(normalized, 5),
                }

            if group_name == "failed":
                results["failed"][task_id] = task_entry
            else:
                results["passed"][task_id] = task_entry

    print("Scanning failed problems...", flush=True)
    process_group(selected_fail, "failed", fail_norms, fail_normed, fail_cosine)
    print("Scanning passed controls...", flush=True)
    process_group(selected_pass, "passed", pass_norms, pass_normed, pass_cosine)

    # Build and print summary
    print(
        "\nPer-layer summary (mean over problems)\n"
        f"{'Layer':>6}  {'fail_L2':>10}  {'pass_L2':>10}"
        f"  {'fail_norm':>10}  {'pass_norm':>10}"
        f"  {'fail_cos30':>10}  {'pass_cos30':>10}"
    )
    for li in range(NUM_HIDDEN):
        fl2 = sum(fail_norms[li])  / max(len(fail_norms[li]),  1)
        pl2 = sum(pass_norms[li])  / max(len(pass_norms[li]),  1)
        fn  = sum(fail_normed[li]) / max(len(fail_normed[li]), 1)
        pn  = sum(pass_normed[li]) / max(len(pass_normed[li]), 1)
        fc  = sum(fail_cosine[li]) / max(len(fail_cosine[li]), 1)
        pc  = sum(pass_cosine[li]) / max(len(pass_cosine[li]), 1)
        results["per_layer_summary"][li] = {
            "fail_mean_l2": round(fl2, 5),
            "pass_mean_l2": round(pl2, 5),
            "fail_mean_normalized": round(fn, 5),
            "pass_mean_normalized": round(pn, 5),
            "fail_mean_cos_vs_l30": round(fc, 5),
            "pass_mean_cos_vs_l30": round(pc, 5),
        }
        print(
            f"{li:>6}  {fl2:>10.4f}  {pl2:>10.4f}"
            f"  {fn:>10.4f}  {pn:>10.4f}"
            f"  {fc:>10.4f}  {pc:>10.4f}"
        )

    out_path = "locate_results_scan.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


# ---------------------------------------------------------------------------
# Mode 2: patch
# ---------------------------------------------------------------------------

def _apply_control(deltas_list, layer_id, problem_idx, control, rng):
    """Return a (possibly modified) delta vector [H] on CPU, fp32.

    deltas_list : list of [37, H] tensors, one per selected problem (CPU, fp32)
    problem_idx : index of the current problem within that list
    layer_id    : which layer index to pull from
    control     : CONTROL_NONE | CONTROL_RANDOM | CONTROL_SHUFFLED
    rng         : torch.Generator (CPU) for reproducible random draws
    """
    real_delta = deltas_list[problem_idx][layer_id]  # [H], CPU fp32

    if control == CONTROL_NONE:
        return real_delta

    real_norm = real_delta.norm()

    if control == CONTROL_RANDOM:
        rand_vec = torch.randn(real_delta.shape, generator=rng)
        rand_norm = rand_vec.norm()
        if rand_norm > 1e-9:
            rand_vec = rand_vec * (real_norm / rand_norm)
        return rand_vec

    if control == CONTROL_SHUFFLED:
        # Rotate by one position: problem i uses delta from problem (i+1) % k
        src_idx = (problem_idx + 1) % len(deltas_list)
        src_delta = deltas_list[src_idx][layer_id]
        return src_delta

    raise ValueError(f"Unknown control: {control!r}")


def run_patch(args, model, tokenizer, device, dataset, failed_ids):
    k = min(args.k, len(failed_ids))
    selected = failed_ids[:k]
    layer_ids = [int(x) for x in args.layers.split(",")]
    scale = args.scale
    control = args.control

    instruct_tmpl = (
        "<|im_start|>user\nSolve this Python coding problem:\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    results = {}  # task_id -> {layer_id -> entry}

    # Determine output filename suffix
    ctrl_tag = f"_{control}" if control != CONTROL_NONE else ""

    # Open per-layer jsonl writers
    jsonl_handles = {
        li: open(f"humaneval_samples_patched_L{li}{ctrl_tag}.jsonl", "w")
        for li in layer_ids
    }

    # Pre-compute all deltas so shuffled control can rotate across problems
    print("Pre-computing deltas for all selected problems ...", flush=True)
    all_deltas = []  # list of [37, H] CPU fp32 tensors
    for task_id in selected:
        problem = dataset[task_id]
        knowing_text, ignorant_text = build_texts(
            problem["prompt"], problem["canonical_solution"], tokenizer
        )
        h_know = get_all_hidden_states(knowing_text, model, tokenizer, device)
        h_ignor = get_all_hidden_states(ignorant_text, model, tokenizer, device)
        all_deltas.append((h_know - h_ignor).float())  # keep fp32 on CPU

    # One Generator per run; seeded once so per-problem/per-layer draws are
    # deterministic regardless of iteration order.
    rng = torch.Generator()
    rng.manual_seed(args.seed)

    for prob_idx, task_id in enumerate(selected):
        problem = dataset[task_id]
        prompt = problem["prompt"]
        results[task_id] = {}

        print(f"\n[patch/{control}] {task_id}", flush=True)

        for layer_id in layer_ids:
            controlled = _apply_control(all_deltas, layer_id, prob_idx, control, rng)
            delta_vec = controlled.to(device, dtype=torch.bfloat16)  # [H]

            # Build instruct prompt
            gen_text = instruct_tmpl.format(prompt=prompt)
            inputs = tokenizer(gen_text, return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[1]

            # Hook: add scale*delta at last-token position on every forward pass
            # during generation. HF greedy decode calls forward one token at a
            # time so [:, -1, :] is always the current last position.
            def make_hook(dv, sc):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[:, -1, :] = hs[:, -1, :] + sc * dv
                        return (hs,) + output[1:]
                    out = output.clone()
                    out[:, -1, :] = out[:, -1, :] + sc * dv
                    return out
                return hook_fn

            hook_handle = model.model.layers[layer_id].register_forward_hook(
                make_hook(delta_vec, scale)
            )

            try:
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            finally:
                hook_handle.remove()

            generated = tokenizer.decode(
                out_ids[0][input_len:], skip_special_tokens=True
            )
            code = extract_code(generated)

            # Attempt lightweight local scoring via evalplus internals.
            # This is optional; full scoring happens via the jsonl outputs.
            passed_local = None
            try:
                from evalplus.evaluate import check_solution  # may not be exported
                passed_local = check_solution(task_id, code)
            except Exception:
                passed_local = None  # rely on external evalplus run

            result_entry = {
                "completion": code,
                "patched_layer": layer_id,
                "scale": scale,
                "control": control,
                "passed_local": passed_local,
            }
            results[task_id][layer_id] = result_entry

            jsonl_handles[layer_id].write(
                json.dumps({"task_id": task_id, "completion": code}) + "\n"
            )
            jsonl_handles[layer_id].flush()

            flip_str = (
                f"pass={passed_local}" if passed_local is not None else "unscored"
            )
            print(
                f"  layer={layer_id:>2}  code_len={len(code):>5}  {flip_str}",
                flush=True,
            )

    for fh in jsonl_handles.values():
        fh.close()

    # Serialise with str keys for JSON compatibility
    serialisable = {
        tid: {str(li): v for li, v in layers.items()}
        for tid, layers in results.items()
    }
    out_path = f"locate_results_patch{ctrl_tag}.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nSaved {out_path}")
    jsonl_names = ", ".join(
        f"humaneval_samples_patched_L{li}{ctrl_tag}.jsonl" for li in layer_ids
    )
    print(f"Per-layer completion jsonls: {jsonl_names}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Locate coding delta vectors in Qwen2.5-3B"
    )
    parser.add_argument(
        "--mode", choices=["scan", "patch"], required=True,
        help="scan=layer brain scan, patch=causal injection test"
    )
    parser.add_argument(
        "--n", type=int, default=15,
        help="(scan) number of failed problems to analyse"
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help="(patch) number of failed problems to patch"
    )
    parser.add_argument(
        "--layers", type=str, default="6,10,14,18,22,26,30,34",
        help="(patch) comma-separated layer indices to inject"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="(patch) delta injection scale factor"
    )
    parser.add_argument(
        "--control", choices=[CONTROL_NONE, CONTROL_RANDOM, CONTROL_SHUFFLED],
        default=CONTROL_NONE,
        help=(
            "(patch) control condition: "
            "'none' = real delta (default); "
            "'random' = Gaussian noise rescaled to real delta L2 norm per problem/layer; "
            "'shuffled' = real delta from the next problem in the list (rotated by 1)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="(patch/random) RNG seed for reproducible random control vectors"
    )
    parser.add_argument(
        "--eval-results", type=str, default=EVAL_RESULTS,
        help="Path to evalplus detailed results JSON"
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"Loading eval results from {args.eval_results} ...")
    failed_ids, passed_ids = load_eval_results(args.eval_results)
    print(f"  {len(failed_ids)} failed, {len(passed_ids)} passed")

    print("Loading evalplus dataset ...")
    from evalplus.data import get_human_eval_plus
    dataset = get_human_eval_plus()

    print(f"Loading {MODEL_NAME} (bf16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if args.mode == "scan":
        run_scan(args, model, tokenizer, device, dataset, failed_ids, passed_ids)
    else:
        run_patch(args, model, tokenizer, device, dataset, failed_ids)


if __name__ == "__main__":
    main()
