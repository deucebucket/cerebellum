"""Correlate osmosis imatrix per-tensor importance against cerebellum ablation PPL deltas.

Question: does the osmosis imatrix already tell us which tensors are sensitive,
so future builds can skip the ablation sweep?
"""
import json
import struct
import sys
from pathlib import Path
from statistics import mean


def parse_imatrix(path):
    """Parse llama.cpp legacy imatrix format. Returns {tensor_name: [floats]}."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    (n_entries,) = struct.unpack_from("<i", data, pos); pos += 4
    out = {}
    for _ in range(n_entries):
        (name_len,) = struct.unpack_from("<i", data, pos); pos += 4
        name = data[pos:pos + name_len].decode("utf-8"); pos += name_len
        (ncall,) = struct.unpack_from("<i", data, pos); pos += 4
        (nval,) = struct.unpack_from("<i", data, pos); pos += 4
        values = list(struct.unpack_from(f"<{nval}f", data, pos))
        pos += 4 * nval
        # Some formats store values * ncall — undo that for raw activations
        if ncall > 0:
            values = [v / ncall for v in values]
        out[name] = values
    return out


def per_tensor_importance(values):
    """Aggregate per-column imatrix values into a single per-tensor scalar.

    The osmosis imatrix already encodes L2*maxabs*variance per column, normalized 0-1.
    Mean across columns gives overall tensor importance.
    """
    if not values:
        return 0.0
    return mean(values)


def spearman(xs, ys):
    """Spearman rank correlation. No scipy."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(seq):
        idx = sorted(range(n), key=lambda i: seq[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and seq[idx[j + 1]] == seq[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def analyze(model_label, imatrix_path, ablation_path):
    print(f"\n{'=' * 70}")
    print(f"  {model_label}")
    print(f"{'=' * 70}")
    print(f"imatrix : {imatrix_path}")
    print(f"ablation: {ablation_path}")

    imatrix = parse_imatrix(imatrix_path)
    ablation = json.load(open(ablation_path))
    baseline = ablation["baseline_ppl"]
    tests = ablation["tests"]
    print(f"\nimatrix tensors: {len(imatrix)}")
    print(f"ablation tests : {len(tests)} (baseline PPL {baseline:.4f})")

    # Match ablated tensors (HF names) to imatrix entries (gguf names).
    # Some ablation files store gguf names without the `.weight` suffix.
    paired = []  # list of (hf_name, gguf_name, imatrix_score, ppl_delta)
    missing = []
    for hf_name, t in tests.items():
        gguf = t.get("gguf_tensor")
        ppl = t.get("ppl")
        if gguf is None or ppl is None:
            continue
        delta = ppl - baseline  # positive = sensitive (crushing hurt)
        key = gguf if gguf in imatrix else (gguf + ".weight" if (gguf + ".weight") in imatrix else None)
        if key:
            score = per_tensor_importance(imatrix[key])
            paired.append((hf_name, key, score, delta))
        else:
            missing.append((hf_name, gguf))

    print(f"matched        : {len(paired)} / {len(tests)}")
    if missing:
        print(f"missing in imatrix: {len(missing)}")
        for hf, gg in missing[:5]:
            print(f"    {hf} -> {gg}")
        if len(missing) > 5:
            print(f"    ... ({len(missing) - 5} more)")

    if len(paired) < 2:
        print("not enough overlap to correlate")
        return

    xs = [p[2] for p in paired]  # imatrix importance
    ys = [p[3] for p in paired]  # ppl delta

    print(f"\nimatrix importance range: [{min(xs):.4g}, {max(xs):.4g}]")
    print(f"PPL delta range         : [{min(ys):+.4f}, {max(ys):+.4f}]")

    print(f"\nSpearman ρ (imatrix vs PPL delta): {spearman(xs, ys):+.3f}")
    print(f"Pearson  r (imatrix vs PPL delta): {pearson(xs, ys):+.3f}")
    print("(positive = high imatrix importance ↔ high PPL damage when crushed)")

    # Top-k agreement: do top-k by imatrix overlap with top-k by ablation?
    print(f"\nTop-{min(10, len(paired))} most-sensitive tensors")
    print(f"{'rank':>4}  {'by ablation (PPL Δ)':<55}  {'by imatrix':<55}")
    by_abl = sorted(paired, key=lambda p: -p[3])
    by_im = sorted(paired, key=lambda p: -p[2])
    for i in range(min(10, len(paired))):
        a = by_abl[i]
        m = by_im[i]
        a_str = f"{a[1]:<35} Δ={a[3]:+.4f}"
        m_str = f"{m[1]:<35} I={m[2]:.4g}"
        print(f"  {i+1:>2}  {a_str:<55}  {m_str:<55}")

    # Set overlap of top quartile
    k = max(1, len(paired) // 4)
    abl_top = {p[1] for p in by_abl[:k]}
    im_top = {p[1] for p in by_im[:k]}
    overlap = abl_top & im_top
    print(f"\nTop-quartile overlap: {len(overlap)}/{k} "
          f"({100 * len(overlap) / k:.0f}% of imatrix top-{k} also in ablation top-{k})")

    # How well does imatrix flag the actually-harmful tensors?
    # "Harmful" = positive delta; "safe" = negative delta (demoting helped)
    harmful = [p for p in paired if p[3] > 0]
    safe = [p for p in paired if p[3] <= 0]
    if harmful and safe:
        h_im = mean(p[2] for p in harmful)
        s_im = mean(p[2] for p in safe)
        print(f"\nMean imatrix importance — harmful tensors (Δ>0): {h_im:.4g} ({len(harmful)} tensors)")
        print(f"Mean imatrix importance — safe tensors    (Δ≤0): {s_im:.4g} ({len(safe)} tensors)")
        print(f"Ratio (harmful / safe): {h_im / s_im if s_im else float('inf'):.2f}x")


def main():
    root = Path("/var/home/deucebucket/ai-drive/cerebellum")
    cases = [
        ("Qwen 3.6 27B",
         root / "osmosis-qwen36-27b/cerebellum_imatrix.dat",
         root / "osmosis-qwen36-27b/ablation_results.json"),
        ("Gemma 4 26B-A4B",
         root / "osmosis-gemma4-26b/imatrix.dat",
         None),  # gemma4-26b has group-level ablation, not unified results.json — handle separately
        ("Gemma 4 E4B",
         root / "osmosis-gemma4-e4b/imatrix.dat",
         root / "osmosis-gemma4-e4b/ablation/ablation_results.json"),
        ("Qwen 3.5 9B (n=202)",
         root / "osmosis-qwen35-9b/cerebellum_imatrix.dat",
         root / "osmosis-qwen35-9b/ablation_results.json"),
    ]
    for label, im, abl in cases:
        if abl is None:
            print(f"\n[skipping {label} — group-level ablation, needs different handling]")
            continue
        if not im.exists():
            print(f"\n[skipping {label} — no imatrix at {im}]")
            continue
        if not abl.exists():
            print(f"\n[skipping {label} — no ablation at {abl}]")
            continue
        try:
            analyze(label, im, abl)
        except Exception as e:
            print(f"\n[{label} failed: {e}]")
            raise


if __name__ == "__main__":
    main()
