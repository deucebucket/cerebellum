# Cerebellum Ablation Analysis

- input: `cerebellum-dev/sparse-upcycling/runs/moe_ablation_routed_q4km_to_q3.json`
- rows: 3 tested, 0 skipped
- overrides: 0 tensors
- counts: {'demotable': 0, 'tolerant': 1, 'sacred': 2, 'critical': 0}

| class | tensor | weighted | worst | worst domain |
| --- | --- | ---: | ---: | --- |
| sacred | `blk.0.ffn_down_exps.weight` | +2.313% | +4.050% | reasoning |
| sacred | `blk.1.ffn_gate_up_exps.weight` | +1.769% | +2.344% | reasoning |
| tolerant | `blk.0.ffn_gate_up_exps.weight` | +1.012% | +1.341% | reasoning |
