# Cerebellum Ablation Analysis

- input: `cerebellum-dev/sparse-upcycling/runs/moe_ablation_smoke_q4km_to_q3.json`
- rows: 8 tested, 0 skipped
- overrides: 4 tensors
- counts: {'demotable': 4, 'tolerant': 3, 'sacred': 1, 'critical': 0}

| class | tensor | weighted | worst | worst domain |
| --- | --- | ---: | ---: | --- |
| sacred | `blk.0.ffn_down_exps.weight` | +2.313% | +4.050% | reasoning |
| tolerant | `blk.0.ffn_gate_up_exps.weight` | +1.012% | +1.341% | reasoning |
| tolerant | `blk.11.ffn_down_exps.weight` | +0.414% | +1.067% | reasoning |
| tolerant | `blk.11.ffn_down_shexp.weight` | +0.249% | +1.024% | chat |
| demotable | `blk.11.ffn_gate_up_exps.weight` | -0.038% | +0.685% | chat |
| demotable | `blk.31.ffn_down_exps.weight` | +0.023% | +0.468% | code |
| demotable | `blk.11.ffn_gate_shexp.weight` | +0.095% | +0.253% | code |
| demotable | `blk.31.ffn_gate_up_exps.weight` | -0.052% | +0.195% | chat |
