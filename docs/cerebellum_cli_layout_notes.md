# Cerebellum CLI layout notes

## Current visual problems

- The compact dashboard is too vertically stacked. It reads as many independent
  one-line boxes instead of one organized control surface.
- `Run`, `Activity / health`, `Resources / ETA`, and `Timing` overlap in purpose
  and should be combined into a single operational overview.
- The framing is visually light. Important values need heavier table boundaries,
  clearer section weights, and stronger separation between labels and values.
- Most data is rendered as key/value lines. This wastes horizontal space and
  makes the CLI feel less intentional than a grid/table dashboard.
- ETA is too generic. It should explicitly show:
  - current tensor elapsed
  - average tensor time
  - average layer time when enough layer data exists
  - estimated remaining run time
  - confidence/quality of estimate based on completed count
- The event stream is useful but too dominant in compact mode. It should be
  visually subordinate to live state and measurements.

## Compact dashboard target

Use a bounded, fixed-height dashboard optimized for a normal terminal.

Top banner:

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║ CEREBELLUM  gemma-4/gemma-4-12b-it  wiki  RUNNING                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

Primary grid:

```text
╔═══════════════════════ Progress ═══════════════════════╦════════ Resources ═══════╗
║ [###----------------] 12/616 1.9%                      ║ GPU 100% 10.8/24.6 GiB   ║
║ Current tensor blk.0.attn_v.weight  q3_K               ║ CPU quantize 1180%       ║
║ Active job PPL  elapsed 1m42s                          ║ Disk 61.5 GiB free       ║
╠════════════════════════ Timing / ETA ══════════════════╬════════ GGUF Sizes ══════╣
║ Current tensor 1m42s  avg tensor 9m18s                 ║ baseline 8.0 GiB         ║
║ Avg layer --         total ETA 97h12m                  ║ active   6.2 GiB         ║
║ Estimate confidence low: 2 tensors locked              ║ recent   8.1 GiB         ║
╚═════════════════════════════════════════════════════════╩══════════════════════════╝
```

Measurement table:

```text
╔════════ Recent Measurements ═══════════════════════════════════════════════════════╗
║ Quant  PPL        Δ PPL       Size     Tensor                                    ║
║ q3_K   2421.9309  -82.3478    8.0 GiB  blk.0.ffn_up.weight       better          ║
║ q2_K   2459.0451  -45.2336    7.9 GiB  blk.0.ffn_up.weight       better          ║
║ q5_K   2345.9479  -158.3308   8.1 GiB  blk.0.ffn_up.weight       best            ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

Bottom strip:

```text
Events: ppl_finish q5_K blk.0.ffn_up.weight | tensor_locked q5_K | quant_start q3_K
```

## Scrollable TUI target

The `--tui` mode should become the “research cockpit.”

- Top fixed header with model/run/status/progress.
- Left pane: current tensor/job and timing.
- Right pane: resources, disk, process health, failure detector.
- Bottom tabbed panes:
  - Events
  - Measurements
  - Tensors/decisions
  - Files/sizes
  - System/processes
- Each tab has independent scroll offset.
- Keyboard:
  - `Tab` / arrows switch panes
  - `j/k` or arrows scroll
  - `/` search active pane
  - `f` filter pane
  - `r` reset scroll/filter
  - `q` quit UI only

## Semantic formatting rules

- Labels: dim gray.
- Model/run identity: cyan.
- Active tensor/layer: amber.
- Quant level: magenta.
- PPL: amber.
- Better delta: green.
- Worse delta: red.
- Neutral delta: white.
- Active CPU/GPU process: green.
- Waiting state: yellow.
- Failure/stall: red.
- ETA confidence:
  - low: yellow
  - medium: cyan
  - high: green

## ETA rules

- `current_tensor_elapsed`: elapsed time since latest `tensor_start`.
- `avg_tensor_time`: total measured quant+ppl time divided by locked tensors.
- `avg_layer_time`: average sum of tensor times grouped by `blk.N.*`, only shown
  after at least one layer has enough completed tensors to be meaningful.
- `total_eta`: remaining tensors multiplied by `avg_tensor_time`, upgraded to
  layer-based ETA once layer averages exist.
- `confidence`: low until 5 tensors, medium until 2 layers, high after 3 layers.

## Implementation notes

- Compact mode should stay non-interactive and safe for logs/screenshots.
- TUI mode can be denser and interactive.
- Avoid showing more than 5-8 measurements and 3-5 recent events in compact
  mode unless explicit limits are passed.
- Keep heavy event browsing inside `--tui`.
- Continue recording all raw detail to event/candidate logs; the UI should
  summarize, not drop data.
