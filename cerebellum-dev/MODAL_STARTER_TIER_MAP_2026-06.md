# Modal Starter Tier Capability Map — verified 2026-06-12

Researched from modal.com primary sources (pricing page, docs) on 2026-06-12.
Account context: Starter plan, payment method on file, ~$24.69 credits left this cycle.

## Starter plan limits

| Limit | Value | Source confidence |
|---|---|---|
| Monthly free credits | **$30 / month** | Confirmed (pricing page + 3 secondary sources) |
| Plan fee | $0 | Confirmed |
| Workspace seats | 3 | Confirmed (pricing page) |
| Concurrent containers | **100** | Confirmed (pricing page) |
| Concurrent GPUs | **10 (workspace-wide, NOT per GPU type)** | Confirmed (pricing page); no per-type breakdown documented anywhere |
| Function timeout | default 300 s, **max 24 h per execution** | Confirmed (docs/guide/timeouts) |
| Pending inputs per function | 2,000 (1M via `.spawn()`) | Confirmed (docs/guide/scale) |
| Total inputs per function | 25,000; `.map()` ≤1000 concurrent | Confirmed |
| Container disk | 512 GiB default, 3.0 TiB max | Confirmed (docs/guide/resources) |
| CPU default | 0.125 cores request, soft limit ~16.125 cores | Confirmed |
| Volume storage | $0.09/GiB/mo, **first 1 TiB/mo free**; v2 volumes: no inode limit, files <1 TiB, ≤262,144 files/dir | Confirmed (pricing + docs/guide/volumes) |
| Scheduled & web functions | "limited" on Starter (exact limits undocumented) | Pricing page wording only |
| Regions | No plan restriction, BUT pinning a region costs a **1.5x (broad, e.g. `us`) / 1.75x (narrow, e.g. `us-west`) price multiplier**. Leave region unset for base price. | Confirmed (docs/guide/region-selection) |

Team plan for reference: $250/mo + compute, $100/mo credits, 1000 containers, 50 GPU concurrency.

## GPU pricing (June 2026, base / no region pin)

| GPU | $/sec | $/hr |
|---|---|---|
| T4 (16 GB) | $0.000164 | $0.59 |
| L4 (24 GB) | $0.000222 | $0.80 |
| A10 (24 GB) | $0.000306 | $1.10 |
| L40S (48 GB) | $0.000542 | $1.95 |
| A100-40GB | $0.000583 | $2.10 |
| A100-80GB | $0.000694 | $2.50 |
| RTX PRO 6000 (96 GB) | $0.000842 | $3.03 |
| H100 (80 GB) | $0.001097 | $3.95 |
| H200 (141 GB) | $0.001261 | $4.54 |
| B200 (B200+ may give B300 at same price) | $0.001736 | $6.25 |

CPU: **$0.0000131 / core / sec** = $0.0472 / core / hr (0.125-core minimum billed).
Memory: **$0.00000222 / GiB / sec** = $0.0080 / GiB / hr.
GPU, CPU, and RAM are billed **separately and additively** — a "GPU container" pays GPU + cores + GiB.

Multi-GPU: up to 8x per container for T4/L4/L40S/A100/H100/H200/B200, 4x for A10. Docs warn >2 GPUs/container usually means longer cold-start waits.

## Credits / overage / hard-cap mechanics

- **Credits do NOT roll over.** Reset monthly with the billing cycle. (Stated by multiple secondary sources incl. eesel.ai and corroborated by user reports; Modal's own docs page doesn't state it explicitly — dashboard "Usage & Billing" shows the reset date.)
- **Cycle reset date: not documented publicly** — it's your billing-cycle anchor (likely signup date). Verify in dashboard → Usage & Billing.
- **With a payment method on file, exhausting credits does NOT stop work** — usage past credits accrues and is auto-charged at end of cycle. Worse: Modal docs say *"you will be auto-charged for incremental usage the first time you exceed certain thresholds. These charges occur within the billing cycle"* — so mid-cycle card charges happen once you cross spend thresholds.
- **The hard-cap mechanism is the "Workspace budget"** in dashboard → Usage & Billing. Docs: *"To set limits on how much Modal usage can be incurred within the monthly billing period, go to the 'Workspace budget' section of Usage & Billing."* Caveat from docs: *"The max you can set this limit to is based on the history of prior successful charges"* — that constrains the *maximum*, not the minimum, so a low cap should be settable.
- **What happens at the budget cap — EMPIRICALLY VERIFIED on this account (2026-06-12):** Jerry set the budget to $0 and Modal refused even a trivial container ("App creation failed: workspace billing cycle spend limit reached") — the budget hard-blocks app creation. It meters **GROSS usage including free credits** (a $0 budget bricked the account despite $30 credits being available). New launches are blocked at the cap; assume an in-flight run can also die mid-cycle if it crosses the line — size each run to fit remaining headroom anyway.
- **The cap is SET on this account: $30**, exactly matching the monthly free credits — usage stops where credits end, the card is unreachable. Do not set $0 (bricks the account). Every Modal function still carries an explicit `timeout=` (project policy) as the second line of defense.

## What this means for parallel quantize + PPL work

- **Concurrency is generous for our scale:** 10 concurrent GPUs and 100 containers means 2x L4, 4x L4, L4+L40S, even 8x T4 lanes all fit on Starter. Mixed GPU types count against the same pool of 10. Queue behavior past 10: requests queue/cold-start-wait; no documented error below the 2,000-pending-inputs ceiling.
- **The binding constraint is dollars, not concurrency.** What $24.69 buys (incl. typical 4-core/16-GiB sidecar ≈ +$0.32/hr):
  - **L4 PPL lanes:** ~$1.12/hr loaded → **~22 L4-hours** (or ~31 raw GPU-hours). Two parallel L4 lanes = ~11 wall-clock hours.
  - **L40S (48 GB, fits bigger PPL contexts):** ~$2.27/hr loaded → **~10.9 L40S-hours**.
  - **T4 (cheapest, 16 GB):** ~$0.91/hr loaded → **~27 T4-hours**.
  - **CPU-only quantize lanes:** 16 cores + 64 GiB = $1.27/hr → **~19.5 hours**; 8 cores + 48 GiB = $0.76/hr → **~32 hours**.
- **Do not pin regions** — 1.5–1.75x multiplier wipes out a third of the budget for nothing we need.
- **24 h max per function execution** — fine for detached quantize/PPL stages; checkpoint anything that could exceed it.
- **Volumes are effectively free** at our scale (1 TiB/mo included); note deleted data can still bill for up to 4 days, and daily snapshot is the metering basis. Use Volume v2 (no inode limit) for GGUF shard dirs.

## 2026 notes / changelog findings

- New GPU SKUs on the menu: **RTX PRO 6000 (96 GB @ $3.03/hr)** — interesting middle option between L40S and H100 for big-model PPL; **B200/B200+** ($6.25/hr, B200+ may auto-upgrade to B300 at B200 price); **H100!** (forces H100, blocks auto-upgrade to H200); **H200** sometimes given for H100 price via auto-upgrade.
- **No spot/preemptible pricing exists** on Modal as of June 2026 (searched; nothing found).
- **No volume read/write/egress charges found** — volumes bill on stored GiB only (daily snapshot).
- Could not fetch Modal's changelog directly (modal.com/blog/changelog and /docs/changelog both 404 from here) — no evidence of Starter-tier changes in the last 6 months; the $30/credit, 100-container, 10-GPU numbers are current as of today.

## Could NOT verify (dashboard-only)

1. Exact billing-cycle reset date for this account → dashboard Usage & Billing.
2. ~~Whether Workspace budget cap kills in-flight runs or only blocks new containers.~~ RESOLVED 2026-06-12: blocks app creation at minimum (verified empirically via Jerry's $0-budget test); in-flight kill behavior still assumed-yes, unproven.
3. ~~Whether the budget cap is gross usage or net-of-credits.~~ RESOLVED 2026-06-12: GROSS — $0 budget blocked everything despite full credits remaining. Budget $30 = spend stops exactly where free credits end.
4. The incremental-usage charge threshold amounts (moot while the $30 gross cap is set — the card can't be reached).
5. Any per-GPU-type sub-quotas within the 10-GPU pool (none documented; assume shared pool).
6. "Scheduled and Web Functions (limited)" — exact Starter limits undocumented.

## Sources

- https://modal.com/pricing (plans, credits, GPU/CPU/RAM/volume prices, concurrency)
- https://modal.com/docs/guide/billing (cycle, workspace budget, incremental charges)
- https://modal.com/docs/guide/timeouts (300 s default, 24 h max)
- https://modal.com/docs/guide/gpu (GPU types, multi-GPU limits)
- https://modal.com/docs/guide/resources (CPU/memory/disk defaults and caps)
- https://modal.com/docs/guide/volumes (v1/v2 limits, metering)
- https://modal.com/docs/guide/scale (input limits, Resource Exhausted)
- https://modal.com/docs/guide/region-selection (1.5x/1.75x multipliers)
- Secondary cross-checks: checkthat.ai/brands/modal/pricing, eesel.ai/blog/modal-ai-pricing, costbench.com/software/ai-gpu-cloud/modal, modal.com/blog/nvidia-b200-pricing

## Observed enforcement slop (2026-06-12, Jerry's account)

The budget wall is not penny-exact in real time: the dashboard briefly showed
-$0.03 past the cap, then reconciled back to -$0.01 ("like a rollback").
Metering is eventually-consistent — in-flight container-seconds can leak cents
past the boundary before enforcement and reconciliation catch up. Implications:
a few cents negative does NOT mean the cap failed; and conversely, don't plan
runs that depend on stopping at exactly $0.00 remaining — leave dollar-level
headroom for the final container to wind down.
