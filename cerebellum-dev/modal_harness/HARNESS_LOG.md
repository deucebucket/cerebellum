# Modal harness log

Bug history + fixes for the Modal campaign drivers. Newest first.

## 2026-06-12 — XXS stage staged: `cerebellum_flash_xxs.py` + `launch_xxs_when_north_done.sh`

New file: `cerebellum_flash_xxs.py` — IQ2-class extreme-small PPL screen for
GLM-4.7-Flash (3 candidates, PPL-only, no benchmarks on Modal).

**Candidates chosen from the ablation map** (no new ablation):

| candidate | base type | est. GB | rationale |
|---|---|---|---|
| XXS-A | IQ2_M  | ~8.8 | Smallest safe step below C1; PROTECT lifted Q4_K/Q5_K |
| XXS-B | IQ2_XS | ~8.3 | One notch deeper; down_exps→IQ2_M cushion (TOLERANT +4% wiki) |
| XXS-C | IQ2_XXS | ~7.9 | Floor probe; maps the hard lower bound for 8 GB card viability |

All three share the same PROTECT lifts:
`shared_expert→Q5_K`, `attn_kv_b→Q4_K`, `attn_q_ab→Q4_K`,
`token_embd→Q5_K`, `output→Q5_K`.
DEMOTABLE groups (gate_up_exps, kv_compress, dense_l0) and TOLERANT groups
(down_exps, attn_output) are left at the IQ2 base type — the base IS the
demotion.  XXS-B/C add a per-candidate down_exps line one notch above the
base as a tolerance cushion.

**Design decisions carried over from prior harness lessons:**
- `quantize_vol` reused from `cerebellum_flash_campaign` (skip-if-exists,
  `_sanitize_override`, `vol.commit()` after write, `_wait_for_vol_file` on
  the reader side).
- `ppl_xxs` uses blocking `subprocess.run` only — no llama-server, no open
  file handles on the Volume between calls.  Avoids the 2026-06-12 open-file
  bug entirely (that bug required a long-lived server Popen writing to a
  Volume-resident log; PPL containers have no such handle).
- Env merging uses `{**os.environ, ..., **env_extra}` pattern throughout
  (the 2026-06-12 duplicate-kwarg lesson), though no `run_logged` pattern
  is needed here.
- All candidates < 12.5 GB → T4 tier for PPL (confirmed by size routing rule
  in `cerebellum_north_campaign.pick_ppl`).
- Budget: $5.00 hard cap; per-step `gate()` checks before every build batch
  and every PPL call.  Estimated actual spend ~$1.15.
- Driver runs ON MODAL (detached), timeout 6 h.

**Launcher:** `cerebellum-glm47-flash/launch_xxs_when_north_done.sh` polls
`modal volume ls cerebellum-north results/ablation_summary.md` every 10 min;
launches when north v1 is confirmed done; idempotence marker
`state_xxs_launched.done`; logs to OPS_LOG.md with `[xxs]` tags.
`bash -n` clean, `chmod +x` set.  NOT started — staged only.

## 2026-06-12 — stage 4 double bench failure (both fixed in `cerebellum_flash_stage4.py`)

Both stage-4 bench calls failed (total burn $0.20; summary at
`cerebellum-glm47-flash/modal_results/results/stage4/stage4_summary.md`).

**Bug 1 — `dict() got multiple values for keyword argument 'RESULTS_DIR'`**
(`glm47_flash_cerebellum_c1`). `run_logged()` built the subprocess env as
`dict(os.environ, ..., RESULTS_DIR=resdir, **env_extra)`. The very first call
is the evalplus probe, whose `env_extra` also carries `RESULTS_DIR=probe_dir`
→ duplicate kwarg → TypeError before the probe even logged its RUN line
(confirmed in `bench_container.log`: dies right after "smoke chat").
Fix: build the env with a dict literal (`{**os.environ, ..., **env_extra}`)
where later keys override instead of raising.

**Bug 2 — `there are open files preventing the operation: ... llama_server.log`**
(`glm47_flash_uniform_q3km`, failed in 0.4 s). Bug 1's exception escaped
`bench_suite` with the healthy llama-server child still running and the
`open(server_log, "ab")` handle never closed (it was *never* closed, even on
the success path — `proc.kill()` at the end didn't `wait()` either). Modal
reused the warm container for the second bench; its first `vol.reload()` in
`_wait_for_vol_file` refused because the dead bench's server still held
`results/stage4/glm47_flash_cerebellum_c1/llama_server.log` open on the
Volume. Fix: `bench_suite` is now a try/finally wrapper around `_bench_body`;
`_shutdown_server()` (terminate → wait(15) → kill → wait, then flush/fsync/
close the log handle) runs on every exit path — success, exception, or
flag-set retry — followed by a best-effort `vol.commit()`.

**North campaign exposure** (`cerebellum_north_campaign.py`, running detached
at the time): audited, **neither bug present**. No `dict(env, **extra)`
pattern anywhere (its `RESULTS_DIR` is a module path constant; subprocess envs
are never merged), and no long-lived server processes — every subprocess is a
blocking `subprocess.run` (llama-quantize / llama-perplexity / converter) that
exits and releases its fds before any `vol.reload()`. Its bench stage is
deliberately local (phase 3 STOPs), so the server-log-on-volume pattern can't
occur. No relaunch needed for these bugs. Files pass `ruff` + `py_compile` +
import.

Rule going forward: any harness that Popens a server writing to a
Volume-mounted log must own a try/finally that kills the process *and* closes
the log handle before the function exits — warm-container reuse makes leaked
fds the next call's problem, and `vol.reload()` hard-fails on open files.

## 2026-06-12 — north campaign: PPL_LANES=2 (phase 2 parallel PPL)

Phase-2 ablation PPL was serial (one GPU lane) while CPU builds ran 2-wide.
Patched to keep up to 2 PPL spawns in flight (same total GPU-seconds, ~half
the wall clock); L40S-class PPLs always run solo so two of them can't trip
the $4/hr watchdog. Budget margin in the per-group stop check now scales
with in-flight count. Running instance predates the patch — detached monitor
cerebellum-north-mini-code/relaunch_2lane.sh stops it at the phase-1
checkpoint (image cached, phase 0+1 banked, cost_ledger.json persists) and
relaunches; driver resumes from volume checkpoints.

## 2026-06-12 20:21 — two campaigns launched (North P3 + 35B v4)
- `cerebellum_north_phase3.py` (NEW): North allocation candidates C1/C2/C3, ON-Modal
  driver (fc-01KTZ8HYYH8JDXMS7YZTRBHCQC, app ap-MvllQd9JMS46m5jhfnilOx), budget $2.50.
  Candidates ADAPTED: map has no DEMOTABLE groups, experts are PROTECT — see
  cerebellum-north-mini-code/OPS_LOG.md. Entrypoint is `launch_p3` (the imported
  campaign module already owns `launch` — duplicate-entrypoint error otherwise).
- `cerebellum_35b_v4_campaign.py` (NEW): Qwen3.6-35B-A3B tensor-level pass, Volume
  cerebellum-35b-v4, b9603 release images (qwen35moe supported), budget $9.00.
  P0 (CPU convert) launched immediately (fc-01KTZ920HYV9MDYTQ5GGAMDM1J, app
  ap-RRdrGSwtaun9Hi4O9ABGBp, gpu_phases=False); GPU phases launch via the LOCAL
  sequencer cerebellum-qwen36-35b-v4/launch_when_clear.sh once `modal app list`
  shows zero running apps (one-GPU-lane rule across all campaigns).
- modal_watchdog.py: added cerebellum-35b-v4 to doors-close final_sync; watchdog
  restarted (PID 1203605).
