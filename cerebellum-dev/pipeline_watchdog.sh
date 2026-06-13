#!/usr/bin/env bash
# pipeline_watchdog.sh — single tracked watcher over every moving piece
# (2026-06-12, agent). Polls every 4 min and EXITS (which alerts the agent
# session) on the first anomaly OR major milestone. Print-then-exit is the
# whole design: the harness notifies the agent when this process ends.
set -u
ROOT=/var/home/deucebucket/ai-drive/cerebellum
GLM=$ROOT/cerebellum-glm47-flash
CLUB=$ROOT/cerebellum-club3090
NORTH=$ROOT/cerebellum-north-mini-code
MODAL=/home/deucebucket/.local/bin/modal

say() { echo "WATCHDOG[$(date '+%F %T')]: $*"; }

while true; do
  # 1) error strings in the live OPS logs (new ABORT/FAILED/FATAL entries)
  for f in "$GLM/OPS_LOG.md" "$CLUB/OPS_LOG.md" "$NORTH/OPS_LOG.md"; do
    recent=$(tail -8 "$f" 2>/dev/null | grep -E 'ABORT|FATAL|FAILED rc=|WITH FAILURES|HARD STOP|BUDGET ABORT' | tail -2)
    if [ -n "$recent" ]; then
      say "error lines in $(basename "$(dirname "$f")")/OPS_LOG.md:"; echo "$recent"; exit 0
    fi
  done

  # 2) north campaign on Modal: must be running OR have delivered the map.
  # NB: must use --json — the table output truncates descriptions.
  if ! "$MODAL" volume ls cerebellum-north results/ablation_summary.md >/dev/null 2>&1; then
    north_alive=$("$MODAL" app list --json 2>/dev/null | python3 -c "
import json, sys
try:
    apps = json.load(sys.stdin)
except Exception:
    print('unknown'); sys.exit()
print('yes' if any('north' in str(a.get('description','')).lower()
                   and any(s in str(a.get('state','')).lower() for s in ('running','ephemeral'))
                   for a in apps) else 'no')")
    if [ "$north_alive" = "no" ]; then
      say "north app NOT running and no ablation_summary.md on volume — campaign died mid-flight"; exit 0
    fi
  else
    say "MILESTONE: north v1 ablation map is on the volume (XXS launcher should fire within 10 min)"; exit 0
  fi

  # 3) GLM v2a chain: script alive or marker set — else chain broken
  if [ ! -f "$GLM/state_v2a_gate.done" ] \
     && ! pgrep -f 'build_gate_v2a[.]sh' >/dev/null 2>&1; then
    say "v2a build+gate script not running and marker not set — GLM v2 chain broken (logs/build_gate_v2a.log)"; exit 0
  fi

  # 4) GLM milestone: v2a gate verdict recorded
  if grep -q 'v2a GATE vs' "$GLM/OPS_LOG.md" 2>/dev/null; then
    say "MILESTONE: GLM v2a gate verdict:"; grep 'v2a GATE vs' "$GLM/OPS_LOG.md" | tail -1; exit 0
  fi

  # 5) local compute idle while v2a still pending (idle-waste detector)
  if [ ! -f "$GLM/state_v2a_gate.done" ]; then
    if ! pgrep -x llama-server >/dev/null 2>&1 \
       && ! pgrep -f 'llama-perplexity|llama-quantize' >/dev/null 2>&1; then
      # give handoffs a grace window: only alert if still idle next round
      sleep 240
      if ! pgrep -x llama-server >/dev/null 2>&1 \
         && ! pgrep -f 'llama-perplexity|llama-quantize' >/dev/null 2>&1 \
         && [ ! -f "$GLM/state_v2a_gate.done" ]; then
        say "local compute idle ~8 min while v2a is still pending — a handoff may be stuck"; exit 0
      fi
      continue
    fi
  fi

  sleep 240
done
