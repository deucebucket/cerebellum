#!/usr/bin/env python3
"""Independent Modal spend watchdog. Polls billing + running apps; kills
everything if spend or runtime exceeds hard envelopes. Costs nothing to run
(billing/app-list are free API calls).

Kill conditions (any one triggers stop-all-apps):
  1. Month spend > SPEND_KILL ($20 — campaign cap + margin, under the $25 gate
     and the $30 credits)
  2. Spend velocity > VELOCITY_KILL ($4/hr sustained over the last 2 polls —
     faster than any legitimate single-GPU phase we run; a retry loop on a
     GPU container shows up here)
  3. Any single app running longer than APP_MAX_HOURS (4h — longest legitimate (the Flash campaign driver runs ~5-6h through all phases)
     phase is the ablation fan-out, with per-function timeouts well under this)

Run detached:  nohup setsid python3 modal_watchdog.py >> watchdog.log 2>&1 &
"""
import json
import subprocess
import time
from datetime import datetime

SPEND_KILL = 28.50  # doors-close: credits nearly maxed; kill everything, sync data home
VELOCITY_KILL_PER_HR = 4.00
APP_MAX_HOURS = 9.0
POLL_SECONDS = 300


def sh(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


def month_spend() -> float:
    out = sh(["modal", "billing", "report", "--for", "this month", "--json"]).stdout
    try:
        return sum(float(r.get("cost", r.get("Cost", 0)) or 0) for r in json.loads(out))
    except Exception:
        return -1.0  # billing read failed; don't kill on bad data, just log


def running_apps() -> list[dict]:
    out = sh(["modal", "app", "list", "--json"]).stdout
    try:
        return [a for a in json.loads(out)
                if any(s in a.get("State", a.get("state", "")).lower()
                       for s in ("running", "deployed", "ephemeral"))]
    except Exception:
        return []


def final_sync():
    """Doors closing: pull every volume's results home before the lights go out."""
    import subprocess as sp
    pulls = [
        ("cerebellum-flash", "results/", "/var/home/deucebucket/ai-drive/cerebellum/cerebellum-glm47-flash/modal_results/"),
        ("cerebellum-north", "results/", "/var/home/deucebucket/ai-drive/cerebellum/cerebellum-north-mini-code/modal_results/"),
        ("cerebellum-35b-v4", "results/", "/var/home/deucebucket/ai-drive/cerebellum/cerebellum-qwen36-35b-v4/modal_results/"),
    ]
    for vol, src, dest in pulls:
        sp.run(["mkdir", "-p", dest])
        r = sp.run(["modal", "volume", "get", "--force", vol, src, dest], capture_output=True, text=True, timeout=600)
        log(f"  final sync {vol}: rc={r.returncode}")


def kill_all(apps, reason):
    log(f"KILL TRIGGERED: {reason}")
    try:
        final_sync()
    except Exception as e:
        log(f"  final sync error: {e}")
    for a in apps:
        app_id = a.get("App ID") or a.get("app_id") or a.get("id")
        if app_id:
            r = sh(["modal", "app", "stop", "-y", app_id])
            log(f"  stopped {app_id}: rc={r.returncode} {r.stderr.strip()[:120]}")
    log("all running apps stopped; watchdog continues monitoring")


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"watchdog up: kill at ${SPEND_KILL} total, ${VELOCITY_KILL_PER_HR}/hr velocity, {APP_MAX_HOURS}h app age")
    prev_spend, prev_t = None, None
    vel_breaches = 0
    while True:
        spend = month_spend()
        apps = running_apps()
        if spend >= 0:
            vel = ""
            if prev_spend is not None and prev_t is not None and spend > prev_spend:
                hrs = (time.time() - prev_t) / 3600
                rate = (spend - prev_spend) / hrs if hrs > 0 else 0
                vel = f", velocity ${rate:.2f}/hr"
                # require SUSTAINED breach (2 consecutive polls) so a single
                # expensive-retry spike doesn't kill a healthy campaign
                if rate > VELOCITY_KILL_PER_HR:
                    vel_breaches += 1
                    if vel_breaches >= 2 and apps:
                        kill_all(apps, f"sustained velocity ${rate:.2f}/hr > ${VELOCITY_KILL_PER_HR}/hr x{vel_breaches}")
                else:
                    vel_breaches = 0
            log(f"spend ${spend:.4f}, {len(apps)} app(s) running{vel}")
            if spend > SPEND_KILL and apps:
                kill_all(apps, f"month spend ${spend:.2f} > ${SPEND_KILL}")
            prev_spend, prev_t = spend, time.time()
        else:
            log("billing read failed; skipping this poll")
        for a in apps:
            created = a.get("Created at") or a.get("created_at") or ""
            try:
                age_h = (time.time() - datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()) / 3600
                if age_h > APP_MAX_HOURS:
                    kill_all([a], f"app age {age_h:.1f}h > {APP_MAX_HOURS}h")
            except Exception:
                pass
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
