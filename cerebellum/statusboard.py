"""Campaign status board for the Cerebellum CLI.

Read-only "what is happening right now" view across every active campaign
directory (``cerebellum-*/`` with an OPS_LOG / RUN_PLAN / driver log).
Designed for a tired human at 4am: one screen, plain language, zero
required arguments, graceful when files are missing.

Never touches the GPU or the network. The only shell-out is the optional
Modal credits script (local ``modal`` CLI, 10s timeout, tolerated absent).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Files whose presence marks a directory as an active campaign.
CAMPAIGN_MARKERS = (
    "OPS_LOG.md",
    "RUN_PLAN.md",
    "logs/driver.log",
    "logs/continuation.log",
)

# Logs with real narrative content (stage / latest-event lines).
CONTENT_LOGS = (
    "logs/continuation.log",
    "logs/driver.log",
    "OPS_LOG.md",
    "modal_results/results/progress.log",
)

# Everything that counts as "the campaign moved" — content logs plus
# bookkeeping logs whose lines aren't informative on their own.
ACTIVITY_LOGS = CONTENT_LOGS + ("modal_results/sync.log",)

# Directories that match cerebellum-* but are not campaigns.
NON_CAMPAIGN_DIRS = {"cerebellum-dev", "cerebellum-runs"}

GPU_PROCESS_NAMES = ("llama-perplexity", "llama-server", "llama-quantize", "llama-imatrix")

TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\]")
CUM_SPEND_RE = re.compile(r"\(cum \$([0-9]+(?:\.[0-9]+)?)\)")
ETA_RE = re.compile(r"\bETA\b[:\s]*([^\n]*)", re.IGNORECASE)

STALE_AFTER_SECONDS = 3 * 3600  # no log movement for 3h -> flag it

METHOD_FALLBACK = """\
CURRENT_METHOD.md not found — built-in short form of the canon:

  The OG group-first, bench-gated formula (the only shipping method):
   1. HF/BF16 -> F16 GGUF
   2. Imatrix with FULL tensor coverage (experts + routers on MoE)
   3. Uniform baselines (Q2_K/Q3_K_M/Q4_K_M +imatrix), WikiText PPL
   4. GROUP ablation (~20-25 points): crush each group to Q2_K, PPL delta
   5. Classify: PROTECT (>+1.5%) / DEMOTABLE (0..+1.5%) / FREE (<=0%)
   6. Reverse ablation to confirm real regressions
   7. Optional per-layer probe / router curve
   8. BUDGET stage: override file -> stock llama-quantize --tensor-type-file
   9. PPL sanity check
  10. BENCHMARK GATES vs same-size uniform baseline (ARC, HellaSwag,
      MMLU-Redux, HumanEval+, BigCodeBench). Pass -> ship. Fail -> promote
      the regressing group and rebuild.
  11. Audit wrong answers before recording any score
  12. Tune launch args post-gate, pre-publish

  DEPRECATED: exhaustive hillstep per-tensor hill-climb and any PPL-only
  ship gate. PPL is a damage sensor, never the release objective.
"""


@dataclass
class Campaign:
    name: str
    path: Path
    log_path: Path | None = None
    stage: str | None = None
    latest: str | None = None
    last_ts: datetime | None = None
    eta: str | None = None
    modal_spend: str | None = None
    has_run_plan_gate: bool = False
    waiting: list[str] = field(default_factory=list)
    status: str = "ok"
    status_reason: str = ""


# --------------------------------------------------------------------------
# discovery + parsing
# --------------------------------------------------------------------------

def find_root(cwd: Path | None = None) -> Path:
    """Best-effort repo root: cwd, then the directory this package lives in."""
    candidates = []
    if cwd is not None:
        candidates.append(Path(cwd))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[1])
    for cand in candidates:
        try:
            if any(_is_campaign_dir(p) for p in cand.glob("cerebellum-*")):
                return cand
        except OSError:
            continue
    return candidates[-1]


def _is_campaign_dir(path: Path) -> bool:
    if not path.is_dir() or path.name in NON_CAMPAIGN_DIRS:
        return False
    return any((path / marker).exists() for marker in CAMPAIGN_MARKERS)


def find_campaigns(root: Path) -> list[Path]:
    try:
        dirs = sorted(p for p in root.glob("cerebellum-*") if _is_campaign_dir(p))
    except OSError:
        dirs = []
    return dirs


def _tail_lines(path: Path, n: int = 80) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [ln.rstrip() for ln in text.splitlines()[-n:] if ln.strip()]


def _parse_line_ts(line: str) -> datetime | None:
    m = TIMESTAMP_RE.match(line)
    if not m:
        return None
    raw = m.group(1)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _strip_ts(line: str) -> str:
    return TIMESTAMP_RE.sub("", line).strip()


def _newest_log(path: Path, sources: tuple[str, ...]) -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    for rel in sources:
        cand = path / rel
        if cand.is_file():
            try:
                mtime = cand.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = cand, mtime
    return best


def _age_str(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return "moments ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h {minutes % 60}m ago"
    return f"{hours // 24}d ago"


def inspect_campaign(path: Path, now: datetime | None = None) -> Campaign:
    """Build a Campaign snapshot from on-disk files. Never raises on bad files."""
    now = now or datetime.now()
    camp = Campaign(name=path.name.removeprefix("cerebellum-"), path=path)

    # Content (stage / latest event) comes from narrative logs; activity
    # freshness also counts bookkeeping logs like modal sync.log.
    log = _newest_log(path, CONTENT_LOGS)
    activity_log = _newest_log(path, ACTIVITY_LOGS)
    camp.log_path = activity_log or log
    lines = _tail_lines(log) if log else []
    activity_lines = _tail_lines(activity_log) if activity_log else []

    if activity_lines:
        for line in reversed(activity_lines):
            ts = _parse_line_ts(line)
            if ts:
                camp.last_ts = ts
                break
        if camp.last_ts is None and activity_log is not None:
            try:
                camp.last_ts = datetime.fromtimestamp(activity_log.stat().st_mtime)
            except OSError:
                pass

    if lines:
        camp.latest = _strip_ts(lines[-1])
        for line in reversed(lines):
            bare = _strip_ts(line)
            if "stage" in bare.lower() or bare.startswith("OPS:"):
                camp.stage = bare.strip("= ").strip()
                break
        for line in reversed(lines):
            m = ETA_RE.search(line)
            if m and m.group(1).strip():
                camp.eta = m.group(1).strip()
                break
        spends = [m.group(1) for ln in lines for m in [CUM_SPEND_RE.search(ln)] if m]
        if spends:
            camp.modal_spend = f"${spends[-1]}"

    # Human gates
    run_plan = path / "RUN_PLAN.md"
    if run_plan.is_file():
        try:
            plan_text = run_plan.read_text(errors="replace")
        except OSError:
            plan_text = ""
        if "STOP LINE" in plan_text or "STOPS" in plan_text:
            camp.has_run_plan_gate = True

    summary = path / "SUMMARY_FOR_HUMAN.md"
    if summary.is_file():
        camp.waiting.append(f"SUMMARY_FOR_HUMAN.md is waiting for your review ({summary})")

    # Only the driver's explicit "STOP HERE" marker means it actually stopped —
    # OPS logs mention "STOP LINE" in prose while still running.
    tail5 = [_strip_ts(ln) for ln in lines[-5:]]
    if any("STOP HERE" in ln for ln in tail5):
        camp.waiting.append("driver stopped at the human-review gate (see RUN_PLAN.md)")

    # Status verdict
    bad_words = ("traceback", "fatal", "killed")
    if camp.waiting:
        camp.status, camp.status_reason = "attention", "waiting on you"
    elif any(any(w in ln.lower() for w in bad_words) for ln in tail5):
        camp.status, camp.status_reason = "attention", "recent log lines look like a crash — read the log"
    elif not lines and not activity_lines:
        camp.status, camp.status_reason = "attention", "no activity recorded (planned but nothing has logged yet)"
    elif camp.last_ts and (now - camp.last_ts).total_seconds() > STALE_AFTER_SECONDS:
        age = _age_str((now - camp.last_ts).total_seconds()).removesuffix(" ago")
        camp.status, camp.status_reason = "attention", (
            f"no log movement for {age} — stalled or quietly finished"
        )
    else:
        camp.status, camp.status_reason = "ok", "running"
    return camp


# --------------------------------------------------------------------------
# GPU + Modal
# --------------------------------------------------------------------------

def gpu_processes() -> list[str]:
    """Local llama.cpp processes that hold the GPU. Read-only (pgrep)."""
    pattern = "|".join(GPU_PROCESS_NAMES)
    try:
        out = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        name = next((n for n in GPU_PROCESS_NAMES if n in cmd), cmd.split()[0])
        procs.append(f"{name} (pid {pid})")
    return procs


def modal_credits_summary(root: Path, timeout: float = 10.0) -> str:
    script = root / "cerebellum-dev" / "modal_harness" / "modal_credits.py"
    if not script.is_file():
        return "not checked (modal_credits.py not found)"
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "unavailable (credits check timed out after 10s)"
    except (OSError, subprocess.SubprocessError):
        return "unavailable (credits check failed to launch)"
    if proc.returncode != 0:
        return "unavailable (credits script errored — is the modal CLI logged in?)"
    return proc.stdout.strip() or "unavailable (credits script printed nothing)"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_status(
    root: Path | None = None,
    *,
    now: datetime | None = None,
    include_processes: bool = True,
    include_modal: bool = True,
) -> str:
    now = now or datetime.now()
    root = root or find_root()
    out: list[str] = []
    out.append(f"CEREBELLUM STATUS — {now:%Y-%m-%d %H:%M}")
    out.append(f"root: {root}")
    out.append("")

    campaigns = [inspect_campaign(p, now=now) for p in find_campaigns(root)]

    out.append("== CAMPAIGNS ==")
    if not campaigns:
        out.append("  none found (no cerebellum-*/ dir has an OPS_LOG, RUN_PLAN, or driver log)")
    for camp in campaigns:
        out.append("")
        out.append(f"  {camp.name}    STATUS: {camp.status} — {camp.status_reason}")
        if camp.stage:
            out.append(f"      stage:   {camp.stage}")
        if camp.latest:
            out.append(f"      latest:  {camp.latest}")
        if camp.last_ts:
            age = _age_str((now - camp.last_ts).total_seconds())
            src = camp.log_path.relative_to(camp.path) if camp.log_path else "?"
            out.append(f"      last activity: {age} ({src})")
        elif not camp.latest:
            out.append("      no activity recorded")
        if camp.eta:
            out.append(f"      eta:     {camp.eta}")
        if camp.modal_spend:
            out.append(f"      modal spend so far: {camp.modal_spend}")
        if camp.has_run_plan_gate:
            out.append("      human gate: RUN_PLAN.md has a stop line — review before the next stage")
        for item in camp.waiting:
            out.append(f"      WAITING ON YOU: {item}")

    out.append("")
    out.append("== LOCAL GPU ==")
    if include_processes:
        procs = gpu_processes()
        if procs:
            for p in procs:
                out.append(f"  busy: {p}")
        else:
            out.append("  looks free — no llama-perplexity / llama-server / llama-quantize running")
    else:
        out.append("  (process check skipped)")

    out.append("")
    out.append("== MODAL ==")
    if include_modal:
        for line in modal_credits_summary(root).splitlines():
            out.append(f"  {line}")
    else:
        out.append("  (credits check skipped)")

    waiting_all = [(c.name, w) for c in campaigns for w in c.waiting]
    out.append("")
    out.append("== WAITING FOR YOU ==")
    if waiting_all:
        for name, item in waiting_all:
            out.append(f"  - [{name}] {item}")
    else:
        out.append("  nothing — go back to sleep")
    return "\n".join(out)


def render_next(root: Path | None = None) -> str:
    root = root or find_root()
    backlog = root / "cerebellum-dev" / "BACKLOG.md"
    if not backlog.is_file():
        return f"backlog not found at {backlog} — nothing queued, or wrong root"
    try:
        lines = backlog.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"could not read {backlog}: {exc}"
    out: list[str] = []
    keep = False
    for line in lines:
        if line.startswith("## "):
            header = line[3:].strip().upper()
            keep = header.startswith(("NOW", "NEXT"))
            if not keep and out:
                break  # past NOW/NEXT — stop
        if keep:
            out.append(line)
    if not out:
        return f"no NOW/NEXT sections found in {backlog}"
    return "\n".join(out).strip()


def render_method(root: Path | None = None) -> str:
    root = root or find_root()
    method = root / "cerebellum-dev" / "knowledge" / "CURRENT_METHOD.md"
    if not method.is_file():
        return METHOD_FALLBACK
    try:
        text = method.read_text(errors="replace")
    except OSError:
        return METHOD_FALLBACK
    # Keep it compact: the Canonical and Deprecated sections only.
    out: list[str] = ["THE METHOD (canon — full text: " + str(method) + ")", ""]
    keep = False
    for line in text.splitlines():
        if line.startswith("## "):
            header = line[3:].strip().lower()
            keep = header.startswith(("canonical", "deprecated"))
        if keep and line.strip():
            out.append(line)
    if len(out) <= 2:
        return METHOD_FALLBACK
    return "\n".join(out)


# --------------------------------------------------------------------------
# command entrypoints
# --------------------------------------------------------------------------

def cmd_status(argv: list[str]) -> int:
    if argv:
        print(
            "`cerebellum status` takes no arguments — it shows every campaign.\n"
            "For the old hillstep run-level status use: cerebellum hillstep status",
            file=sys.stderr,
        )
        return 2
    try:
        from cerebellum._banner import banner
        print(banner())
        print()
    except Exception:
        pass
    print(render_status())
    return 0


def cmd_watch(argv: list[str], *, interval: float = 30.0, iterations: int | None = None) -> int:
    if argv:
        print(
            "`cerebellum watch` takes no arguments — it refreshes `cerebellum status` every 30s.\n"
            "For the old hillstep run TUI use: cerebellum hillstep watch",
            file=sys.stderr,
        )
        return 2
    count = 0
    try:
        while True:
            print("\033[2J\033[H", end="")
            print(render_status())
            print(f"\n(refreshing every {interval:.0f}s — Ctrl-C to exit)")
            count += 1
            if iterations is not None and count >= iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
    return 0


def cmd_next(argv: list[str]) -> int:
    print(render_next())
    return 0


def cmd_method(argv: list[str]) -> int:
    print(render_method())
    return 0


def dispatch(cmd: str, argv: list[str]) -> int:
    handlers = {
        "status": cmd_status,
        "watch": cmd_watch,
        "next": cmd_next,
        "method": cmd_method,
    }
    return handlers[cmd](argv)
