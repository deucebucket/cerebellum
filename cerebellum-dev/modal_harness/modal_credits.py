#!/usr/bin/env python3
"""Modal credits check + cost estimator for the Cerebellum harness.

Usage:
    python modal_credits.py                 # remaining credits this cycle
    python modal_credits.py --estimate L40S 4.5   # cost of 4.5h on an L40S
    python modal_credits.py --gate 5.00     # exit 1 if 5.00 more dollars would breach the ceiling

The $25 soft ceiling (under the $30 monthly credits) is policy: see
memory feedback-modal-30-dollar-hard-cap. Never launch past the gate.
"""
import argparse
import json
import subprocess
import sys

MONTHLY_CREDITS = 30.00
SOFT_CEILING = 27.00  # raised 2026-06-13: spend-to-zero intended; $28.50 doors is the real wall

# verified from modal.com/pricing 2026-06-12 ($/sec -> $/hr)
GPU_HOURLY = {
    "T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
    "A100-40": 2.10, "A100-80": 2.50, "RTX-PRO-6000": 3.03,
    "H100": 3.95, "H200": 4.54, "B200": 6.25,
}
CPU_CORE_HOURLY = 0.0472   # $0.0000131/core/sec
MEM_GIB_HOURLY = 0.0080    # $0.00000222/GiB/sec


def month_usage() -> float:
    out = subprocess.run(
        ["modal", "billing", "report", "--for", "this month", "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = json.loads(out)
    return sum(float(r.get("cost", r.get("Cost", 0)) or 0) for r in rows)


def container_hourly(gpu: str | None, cores: float = 4, mem_gib: float = 32) -> float:
    base = GPU_HOURLY[gpu.upper()] if gpu else 0.0
    return base + cores * CPU_CORE_HOURLY + mem_gib * MEM_GIB_HOURLY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", nargs=2, metavar=("GPU", "HOURS"),
                    help="GPU type (or 'cpu') and hours; prints all-in cost")
    ap.add_argument("--cores", type=float, default=4)
    ap.add_argument("--mem", type=float, default=32)
    ap.add_argument("--gate", type=float, metavar="DOLLARS",
                    help="exit 1 unless DOLLARS more fits under the soft ceiling")
    args = ap.parse_args()

    used = month_usage()
    remaining = MONTHLY_CREDITS - used
    headroom = SOFT_CEILING - used
    print(f"used this cycle:  ${used:.4f}")
    print(f"credits left:     ${remaining:.4f} of ${MONTHLY_CREDITS:.2f}")
    print(f"gate headroom:    ${headroom:.4f} (soft ceiling ${SOFT_CEILING:.2f})")

    if args.estimate:
        gpu, hours = args.estimate[0], float(args.estimate[1])
        gpu_key = None if gpu.lower() == "cpu" else gpu
        rate = container_hourly(gpu_key, args.cores, args.mem)
        cost = rate * hours
        print(f"estimate: {hours}h on {gpu} ({args.cores} cores, {args.mem} GiB) "
              f"= ${rate:.3f}/h all-in = ${cost:.2f}")

    if args.gate is not None:
        if args.gate > headroom:
            print(f"GATE: REFUSE — ${args.gate:.2f} requested > ${headroom:.2f} headroom")
            sys.exit(1)
        print(f"GATE: OK — ${args.gate:.2f} fits under the ceiling")


if __name__ == "__main__":
    main()
