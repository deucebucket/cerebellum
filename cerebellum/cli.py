"""Public Cerebellum CLI entrypoint.

The flagship commands are the read-only campaign board (``status``, ``watch``,
``next``, ``method``) — zero required arguments, plain language, safe at 4am.
The legacy hillstep engine stays fully reachable (directly for its
subcommands, or namespaced via ``cerebellum hillstep ...`` where a name is
shadowed by the board), but it is DEPRECATED as a shipping method — the
canonical flow is the OG group->budget bench-gated formula
(``cerebellum method``).
"""

from __future__ import annotations

import sys

TOP_HELP = """\
cerebellum — ablation-informed mixed-precision GGUF quantization

what is going on right now (start here):
  cerebellum status      one screen: every campaign, who has the GPU,
                         Modal spend, and what is waiting on YOU
  cerebellum watch       same as status, refreshed every 30s
  cerebellum next        top of the backlog (NOW / NEXT)
  cerebellum method      the canonical method (OG group->budget,
                         bench-gated). Read this before building anything.

building:
  cerebellum imatrix     generate a Cerebellum/llama.cpp imatrix
                         (streaming, constant RAM) — `imatrix --help`

legacy engine (DEPRECATED as a shipping method — see `cerebellum method`):
  cerebellum run         hillstep per-tensor search. Lost 14 HumanEval+
                         points while wiki PPL improved 35% — only legal
                         as a targeted pass AFTER a group-first scan.
  cerebellum hillstep <subcommand>
                         full legacy engine, including the old run-level
                         `status` / `watch` that the board shadows.
                         `cerebellum hillstep --help` lists everything.
"""

RUN_DEPRECATION_BANNER = """\
================================================================================
 DEPRECATED: `cerebellum run` drives the hillstep per-tensor hill-climb.
 That is NOT the shipping method. Proof: the Gemma 4 12B hillstep checkpoint
 improved wiki PPL 35% while LOSING 14 HumanEval+ points.

 The canonical flow is the OG group->budget bench-gated formula:
     cerebellum method
 Hillstep remains legal only as a targeted pass AFTER a group-first scan.
 Continuing anyway...
================================================================================
"""

BOARD_COMMANDS = ("status", "watch", "next", "method")


def main() -> None:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""

    if cmd == "imatrix":
        from cerebellum.imatrix import main as imatrix_main

        imatrix_main(argv[1:])
        return

    if cmd in BOARD_COMMANDS:
        from cerebellum import statusboard

        raise SystemExit(statusboard.dispatch(cmd, argv[1:]))

    if cmd in ("", "-h", "--help", "help"):
        print(TOP_HELP)
        return

    from cerebellum import main as engine_main

    if cmd == "hillstep":
        engine_main(argv[1:])
        return

    if cmd == "run":
        print(RUN_DEPRECATION_BANNER, file=sys.stderr)

    engine_main(argv)


if __name__ == "__main__":
    main()
