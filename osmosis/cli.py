"""CLI entrypoint for Cerebellum.

The old ``osmosis`` command surface remains only for compatibility. New user
flows should use ``cerebellum`` and the Cerebellum command names directly.
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h", "help"}:
        print("Usage: cerebellum <command> [args]")
        print("Commands:")
        print("  run       Start/resume a Cerebellum quant search")
        print("  watch     Open the live Cerebellum terminal interface")
        print("  status    Show a run status snapshot")
        print("  events    Show run event stream")
        print("  runs      List known runs")
        print("  system    Inspect local resources and tool availability")
        print("  plan-space Recommend scratch/offload strategy")
        print("  report    Write clean reports")
        print("  export    Export data for AI/infographics")
        print("  api       Serve JSON API for automation/web UI")
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    cerebellum_commands = {
        "run", "watch", "status", "events", "runs", "schedule", "system",
        "plan-space", "tutorial", "tips", "db", "report", "export", "auth",
        "upload", "api", "stop",
    }

    if command in cerebellum_commands:
        from osmosis.hillstep import main as hillstep_main
        hillstep_main([command] + sys.argv[1:])
    elif command == "analyze":
        from osmosis.sensitivity import main as analyze_main
        analyze_main()
    elif command == "pipeline":
        from osmosis.pipeline import main as pipeline_main
        pipeline_main()
    elif command == "load":
        from osmosis.loader import main as loader_main
        loader_main()
    elif command == "gguf":
        from osmosis.gguf_writer import main as gguf_main
        gguf_main()
    elif command == "dashboard":
        from osmosis.dashboard.server import run as dashboard_run
        dashboard_run()
    elif command == "hill":
        from osmosis.hillstep import main as hillstep_main
        hillstep_main()
    elif command in ("hill-step", "hillstep"):
        from osmosis.hillstep import main as hillstep_main
        hillstep_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
