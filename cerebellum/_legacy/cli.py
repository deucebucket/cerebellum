"""CLI entrypoint for Cerebellum.

The old ``osmosis`` command surface remains only for compatibility. New user
flows should use ``cerebellum`` and the Cerebellum command names directly.
"""
import sys


def main():
    if len(sys.argv) < 2:
        from cerebellum import main as hillstep_main
        hillstep_main([])
        sys.exit(0)

    if sys.argv[1] in {"--help", "-h", "help"}:
        print("Usage: cerebellum <command> [args]")
        print("Commands:")
        print("  home      Show the Cerebellum local menu and recent run summary")
        print("  run       Start/resume a Cerebellum quant search")
        print("  resume    Resume an existing run from manifest/state")
        print("  watch     Open the live Cerebellum terminal interface")
        print("  status    Show a run status snapshot")
        print("  events    Show run event stream")
        print("  recover   Print a crash-recovery plan")
        print("  cleanup   Clean safe temp/artifact files")
        print("  rollback  Roll durable state back to a clean boundary")
        print("  backup    Mirror critical run metadata/checkpoints")
        print("  runs      List known runs")
        print("  project   Inspect Cerebellum model projects")
        print("  legacy-flow Write automated group-first targeted-hillstep flow")
        print("  sparse-replay Run OG sparse ablation replay pipeline")
        print("  provenance Inspect or generate Cerebellum GGUF provenance")
        print("  inspect-gguf-types Summarize GGUF tensor quantization types")
        print("  compare-locks Compare tensor locks between a run and archive/state")
        print("  history   Build/search a browsable Cerebellum model history index")
        print("  finalize  Write final reports/model card and tag GGUF provenance")
        print("  package   Write portable upload/package manifest")
        print("  system    Inspect local resources and tool availability")
        print("  doctor    Check portable setup and explain fixes")
        print("  self-test Run read-only CLI/API smoke checks")
        print("  plan-space Recommend scratch/offload strategy")
        print("  report    Write clean reports")
        print("  export    Export data for AI/infographics")
        print("  imatrix   Generate Cerebellum imatrix files")
        print("  api       Serve JSON API for automation/web UI")
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    cerebellum_commands = {
        "run", "watch", "status", "events", "runs", "schedule", "system",
        "doctor", "self-test", "provenance", "finalize", "package", "plan-space", "tutorial", "tips", "db", "report", "export", "auth",
        "upload", "api", "stop", "resume", "recover", "cleanup", "rollback", "backup",
        "project", "home", "legacy-plan", "legacy-flow", "public-card-policy", "inspect-gguf-types", "compare-gguf-types", "compare-locks",
        "history", "sparse-replay",
        "benchmark-report", "benchmark-plan", "benchmark-run", "benchmark-status", "benchmark-audit",
    }

    if command == "imatrix":
        from cerebellum.imatrix import main as imatrix_main
        imatrix_main(sys.argv[1:])
    elif command in cerebellum_commands:
        from cerebellum import main as hillstep_main
        hillstep_main([command] + sys.argv[1:])
    elif command == "analyze":
        from cerebellum._legacy.sensitivity import main as analyze_main
        analyze_main()
    elif command == "pipeline":
        from cerebellum._legacy.pipeline import main as pipeline_main
        pipeline_main()
    elif command == "load":
        from cerebellum._legacy.loader import main as loader_main
        loader_main()
    elif command == "gguf":
        from cerebellum._legacy.gguf_writer import main as gguf_main
        gguf_main()
    elif command == "dashboard":
        from cerebellum.dashboard.server import run as dashboard_run
        dashboard_run()
    elif command == "hill":
        from cerebellum import main as hillstep_main
        hillstep_main()
    elif command in ("hill-step", "hillstep"):
        from cerebellum import main as hillstep_main
        hillstep_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
