"""CLI entrypoint for osmosis."""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: osmosis <command> [args]")
        print("Commands:")
        print("  analyze   Run sensitivity analysis only")
        print("  pipeline  Run full overnight pipeline (download → analyze → crush)")
        print("  load      Load crushed model and generate / compare")
        print("  gguf      Convert crush output to GGUF format")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "analyze":
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
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
