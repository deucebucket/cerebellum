"""Public Cerebellum imatrix entrypoint."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Cerebellum/llama.cpp imatrix for quantization"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--output", required=True, help="Output imatrix file path")
    parser.add_argument(
        "--mode",
        choices=["stream", "calibrated"],
        default="stream",
        help="stream uses safetensors one tensor at a time; calibrated loads the model and blends activation stats",
    )
    parser.add_argument("--no-calibrate", action="store_true", help="For --mode calibrated, skip activation calibration")
    parser.add_argument("--num-samples", type=int, default=8, help="Calibration prompt count for --mode calibrated")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "stream":
        from osmosis.imatrix_stream import generate_imatrix_streaming

        generate_imatrix_streaming(args.model, args.output, verbose=args.verbose)
        return

    from osmosis.imatrix_gen import generate_imatrix

    generate_imatrix(
        args.model,
        args.output,
        calibrate=not args.no_calibrate,
        num_samples=args.num_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
