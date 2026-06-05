"""Public Cerebellum CLI entrypoint."""

from __future__ import annotations

import sys

from cerebellum import main as engine_main


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "imatrix":
        from cerebellum.imatrix import main as imatrix_main

        imatrix_main(sys.argv[2:])
        return
    engine_main(sys.argv[1:])


if __name__ == "__main__":
    main()
