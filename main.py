"""Project-root wrapper for `Model/main.py`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL_MAIN = Path(__file__).resolve().parent / "Model" / "main.py"


def main() -> None:
    subprocess.run([sys.executable, str(MODEL_MAIN), *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
