"""Surface entry point for training and evaluation.

Usage:
  python3 main.py [R5O args...]
  python3 main.py --model r55bs [R5.5B-s args...]
  python3 main.py --usepinn [train] [PINN args...]
  python3 main.py --model pinn [train] [PINN args...]

The script forwards the remaining arguments to the selected backend:
- `Model/R5O.py` for the default non-PINN path
- `Model/R5.5B-s.py` for the notebook-style non-PINN baseline
- `Model/Pinn/R5PINN_perF.py` for the PINN path
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SCRIPTS = {
    "r5o": REPO_ROOT / "Model" / "R5O.py",
    "r55bs": REPO_ROOT / "Model" / "R5.5B-s.py",
    "pinn": REPO_ROOT / "Model" / "Pinn" / "R5PINN_perF.py",
}
PINN_COMMANDS = {"train", "inspect", "preprocess"}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Surface launcher for R5 and R5 PINN training.",
    )
    parser.add_argument(
        "--model",
        choices=["r5o", "r55bs", "pinn"],
        default="r5o",
        help="Select which training backend to launch.",
    )
    parser.add_argument(
        "--usepinn",
        action="store_true",
        help="Backward-compatible alias for `--model pinn`.",
    )
    return parser.parse_known_args(argv)


def build_command(model_name: str, forwarded: list[str]) -> list[str]:
    script = BACKEND_SCRIPTS[model_name]
    cmd = [sys.executable, str(script)]
    if model_name == "pinn":
        if forwarded and forwarded[0] in PINN_COMMANDS:
            cmd.extend(forwarded)
        else:
            cmd.append("train")
            cmd.extend(forwarded)
    else:
        cmd.extend(forwarded)
    return cmd


def main() -> None:
    known, forwarded = parse_args(sys.argv[1:])
    model_name = "pinn" if known.usepinn else known.model
    cmd = build_command(model_name, forwarded)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
