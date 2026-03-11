"""Surface entry point for training and evaluation.

Usage:
  python3 main.py [non-PINN args...]
  python3 main.py --usepinn [preprocess|inspect|train] [PINN args...]

The script forwards the remaining arguments to the selected backend:
- `Model/patch_antenna_ai_r5.py` for the non-PINN path
- `Model/PINN/R5PINN_perF.py` for the PINN path
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
R5_SCRIPT = REPO_ROOT / "Model" / "patch_antenna_ai_r5.py"
PINN_SCRIPT = REPO_ROOT / "Model" / "PINN" / "R5PINN_perF.py"
PINN_COMMANDS = {"preprocess", "inspect", "train"}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Surface launcher for R5 and R5 PINN training.",
    )
    parser.add_argument("--usepinn", action="store_true", help="Route to the PINN backend.")
    return parser.parse_known_args(argv)


def build_command(use_pinn: bool, forwarded: list[str]) -> list[str]:
    script = PINN_SCRIPT if use_pinn else R5_SCRIPT
    cmd = [sys.executable, str(script)]
    if use_pinn:
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
    cmd = build_command(known.usepinn, forwarded)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
