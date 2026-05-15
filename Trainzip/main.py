"""Surface entry point for training and evaluation in Trainzip.

Supports dataset profiles (`60k` / `feed1093`) for backend-specific default paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_SCRIPTS = {
    "r5o": REPO_ROOT / "Model" / "R5O.py",
    "r55bs": REPO_ROOT / "Model" / "R5.5B-s.py",
    "pinn": REPO_ROOT / "Model" / "Pinn" / "R5PINN_perF.py",
    "r6p": REPO_ROOT / "Model" / "Pinn" / "R6P.py",
    "r6o": REPO_ROOT / "Model" / "Pinn" / "R6O.py",
}
PINN_COMMANDS = {"train", "inspect", "preprocess"}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Trainzip launcher for R5/R6 backends.")
    parser.add_argument(
        "--model",
        choices=["r5o", "r55bs", "pinn", "r6p", "r6o"],
        default="r5o",
        help="Select backend model.",
    )
    parser.add_argument("--usepinn", action="store_true", help="Backward-compatible alias for `--model pinn`.")
    parser.add_argument(
        "--dataset-profile",
        choices=["auto", "60k", "feed1093"],
        default="auto",
        help="Apply dataset defaults for selected model unless already provided by user.",
    )
    return parser.parse_known_args(argv)


def _has_arg(forwarded: list[str], key: str) -> bool:
    return key in forwarded


def _apply_dataset_profile(model_name: str, forwarded: list[str], profile: str) -> list[str]:
    if profile == "auto":
        return list(forwarded)

    out = list(forwarded)
    data_dir = REPO_ROOT / "Data"
    if profile == "60k":
        if model_name in {"r5o", "r55bs"} and not _has_arg(out, "--csv-path"):
            out.extend(["--csv-path", str(data_dir / "60k61db.csv"), "--output-mode", "mag_only"])
        if model_name in {"r5o", "r55bs"} and not _has_arg(out, "--target-curve-points"):
            out.extend(["--target-curve-points", "501"])
        if model_name == "r5o" and not _has_arg(out, "--seq-len"):
            out.extend(["--seq-len", "501"])
        if model_name in {"pinn", "r6p", "r6o"}:
            if not _has_arg(out, "--processed-csv-path"):
                out.extend(["--processed-csv-path", str(data_dir / "60k61db.csv")])
            if not _has_arg(out, "--processed-meta-path"):
                out.extend(["--processed-meta-path", str(data_dir / "60k61db.meta.json")])
        if model_name in {"r6p", "r6o"} and not _has_arg(out, "--target-curve-points"):
            out.extend(["--target-curve-points", "501"])
        if model_name == "r6o":
            if not _has_arg(out, "--train-index-path"):
                out.extend(["--train-index-path", str(data_dir / "train_60k.txt")])
            if not _has_arg(out, "--val-index-path"):
                out.extend(["--val-index-path", str(data_dir / "val_60k.txt")])
            if not _has_arg(out, "--labels-csv-path"):
                out.extend(["--labels-csv-path", str(data_dir / "resonance_labels_60k.csv")])

    if profile == "feed1093":
        if model_name in {"r5o", "r55bs"} and not _has_arg(out, "--csv-path"):
            out.extend(["--csv-path", str(data_dir / "feedpatch_1093_501.csv"), "--output-mode", "mag_only"])
        if model_name == "r5o" and not _has_arg(out, "--seq-len"):
            out.extend(["--seq-len", "501"])

        if model_name in {"pinn", "r6p", "r6o"}:
            if not _has_arg(out, "--processed-csv-path"):
                out.extend(["--processed-csv-path", str(data_dir / "feedpatch_1093_501.csv")])
            if not _has_arg(out, "--processed-meta-path"):
                out.extend(["--processed-meta-path", str(data_dir / "feedpatch_1093_501.meta.json")])
        if model_name == "r6o":
            if not _has_arg(out, "--train-index-path"):
                out.extend(["--train-index-path", str(data_dir / "train_feedpatch_1093.txt")])
            if not _has_arg(out, "--val-index-path"):
                out.extend(["--val-index-path", str(data_dir / "val_feedpatch_1093.txt")])
            if not _has_arg(out, "--labels-csv-path"):
                out.extend(["--labels-csv-path", str(data_dir / "resonance_labels_feedpatch_1093.csv")])
            if not _has_arg(out, "--use-feed-channel"):
                out.append("--use-feed-channel")

    return out


def build_command(model_name: str, forwarded: list[str], dataset_profile: str) -> list[str]:
    script = BACKEND_SCRIPTS[model_name]
    profiled_forwarded = _apply_dataset_profile(model_name, forwarded, dataset_profile)
    cmd = [sys.executable, str(script)]

    if model_name in {"pinn", "r6p", "r6o"}:
        if profiled_forwarded and profiled_forwarded[0] in PINN_COMMANDS:
            cmd.extend(profiled_forwarded)
        else:
            cmd.append("train")
            cmd.extend(profiled_forwarded)
    else:
        cmd.extend(profiled_forwarded)
    return cmd


def main() -> None:
    known, forwarded = parse_args(sys.argv[1:])
    model_name = "pinn" if known.usepinn else known.model
    cmd = build_command(model_name, forwarded, known.dataset_profile)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
