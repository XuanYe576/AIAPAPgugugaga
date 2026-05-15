"""Surface entry point for training and evaluation backends.

Usage:
  python3 Model/main.py [R5O args...]
  python3 Model/main.py --model r6o --dataset-profile 60k [R6O args...]
  python3 Model/main.py --model r55bs [R5.5B-s args...]
  python3 Model/main.py --model lgbm [LightGBM args...]
  python3 Model/main.py --usepinn [train] [PINN args...]
  python3 Model/main.py --model pinn [train] [PINN args...]
  python3 Model/main.py --model r6p [train] [R6P args...]
  python3 Model/main.py --model r6o [train] [R6O args...]

The script forwards the remaining arguments to the selected backend:
- `Model/R5O.py` for the default non-PINN path
- `Model/R5.5B-s.py` for the notebook-style non-PINN baseline
- `Model/patch_antenna_ai_colab_GPT_R4_LightGBM_with_validation_excel.py` for LightGBM
- `Model/Pinn/R5PINN_perF.py` for the PINN path
- `Model/Pinn/R6P.py` for pole-residue structured PINN path
- `Model/Pinn/R6O.py` for multi-task notch path
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
    "lgbm": REPO_ROOT / "Model" / "patch_antenna_ai_colab_GPT_R4_LightGBM_with_validation_excel.py",
    "pinn": REPO_ROOT / "Model" / "Pinn" / "R5PINN_perF.py",
    "r6p": REPO_ROOT / "Model" / "Pinn" / "R6P.py",
    "r6o": REPO_ROOT / "Model" / "Pinn" / "R6O.py",
}
PINN_COMMANDS = {"train", "inspect", "preprocess"}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Surface launcher for R5, LightGBM, and PINN training.",
    )
    parser.add_argument(
        "--model",
        choices=["r5o", "r55bs", "lgbm", "pinn", "r6p", "r6o"],
        default="r5o",
        help="Select which training backend to launch.",
    )
    parser.add_argument(
        "--usepinn",
        action="store_true",
        help="Backward-compatible alias for `--model pinn`.",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=["auto", "60k", "feed1093"],
        default="auto",
        help="Apply Trainzip dataset defaults unless already provided by user.",
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
