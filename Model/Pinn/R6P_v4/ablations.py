from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mainPAP.Model.Pinn.R6P_v4.train import (
        RUN_MODES,
        build_arg_parser,
        build_config_from_args,
        format_metrics_table,
        run_training,
    )
else:
    from .train import RUN_MODES, build_arg_parser, build_config_from_args, format_metrics_table, run_training


def _make_run_dir(base: Path, name: str) -> Path:
    return base / name


def _result_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {"run_mode": summary["run_mode"], "test_metrics": summary["test_metrics"]}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    base_dir = cfg.train.output_dir

    ablation_runs = [
        ("A_stage1", RUN_MODES["stage1"]),
        ("B_adapter_off", RUN_MODES["dip_no_adapter"]),
        ("B_adapter_on", RUN_MODES["full"]),
        ("D_full", RUN_MODES["full"]),
    ]
    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for name, mode in ablation_runs:
        run_cfg = replace(cfg)
        run_cfg.data = replace(cfg.data)
        run_cfg.physics = replace(cfg.physics)
        run_cfg.model = replace(cfg.model)
        run_cfg.loss = replace(cfg.loss)
        run_cfg.train = replace(cfg.train)
        run_cfg.train.output_dir = _make_run_dir(base_dir, name)
        summary = run_training(run_cfg, mode)
        summary["run_mode"] = name
        summaries[name] = summary
        rows.append(_result_row(summary))

    print("\nAblation Table")
    print(format_metrics_table(rows))

    stage1_mse = summaries["A_stage1"]["test_metrics"]["final_mse"]
    full_mse = summaries["D_full"]["test_metrics"]["final_mse"]
    off_f1 = summaries["B_adapter_off"]["test_metrics"]["dip_f1"]
    on_f1 = summaries["B_adapter_on"]["test_metrics"]["dip_f1"]
    full_f1 = summaries["D_full"]["test_metrics"]["dip_f1"]
    print("\nKey Comparisons")
    print(f"A. Stage-1 baseline final MSE: {stage1_mse:.6f}")
    print(f"B. Adapter OFF vs ON dip F1 (set-matched): {off_f1:.4f} -> {on_f1:.4f}")
    print(f"C. Full model dip F1 (set-matched): {full_f1:.4f}")
    print(f"D. Full fused vs Stage-1 final MSE: {full_mse:.6f} vs {stage1_mse:.6f}")


if __name__ == "__main__":
    main()
