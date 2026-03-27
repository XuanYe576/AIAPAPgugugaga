"""
1) config
2) load data
3) train/val split
4) train 61 separate regressors
5) save frequency-wise error plots
6) expose full-curve prediction helper
7) export validation predictions to Excel
8) save one example prediction curve
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_csv_path() -> Path:
    candidates = [
        REPO_ROOT / "Data" / "processed" / "Full_60000Data_61dB.csv",
        REPO_ROOT / "Trainzip" / "Data" / "60k61db.csv",
        REPO_ROOT / "Data" / "Ori_30k.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_output_dir() -> Path:
    return REPO_ROOT / "results" / "LightGBM_R4"


@dataclass
class CFG:
    csv_path: Path = field(default_factory=_default_csv_path)
    data_dir: Path | None = None
    csv_pattern: str = "*.csv"
    in_dim: int = 100
    out_dim: int = 61
    test_size: float = 0.2
    seed: int = 42
    early_stopping_rounds: int = 200
    eval_metric: str = "rmse"
    output_dir: Path = field(default_factory=_default_output_dir)

    @property
    def save_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def results_xlsx(self) -> Path:
        return self.output_dir / "validation_s11_predictions_vs_real.xlsx"

    @property
    def model_path(self) -> Path:
        return self.save_dir / "lgbm_models_61.joblib"

    @property
    def rmse_plot_path(self) -> Path:
        return self.output_dir / "rmse_per_frequency.png"

    @property
    def mae_plot_path(self) -> Path:
        return self.output_dir / "mae_per_frequency.png"

    @property
    def example_plot_path(self) -> Path:
        return self.output_dir / "example_curve.png"

    @property
    def summary_json_path(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def lgb_params(self) -> dict[str, object]:
        return dict(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=self.seed,
            n_jobs=-1,
        )


def parse_args() -> CFG:
    cfg = CFG()
    parser = argparse.ArgumentParser(
        description="Train the R4 LightGBM baseline on 100-bit geometry + 61-point S11 CSV data.",
    )
    parser.add_argument("--csv-path", type=Path, default=cfg.csv_path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional folder of CSV files to concatenate. If set, overrides --csv-path.",
    )
    parser.add_argument("--csv-pattern", type=str, default=cfg.csv_pattern)
    parser.add_argument("--test-size", type=float, default=cfg.test_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    args = parser.parse_args()

    cfg.csv_path = args.csv_path
    cfg.data_dir = args.data_dir
    cfg.csv_pattern = args.csv_pattern
    cfg.test_size = args.test_size
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir
    return cfg


def frequency_axis_ghz(cfg: CFG) -> np.ndarray:
    return np.linspace(1.0, 6.0, cfg.out_dim)


def load_csv_files(cfg: CFG) -> list[Path]:
    if cfg.data_dir is not None:
        files = sorted(Path(path) for path in glob.glob(str(cfg.data_dir / cfg.csv_pattern)))
    else:
        files = [cfg.csv_path]
    if not files:
        raise FileNotFoundError("No CSV files found for LightGBM training.")
    return files


def load_dataset(cfg: CFG) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    csv_files = load_csv_files(cfg)
    print("Found CSV files:")
    for path in csv_files:
        print(f"  {path}")

    df_list = [pd.read_csv(path, header=None) for path in csv_files]
    df = pd.concat(df_list, axis=0, ignore_index=True)
    print("Combined data shape:", df.shape)

    expected_cols = cfg.in_dim + cfg.out_dim
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns (100 geom + 61 S11), got {df.shape[1]}"
        )

    X = df.iloc[:, : cfg.in_dim].to_numpy()
    Y = df.iloc[:, cfg.in_dim : cfg.in_dim + cfg.out_dim].to_numpy()
    X = (X > 0.5).astype(np.int8)

    print("X shape:", X.shape, "Y shape:", Y.shape)
    print("X sample (first 20 bits):", X[0, :20])
    print("Y sample (first 5 S11 pts):", Y[0, :5])
    return X, Y, csv_files


def predict_s11(geometry_bits_100, models_61, cfg: CFG) -> np.ndarray:
    x = np.asarray(geometry_bits_100).reshape(1, -1)
    if x.shape[1] != cfg.in_dim:
        raise ValueError(f"Expected geometry length {cfg.in_dim}, got {x.shape[1]}")
    x = (x > 0.5).astype(np.int8)
    return np.array([model.predict(x)[0] for model in models_61], dtype=float)


def save_metric_plots(
    cfg: CFG,
    freq_ghz: np.ndarray,
    train_rmse: np.ndarray,
    val_rmse: np.ndarray,
    val_mae: np.ndarray,
) -> None:
    plt.figure()
    plt.plot(freq_ghz, train_rmse, label="Train RMSE")
    plt.plot(freq_ghz, val_rmse, label="Val RMSE")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("RMSE (dB)")
    plt.title("LightGBM RMSE per S11 Frequency Point")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.rmse_plot_path, dpi=180)
    plt.close()

    plt.figure()
    plt.plot(freq_ghz, val_mae, label="Val MAE")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("MAE (dB)")
    plt.title("LightGBM MAE per S11 Frequency Point")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.mae_plot_path, dpi=180)
    plt.close()


def export_validation_excel(
    cfg: CFG,
    freq_ghz: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Y_val_pred: np.ndarray,
    val_rmse: np.ndarray,
    val_mae: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    freq_labels = [f"{f:.4f}GHz" for f in freq_ghz]
    geom_cols = [f"geom_{i+1:03d}" for i in range(cfg.in_dim)]
    real_cols = [f"real_S11_{lbl}" for lbl in freq_labels]
    pred_cols = [f"pred_S11_{lbl}" for lbl in freq_labels]

    val_results_df = pd.DataFrame(
        np.hstack([X_val, Y_val, Y_val_pred]),
        columns=geom_cols + real_cols + pred_cols,
    )
    val_results_df.insert(0, "val_sample_id", np.arange(len(val_results_df)))

    sample_rmse = np.sqrt(np.mean((Y_val_pred - Y_val) ** 2, axis=1))
    sample_mae = np.mean(np.abs(Y_val_pred - Y_val), axis=1)
    val_results_df["curve_rmse_db"] = sample_rmse
    val_results_df["curve_mae_db"] = sample_mae

    long_rows: list[dict[str, float | int]] = []
    for i in range(X_val.shape[0]):
        geom_dict = {geom_cols[g]: int(X_val[i, g]) for g in range(cfg.in_dim)}
        for k, f in enumerate(freq_ghz):
            row = {
                "val_sample_id": i,
                "frequency_GHz": float(f),
                "real_S11_dB": float(Y_val[i, k]),
                "pred_S11_dB": float(Y_val_pred[i, k]),
                "abs_error_dB": float(abs(Y_val_pred[i, k] - Y_val[i, k])),
                "curve_rmse_db": float(sample_rmse[i]),
                "curve_mae_db": float(sample_mae[i]),
            }
            row.update(geom_dict)
            long_rows.append(row)

    val_results_long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(
        {
            "frequency_GHz": freq_ghz,
            "val_rmse_dB": val_rmse,
            "val_mae_dB": val_mae,
        }
    )

    with pd.ExcelWriter(cfg.results_xlsx, engine="openpyxl") as writer:
        val_results_df.to_excel(writer, sheet_name="val_results_wide", index=False)
        val_results_long_df.to_excel(writer, sheet_name="val_results_long", index=False)
        summary_df.to_excel(writer, sheet_name="freq_summary", index=False)

    print("Validation results saved to:", cfg.results_xlsx)
    print("Wide sheet shape:", val_results_df.shape)
    print("Long sheet shape:", val_results_long_df.shape)
    return val_results_df, val_results_long_df, summary_df


def save_example_curve(
    cfg: CFG,
    freq_ghz: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    models: list[LGBMRegressor],
) -> dict[str, float]:
    idx = np.random.randint(0, X_val.shape[0])
    y_true = Y_val[idx]
    y_pred = predict_s11(X_val[idx], models, cfg)

    plt.figure()
    plt.plot(freq_ghz, y_true, label="True")
    plt.plot(freq_ghz, y_pred, label="Predicted")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Example: True vs Predicted S11 Curve (LightGBM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.example_plot_path, dpi=180)
    plt.close()

    metrics = {
        "example_index": int(idx),
        "curve_rmse_db": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "curve_mae_db": float(mean_absolute_error(y_true, y_pred)),
    }
    print("Example curve metrics:")
    print("Curve RMSE (dB):", metrics["curve_rmse_db"])
    print("Curve MAE (dB): ", metrics["curve_mae_db"])
    print("Example plot:", cfg.example_plot_path)
    return metrics


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1) Config
    # =========================
    freq_ghz = frequency_axis_ghz(cfg)

    # =========================
    # 2) Load data
    # =========================
    X, Y, csv_files = load_dataset(cfg)

    # =========================
    # 3) Train/Val split
    # =========================
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        shuffle=True,
    )
    print("Train:", X_train.shape, Y_train.shape)
    print("Val:  ", X_val.shape, Y_val.shape)

    # =========================
    # 4) Train 61 separate regressors
    # =========================
    models: list[LGBMRegressor] = []
    train_rmse = np.zeros(cfg.out_dim, dtype=float)
    val_rmse = np.zeros(cfg.out_dim, dtype=float)
    val_mae = np.zeros(cfg.out_dim, dtype=float)

    for k in range(cfg.out_dim):
        ytr = Y_train[:, k]
        yva = Y_val[:, k]

        model = LGBMRegressor(**cfg.lgb_params)
        model.fit(
            X_train,
            ytr,
            eval_set=[(X_val, yva)],
            eval_metric=cfg.eval_metric,
            callbacks=[
                early_stopping(cfg.early_stopping_rounds, verbose=False),
                log_evaluation(0),
            ],
        )

        ytr_pred = model.predict(X_train)
        yva_pred = model.predict(X_val)

        train_rmse[k] = np.sqrt(mean_squared_error(ytr, ytr_pred))
        val_rmse[k] = np.sqrt(mean_squared_error(yva, yva_pred))
        val_mae[k] = mean_absolute_error(yva, yva_pred)

        models.append(model)
        if (k % 10 == 0) or (k == cfg.out_dim - 1):
            print(
                f"[{k:02d}/{cfg.out_dim - 1:02d}] "
                f"Train RMSE={train_rmse[k]:.4f} | Val RMSE={val_rmse[k]:.4f}"
            )

    joblib.dump(models, cfg.model_path)
    print("Saved models to:", cfg.model_path)

    # =========================
    # 5) Evaluate: plot error vs frequency
    # =========================
    save_metric_plots(cfg, freq_ghz, train_rmse, val_rmse, val_mae)

    # =========================
    # 6) Predict full curves
    # =========================
    Y_val_pred = np.column_stack([model.predict(X_val) for model in models])

    # =========================
    # 7) Export validation predictions to Excel
    # =========================
    export_validation_excel(cfg, freq_ghz, X_val, Y_val, Y_val_pred, val_rmse, val_mae)

    # =========================
    # 8) Save one example curve
    # =========================
    example_metrics = save_example_curve(cfg, freq_ghz, X_val, Y_val, models)

    summary = {
        "csv_files": [str(path) for path in csv_files],
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "mean_train_rmse_db": float(train_rmse.mean()),
        "mean_val_rmse_db": float(val_rmse.mean()),
        "mean_val_mae_db": float(val_mae.mean()),
        "model_path": str(cfg.model_path),
        "excel_path": str(cfg.results_xlsx),
        "rmse_plot_path": str(cfg.rmse_plot_path),
        "mae_plot_path": str(cfg.mae_plot_path),
        "example_plot_path": str(cfg.example_plot_path),
        **example_metrics,
    }
    with cfg.summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Summary saved to:", cfg.summary_json_path)


if __name__ == "__main__":
    main()
