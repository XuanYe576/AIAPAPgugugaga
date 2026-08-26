from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _mainpap_root() -> Path:
    return _repo_root() / "mainPAP"


def _processed_csv_default() -> Path:
    return _mainpap_root() / "Data" / "processed" / "Full_60000Data_61dB.csv"


def _processed_meta_default() -> Path:
    return _mainpap_root() / "Data" / "processed" / "Full_60000Data_61dB.meta.json"


def _results_dir_default() -> Path:
    return _mainpap_root() / "results" / "R6P_v4"


@dataclass
class DataConfig:
    processed_csv_path: Path = field(default_factory=_processed_csv_default)
    processed_meta_path: Path = field(default_factory=_processed_meta_default)
    use_real_dataset: bool = True
    use_synthetic_if_missing: bool = True
    synthetic_size: int = 512
    grid_height: int = 10
    grid_width: int = 10
    seq_len: int = 61
    freq_start_ghz: float = 1.0
    freq_stop_ghz: float = 6.0
    connectivity: int = 4
    dip_target_radius_bins: int = 2
    min_dip_depth_db: float = 2.0
    dip_min_separation_bins: int = 2
    num_workers: int = 0
    max_samples: int = 0

    @property
    def input_dim(self) -> int:
        return self.grid_height * self.grid_width


@dataclass
class PhysicsConfig:
    relative_permittivity: float = 4.4
    substrate_height_m: float = 1.6e-3
    cell_size_x_m: float = 3.0e-3
    cell_size_y_m: float = 3.0e-3
    physics_modes: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1))
    conditioning_dim: int = 48
    resonance_bandwidth_scale: float = 0.18
    film_hidden_dim: int = 128
    include_global_geometry_stats: bool = True


@dataclass
class ModelConfig:
    d_model: int = 128
    encoder_hidden_1: int = 32
    encoder_hidden_2: int = 64
    geometry_mlp_hidden: int = 128
    num_spectral_blocks: int = 2
    num_transformer_layers: int = 2
    spectral_modes: int = 16
    transformer_heads: int = 8
    ffn_hidden: int = 256
    attention_window: int = 8
    attention_causal: bool = False
    dropout: float = 0.10
    dip_sigma_bins: float = 1.5
    dip_unet_depth: int = 2
    num_queries: int = 10
    query_attention_heads: int = 4
    query_ffn_hidden: int = 256
    query_head_hidden: int = 128


@dataclass
class LossConfig:
    line_weight: float = 1.0
    final_weight: float = 1.0
    existence_weight: float = 1.0
    location_weight: float = 2.0
    depth_weight: float = 1.0
    no_object_weight: float = 0.1
    match_class_weight: float = 1.0
    match_location_weight: float = 5.0
    match_depth_weight: float = 1.0
    geometry_reconstruction_weight: float = 5.0
    embedding_variance_weight: float = 25.0
    embedding_covariance_weight: float = 1.0
    embedding_target_std: float = 0.5
    prune_dip_bins_from_line: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    random_seed: int = 42
    patience: int = 10
    gradient_clip: float = 1.0
    use_amp: bool = True
    device: str = "auto"
    log_every_batches: int = 20
    embedding_std_abort_threshold: float = 1.0e-3
    embedding_min_effective_dims_99: int = 4
    output_dir: Path = field(default_factory=_results_dir_default)
    checkpoint_filename: str = "r6p_v4_best.pt"
    summary_filename: str = "summary.json"
    history_filename: str = "history.csv"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename

    @property
    def summary_path(self) -> Path:
        return self.output_dir / self.summary_filename

    @property
    def history_path(self) -> Path:
        return self.output_dir / self.history_filename


@dataclass
class ExperimentConfig:
    name: str = "R6P_v4"
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value
