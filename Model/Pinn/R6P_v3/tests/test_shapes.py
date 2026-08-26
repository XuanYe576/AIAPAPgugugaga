from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mainPAP.Model.Pinn.R6P_v3.config import ExperimentConfig
from mainPAP.Model.Pinn.R6P_v3.count_adapter import DipCountAdapter
from mainPAP.Model.Pinn.R6P_v3.data import RealAntennaDataset, _make_multi_dip_targets, build_dataloaders
from mainPAP.Model.Pinn.R6P_v3.decoder import Stage1SpectralDecoder
from mainPAP.Model.Pinn.R6P_v3.dip_experts import AttentionMoEDipExperts
from mainPAP.Model.Pinn.R6P_v3.encoder import GridGraphEncoder
from mainPAP.Model.Pinn.R6P_v3.fusion import SoftDipFusion
from mainPAP.Model.Pinn.R6P_v3.losses import embedding_regularization
from mainPAP.Model.Pinn.R6P_v3.model import R6P_v3Model
from mainPAP.Model.Pinn.R6P_v3.physics_adapter import PhysicsConditioningAdapter
from mainPAP.Model.Pinn.R6P_v3.train import embedding_variation


def make_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.grid_height = 6
    cfg.data.grid_width = 6
    cfg.data.seq_len = 17
    cfg.data.use_real_dataset = False
    cfg.data.synthetic_size = 12
    cfg.train.batch_size = 4
    cfg.model.d_model = 64
    cfg.model.encoder_hidden_1 = 16
    cfg.model.encoder_hidden_2 = 32
    cfg.model.geometry_mlp_hidden = 48
    cfg.model.transformer_heads = 4
    cfg.model.ffn_hidden = 96
    cfg.model.spectral_modes = 8
    cfg.physics.conditioning_dim = 24
    cfg.model.expert_hidden = 64
    cfg.model.num_experts = 3
    return cfg


def make_geometry(batch_size: int, height: int, width: int) -> torch.Tensor:
    return (torch.rand(batch_size, height, width) > 0.5).float()


def test_encoder_shape() -> None:
    cfg = make_cfg()
    encoder = GridGraphEncoder(cfg.data, cfg.model)
    geometry = make_geometry(3, cfg.data.grid_height, cfg.data.grid_width)
    outputs = encoder(geometry)
    assert outputs.node_features.shape == (3, cfg.data.input_dim, cfg.model.encoder_hidden_2)
    assert outputs.geometry_embedding.shape == (3, cfg.model.d_model)
    assert outputs.geometry_reconstruction_logits.shape == (3, cfg.data.input_dim)


def test_encoder_preserves_absolute_geometry_position() -> None:
    cfg = make_cfg()
    encoder = GridGraphEncoder(cfg.data, cfg.model)
    geometry = torch.zeros(2, cfg.data.grid_height, cfg.data.grid_width)
    geometry[0, 1, 1] = 1.0
    geometry[1, 1, cfg.data.grid_width - 2] = 1.0
    outputs = encoder(geometry)
    assert not torch.allclose(outputs.geometry_embedding[0], outputs.geometry_embedding[1])


def test_embedding_variation_reports_collapsed_batch() -> None:
    dimension_std, effective_dimensions = embedding_variation(torch.ones(4, 8))
    assert torch.equal(dimension_std, torch.zeros(8))
    assert effective_dimensions == 0


def test_embedding_regularization_penalizes_collapsed_batch() -> None:
    collapsed = torch.ones(8, 16)
    varied = torch.randn(8, 16)
    collapsed_variance, _ = embedding_regularization(collapsed, target_std=0.5)
    varied_variance, _ = embedding_regularization(varied, target_std=0.5)
    assert collapsed_variance > varied_variance


def test_decoder_shape() -> None:
    cfg = make_cfg()
    decoder = Stage1SpectralDecoder(seq_len=cfg.data.seq_len, model_cfg=cfg.model)
    z_g = torch.randn(3, cfg.model.d_model)
    outputs = decoder(z_g)
    assert outputs.decoder_features.shape == (3, cfg.data.seq_len, cfg.model.d_model)
    assert outputs.coarse_line.shape == (3, cfg.data.seq_len)


def test_physics_adapter_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    adapter = PhysicsConditioningAdapter(freq_axis, cfg.data, cfg.model, cfg.physics)
    geometry = make_geometry(3, cfg.data.grid_height, cfg.data.grid_width)
    z_g = torch.randn(3, cfg.model.d_model)
    outputs = adapter(geometry, z_g)
    assert outputs.conditioning.shape == (3, cfg.data.seq_len, cfg.physics.conditioning_dim)
    assert outputs.film_scale.shape == (3, cfg.data.seq_len, cfg.model.d_model)
    assert outputs.mode_frequencies_ghz.shape == (3, len(cfg.physics.physics_modes))


def test_dip_experts_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    adapter = PhysicsConditioningAdapter(freq_axis, cfg.data, cfg.model, cfg.physics)
    experts = AttentionMoEDipExperts(cfg.data.seq_len, freq_axis, cfg.model, cfg.physics)
    geometry = make_geometry(3, cfg.data.grid_height, cfg.data.grid_width)
    stage_features = torch.randn(3, cfg.data.seq_len, cfg.model.d_model)
    coarse = torch.randn(3, cfg.data.seq_len)
    z_g = torch.randn(3, cfg.model.d_model)
    adapter_outputs = adapter(geometry, z_g)
    outputs = experts(stage_features, coarse, adapter_outputs)
    assert outputs.dip_presence_logits.shape == (3, cfg.data.seq_len)
    assert outputs.dip_offset_ghz.shape == (3, cfg.data.seq_len)
    assert outputs.dip_depth_db.shape == (3, cfg.data.seq_len)
    assert outputs.dip_curve.shape == (3, cfg.data.seq_len)
    assert outputs.expert_gates.shape == (3, cfg.data.seq_len, cfg.model.num_experts)


def test_fusion_shape() -> None:
    fusion = SoftDipFusion(temperature=1.0)
    coarse = torch.randn(2, 11)
    dip_curve = torch.randn(2, 11)
    logits = torch.randn(2, 11)
    outputs = fusion(coarse, dip_curve, logits, hard_inference=False)
    assert outputs.final_curve.shape == (2, 11)
    assert outputs.gate.shape == (2, 11)


def test_fusion_topk_gate_uses_predicted_count() -> None:
    fusion = SoftDipFusion(temperature=1.0)
    coarse = torch.randn(2, 11)
    dip_curve = torch.randn(2, 11)
    logits = torch.randn(2, 11)
    count_logits = torch.zeros(2, 5)
    count_logits[0, 0] = 10.0  # sample 0 predicts count=0
    count_logits[1, 3] = 10.0  # sample 1 predicts count=3
    outputs = fusion(coarse, dip_curve, logits, count_logits=count_logits, hard_inference=True)
    assert outputs.gate.shape == (2, 11)
    assert outputs.gate[0].sum().item() == 0.0
    assert outputs.gate[1].sum().item() == 3.0


def test_count_adapter_shape() -> None:
    cfg = make_cfg()
    adapter = DipCountAdapter(cfg.model)
    z_g = torch.randn(3, cfg.model.d_model)
    logits = adapter(z_g)
    assert logits.shape == (3, cfg.model.max_dip_count + 1)


def test_model_end_to_end_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    model = R6P_v3Model(cfg, freq_axis)
    geometry = make_geometry(4, cfg.data.grid_height, cfg.data.grid_width)
    outputs = model(geometry, stage1_only=False, use_adapter=True, hard_inference=False)
    assert outputs.coarse_line.shape == (4, cfg.data.seq_len)
    assert outputs.final_curve.shape == (4, cfg.data.seq_len)
    assert outputs.stage_features.shape == (4, cfg.data.seq_len, cfg.model.d_model)
    assert outputs.geometry_embedding.shape == (4, cfg.model.d_model)
    assert outputs.geometry_reconstruction_logits.shape == (4, cfg.data.input_dim)
    assert outputs.count_logits.shape == (4, cfg.model.max_dip_count + 1)


def test_synthetic_loader_batch_shape() -> None:
    cfg = make_cfg()
    bundle = build_dataloaders(cfg)
    batch = next(iter(bundle.train_loader))
    assert batch["geometry"].shape[1:] == (cfg.data.grid_height, cfg.data.grid_width)
    assert batch["curve"].shape[1] == cfg.data.seq_len
    assert batch["dip_mask"].shape[1] == cfg.data.seq_len


def test_flat_finite_curve_is_valid_zero_dip_supervision() -> None:
    freq_axis = np.linspace(1.0, 6.0, 61, dtype=np.float32)
    targets = _make_multi_dip_targets(
        curve=np.zeros_like(freq_axis),
        freq_axis_ghz=freq_axis,
        target_radius_bins=1,
        min_depth_db=2.0,
        min_separation_bins=2,
    )

    assert targets["label_available"] == 1.0
    assert targets["label_confidence"] == 1.0
    assert targets["dip_count"] == 0.0
    assert targets["dip_index"] == -1
    assert not np.asarray(targets["dip_mask"]).any()
    assert not np.asarray(targets["dip_anchor_mask"]).any()
    assert np.asarray(targets["dip_supervision_mask"]).all()


def test_real_dataset_drops_non_finite_rows_and_aligns_ids(tmp_path: Path) -> None:
    cfg = make_cfg()
    cfg.data.grid_height = 2
    cfg.data.grid_width = 2
    cfg.data.seq_len = 3
    matrix = np.asarray(
        [
            [1, 0, 0, 1, -1, -2, -1],
            [0, 1, 1, 0, np.nan, -3, -2],
            [1, 1, 0, 0, -2, -4, -2],
        ],
        dtype=np.float32,
    )
    csv_path = tmp_path / "curves.csv"
    meta_path = tmp_path / "curves.meta.json"
    np.savetxt(csv_path, matrix, delimiter=",")
    meta_path.write_text(
        json.dumps({"freq_axis_ghz": [1.0, 2.0, 3.0], "matched_antenna_ids": [10, 20, 30]}),
        encoding="utf-8",
    )
    cfg.data.processed_csv_path = csv_path
    cfg.data.processed_meta_path = meta_path

    dataset = RealAntennaDataset(cfg)

    assert len(dataset) == 2
    assert dataset.antenna_ids == [10, 30]
