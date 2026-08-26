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

from mainPAP.Model.Pinn.R6P_v4.config import ExperimentConfig
from mainPAP.Model.Pinn.R6P_v4.data import RealAntennaDataset, _make_multi_dip_targets, build_dataloaders
from mainPAP.Model.Pinn.R6P_v4.decoder import Stage1SpectralDecoder
from mainPAP.Model.Pinn.R6P_v4.dip_queries import DipQueryDecoder
from mainPAP.Model.Pinn.R6P_v4.encoder import GridGraphEncoder
from mainPAP.Model.Pinn.R6P_v4.losses import embedding_regularization, extract_ground_truth_dips, set_prediction_loss
from mainPAP.Model.Pinn.R6P_v4.matching import hungarian_match
from mainPAP.Model.Pinn.R6P_v4.model import R6P_v4Model
from mainPAP.Model.Pinn.R6P_v4.physics_adapter import PhysicsConditioningAdapter
from mainPAP.Model.Pinn.R6P_v4.train import embedding_variation


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
    cfg.model.num_queries = 5
    cfg.model.query_attention_heads = 4
    cfg.model.query_ffn_hidden = 64
    cfg.model.query_head_hidden = 48
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


def test_dip_query_decoder_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    adapter = PhysicsConditioningAdapter(freq_axis, cfg.data, cfg.model, cfg.physics)
    decoder = DipQueryDecoder(model_cfg=cfg.model, physics_cfg=cfg.physics)
    geometry = make_geometry(3, cfg.data.grid_height, cfg.data.grid_width)
    stage_features = torch.randn(3, cfg.data.seq_len, cfg.model.d_model)
    coarse = torch.randn(3, cfg.data.seq_len)
    z_g = torch.randn(3, cfg.model.d_model)
    adapter_outputs = adapter(geometry, z_g)
    outputs = decoder(stage_features, coarse, adapter_outputs, freq_axis, hard_inference=False)
    assert outputs.existence_logits.shape == (3, cfg.model.num_queries)
    assert outputs.location.shape == (3, cfg.model.num_queries)
    assert outputs.depth.shape == (3, cfg.model.num_queries)
    assert outputs.dip_curve.shape == (3, cfg.data.seq_len)
    assert (outputs.location >= 0.0).all() and (outputs.location <= 1.0).all()
    assert (outputs.depth >= 0.0).all()


def test_hungarian_match_empty_ground_truth() -> None:
    existence_logits = torch.randn(2, 4)
    location = torch.rand(2, 4)
    depth = torch.rand(2, 4)
    gt_location = [torch.empty(0), torch.tensor([0.3, 0.7])]
    gt_depth = [torch.empty(0), torch.tensor([2.0, 5.0])]
    match = hungarian_match(existence_logits, location, depth, gt_location, gt_depth)
    assert match.pred_indices[0].numel() == 0
    assert match.gt_indices[0].numel() == 0
    assert match.pred_indices[1].numel() == 2
    assert match.gt_indices[1].numel() == 2


def test_hungarian_match_prefers_closest_location() -> None:
    # Two queries: one located near 0.1, one near 0.9. A single true dip at
    # 0.12 should match the query sitting near it, not the distant one.
    existence_logits = torch.zeros(1, 2)
    location = torch.tensor([[0.1, 0.9]])
    depth = torch.tensor([[3.0, 3.0]])
    gt_location = [torch.tensor([0.12])]
    gt_depth = [torch.tensor([3.0])]
    match = hungarian_match(existence_logits, location, depth, gt_location, gt_depth, location_weight=5.0)
    assert int(match.pred_indices[0].item()) == 0
    assert int(match.gt_indices[0].item()) == 0


def test_set_prediction_loss_is_finite_and_differentiable() -> None:
    cfg = make_cfg()
    existence_logits = torch.randn(3, cfg.model.num_queries, requires_grad=True)
    location = torch.rand(3, cfg.model.num_queries, requires_grad=True)
    depth = torch.rand(3, cfg.model.num_queries, requires_grad=True)

    class _Outputs:
        pass

    outputs = _Outputs()
    outputs.dip_existence_logits = existence_logits
    outputs.dip_location = location
    outputs.dip_depth = depth

    gt_locations = [torch.tensor([0.2, 0.6]), torch.empty(0), torch.tensor([0.5])]
    gt_depths = [torch.tensor([3.0, 4.0]), torch.empty(0), torch.tensor([2.5])]
    existence, location_loss, depth_loss = set_prediction_loss(outputs, gt_locations, gt_depths, cfg.loss)
    total = existence + location_loss + depth_loss
    assert torch.isfinite(total)
    total.backward()
    assert existence_logits.grad is not None
    assert location.grad is not None
    assert depth.grad is not None


def test_model_end_to_end_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    model = R6P_v4Model(cfg, freq_axis)
    geometry = make_geometry(4, cfg.data.grid_height, cfg.data.grid_width)
    outputs = model(geometry, stage1_only=False, use_adapter=True, hard_inference=False)
    assert outputs.coarse_line.shape == (4, cfg.data.seq_len)
    assert outputs.final_curve.shape == (4, cfg.data.seq_len)
    assert outputs.stage_features.shape == (4, cfg.data.seq_len, cfg.model.d_model)
    assert outputs.geometry_embedding.shape == (4, cfg.model.d_model)
    assert outputs.geometry_reconstruction_logits.shape == (4, cfg.data.input_dim)
    assert outputs.dip_existence_logits.shape == (4, cfg.model.num_queries)
    assert outputs.dip_location.shape == (4, cfg.model.num_queries)
    assert outputs.dip_depth.shape == (4, cfg.model.num_queries)


def test_model_stage1_only_shape() -> None:
    cfg = make_cfg()
    freq_axis = torch.linspace(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
    model = R6P_v4Model(cfg, freq_axis)
    geometry = make_geometry(4, cfg.data.grid_height, cfg.data.grid_width)
    outputs = model(geometry, stage1_only=True, use_adapter=False, hard_inference=False)
    assert torch.equal(outputs.final_curve, outputs.coarse_line)
    assert outputs.dip_existence_logits.shape == (4, cfg.model.num_queries)


def test_synthetic_loader_batch_shape() -> None:
    cfg = make_cfg()
    bundle = build_dataloaders(cfg)
    batch = next(iter(bundle.train_loader))
    assert batch["geometry"].shape[1:] == (cfg.data.grid_height, cfg.data.grid_width)
    assert batch["curve"].shape[1] == cfg.data.seq_len
    assert batch["dip_mask"].shape[1] == cfg.data.seq_len


def test_extract_ground_truth_dips_matches_anchor_mask() -> None:
    batch = {
        "dip_anchor_mask": torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
        "dip_depth_db": torch.tensor([[0.0, 3.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0]]),
        "label_available": torch.tensor([1.0, 0.0]),
    }
    gt_locations, gt_depths = extract_ground_truth_dips(batch)
    assert gt_locations[0].tolist() == pytest.approx([1.0 / 3.0, 1.0])
    assert gt_depths[0].tolist() == [3.0, 5.0]
    assert gt_locations[1].numel() == 0
    assert gt_depths[1].numel() == 0


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
