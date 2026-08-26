from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .matching import hungarian_match
from .model import ModelOutputs


@dataclass
class LossBreakdown:
    total: torch.Tensor
    line: torch.Tensor
    existence: torch.Tensor
    location: torch.Tensor
    depth: torch.Tensor
    geometry_reconstruction: torch.Tensor
    embedding_variance: torch.Tensor
    embedding_covariance: torch.Tensor
    final: torch.Tensor

    def as_float_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "line": float(self.line.detach().item()),
            "existence": float(self.existence.detach().item()),
            "location": float(self.location.detach().item()),
            "depth": float(self.depth.detach().item()),
            "geometry_reconstruction": float(self.geometry_reconstruction.detach().item()),
            "embedding_variance": float(self.embedding_variance.detach().item()),
            "embedding_covariance": float(self.embedding_covariance.detach().item()),
            "final": float(self.final.detach().item()),
        }


def extract_ground_truth_dips(
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Turn the dense per-bin dip labels into a variable-length set per sample:
    (normalized location in [0, 1], depth in dB) for each real dip. Samples
    with no real dip get an empty set — there is no "no dip in this batch"
    special-casing needed downstream, Hungarian matching handles it directly.

    Moves the whole batch to CPU once up front. The per-sample loop below is
    unavoidably ragged (each sample has a different dip count), and both
    `.nonzero()` and a Python-level `if tensor <= 0.5` force a CUDA
    synchronization on every call — doing that once per sample, every batch,
    every training step (and again during eval) serializes the GPU pipeline
    and was the actual cause of R6P_v4 runs stalling out around epoch 12-13.
    One transfer + a plain CPU loop removes ~2x batch_size syncs per call.
    """
    anchor_mask = batch["dip_anchor_mask"].detach().cpu()
    depth = batch["dip_depth_db"].detach().cpu()
    label_available = batch["label_available"].detach().cpu()
    seq_len = anchor_mask.shape[1]

    gt_locations: list[torch.Tensor] = []
    gt_depths: list[torch.Tensor] = []
    for b in range(anchor_mask.shape[0]):
        if label_available[b] <= 0.5:
            gt_locations.append(anchor_mask.new_zeros(0))
            gt_depths.append(depth.new_zeros(0))
            continue
        idx = (anchor_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        gt_locations.append(idx.float() / max(seq_len - 1, 1))
        gt_depths.append(depth[b, idx])
    return gt_locations, gt_depths


def embedding_regularization(
    geometry_embedding: torch.Tensor,
    target_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = geometry_embedding.float()
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    dimension_std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1.0e-8)
    variance_loss = F.relu(target_std - dimension_std).mean()
    if embedding.shape[0] <= 1:
        return variance_loss, embedding.new_zeros(())
    standardized = centered / dimension_std.clamp_min(1.0e-6).unsqueeze(0)
    correlation = standardized.T @ standardized / (embedding.shape[0] - 1)
    off_diagonal = correlation - torch.diag(torch.diag(correlation))
    covariance_loss = off_diagonal.square().sum() / embedding.shape[1]
    return variance_loss, covariance_loss


def set_prediction_loss(
    outputs: ModelOutputs,
    gt_locations: list[torch.Tensor],
    gt_depths: list[torch.Tensor],
    loss_cfg: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DETR-style matching loss: match query slots to true dips (Hungarian),
    then supervise existence on every slot (matched=1, unmatched=0, with the
    "no object" class down-weighted so the many empty slots don't dominate),
    and location/depth regression only on the matched slots.
    """
    match = hungarian_match(
        outputs.dip_existence_logits,
        outputs.dip_location,
        outputs.dip_depth,
        gt_locations,
        gt_depths,
        class_weight=loss_cfg.match_class_weight,
        location_weight=loss_cfg.match_location_weight,
        depth_weight=loss_cfg.match_depth_weight,
    )

    device = outputs.dip_existence_logits.device
    batch_size, num_queries = outputs.dip_existence_logits.shape
    existence_target = outputs.dip_existence_logits.new_zeros(batch_size, num_queries)
    location_terms: list[torch.Tensor] = []
    depth_terms: list[torch.Tensor] = []
    for b in range(batch_size):
        # match.pred_indices/gt_indices come from hungarian_match's numpy/scipy
        # call, so they're always CPU tensors regardless of where the model
        # runs. Move the prediction-side index to the model's device explicitly
        # (needed for both indexing and the existence_target assignment below)
        # rather than relying on implicit cross-device behavior.
        pred_idx = match.pred_indices[b].to(device)
        gt_idx = match.gt_indices[b]
        if pred_idx.numel() == 0:
            continue
        existence_target[b, pred_idx] = 1.0
        location_terms.append((outputs.dip_location[b, pred_idx] - gt_locations[b][gt_idx].to(device)).abs())
        depth_terms.append((outputs.dip_depth[b, pred_idx] - gt_depths[b][gt_idx].to(device)).abs())

    existence_loss_raw = F.binary_cross_entropy_with_logits(
        outputs.dip_existence_logits, existence_target, reduction="none"
    )
    slot_weight = torch.where(
        existence_target > 0.5,
        existence_target.new_ones(()),
        existence_target.new_full((), loss_cfg.no_object_weight),
    )
    existence_loss = (existence_loss_raw * slot_weight).sum() / slot_weight.sum().clamp_min(1.0)

    if location_terms:
        location_loss = torch.cat(location_terms).mean()
        depth_loss = torch.cat(depth_terms).mean()
    else:
        location_loss = outputs.dip_existence_logits.new_zeros(())
        depth_loss = outputs.dip_existence_logits.new_zeros(())

    return existence_loss, location_loss, depth_loss


def compute_losses(
    outputs: ModelOutputs,
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    *,
    stage1_only: bool = False,
) -> LossBreakdown:
    target_curve = batch["curve"]
    if loss_cfg.prune_dip_bins_from_line:
        baseline_mask = (batch["dip_mask"] <= 0.05).float()
        weighted = (outputs.coarse_line - target_curve).pow(2) * baseline_mask
        line = weighted.sum() / baseline_mask.sum().clamp_min(1.0)
    else:
        line = F.mse_loss(outputs.coarse_line, target_curve)

    geometry_target = batch["geometry"].flatten(start_dim=1)
    geometry_reconstruction = F.binary_cross_entropy_with_logits(
        outputs.geometry_reconstruction_logits,
        geometry_target,
    )
    embedding_variance, embedding_covariance = embedding_regularization(
        outputs.geometry_embedding,
        loss_cfg.embedding_target_std,
    )

    if stage1_only:
        zero = line.new_zeros(())
        total = (
            loss_cfg.line_weight * line
            + loss_cfg.geometry_reconstruction_weight * geometry_reconstruction
            + loss_cfg.embedding_variance_weight * embedding_variance
            + loss_cfg.embedding_covariance_weight * embedding_covariance
        )
        return LossBreakdown(
            total=total,
            line=line,
            existence=zero,
            location=zero,
            depth=zero,
            geometry_reconstruction=geometry_reconstruction,
            embedding_variance=embedding_variance,
            embedding_covariance=embedding_covariance,
            final=zero,
        )

    gt_locations, gt_depths = extract_ground_truth_dips(batch)
    existence, location, depth = set_prediction_loss(outputs, gt_locations, gt_depths, loss_cfg)

    final = F.mse_loss(outputs.final_curve, target_curve)

    total = (
        loss_cfg.line_weight * line
        + loss_cfg.existence_weight * existence
        + loss_cfg.location_weight * location
        + loss_cfg.depth_weight * depth
        + loss_cfg.geometry_reconstruction_weight * geometry_reconstruction
        + loss_cfg.embedding_variance_weight * embedding_variance
        + loss_cfg.embedding_covariance_weight * embedding_covariance
        + loss_cfg.final_weight * final
    )
    return LossBreakdown(
        total=total,
        line=line,
        existence=existence,
        location=location,
        depth=depth,
        geometry_reconstruction=geometry_reconstruction,
        embedding_variance=embedding_variance,
        embedding_covariance=embedding_covariance,
        final=final,
    )
