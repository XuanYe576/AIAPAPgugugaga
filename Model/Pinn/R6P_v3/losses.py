from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .model import ModelOutputs


@dataclass
class LossBreakdown:
    total: torch.Tensor
    line: torch.Tensor
    mask: torch.Tensor
    iou: torch.Tensor
    dip: torch.Tensor
    count: torch.Tensor
    geometry_reconstruction: torch.Tensor
    embedding_variance: torch.Tensor
    embedding_covariance: torch.Tensor
    final: torch.Tensor
    physics_aux: torch.Tensor

    def as_float_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().item()),
            "line": float(self.line.detach().item()),
            "mask": float(self.mask.detach().item()),
            "iou": float(self.iou.detach().item()),
            "dip": float(self.dip.detach().item()),
            "count": float(self.count.detach().item()),
            "geometry_reconstruction": float(self.geometry_reconstruction.detach().item()),
            "embedding_variance": float(self.embedding_variance.detach().item()),
            "embedding_covariance": float(self.embedding_covariance.detach().item()),
            "final": float(self.final.detach().item()),
            "physics_aux": float(self.physics_aux.detach().item()),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def soft_iou_loss(pred_prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Differentiable per-sample IoU loss over a dense (multi-dip) presence mask.

    Unlike per-bin BCE, IoU is computed over the whole 61-bin mask at once, so
    it directly rewards getting the *set* of dip bins right regardless of how
    many real resonances a sample has, instead of averaging bin-level errors
    that a handful of true-positive bins can barely move against ~60 true
    negatives.
    """
    weighted_pred = pred_prob * mask
    weighted_target = target * mask
    intersection = (weighted_pred * weighted_target).sum(dim=1)
    union = (weighted_pred + weighted_target - weighted_pred * weighted_target).sum(dim=1)
    iou = (intersection + eps) / (union + eps)
    sample_has_support = (mask.sum(dim=1) > 0).float()
    loss = (1.0 - iou) * sample_has_support
    denom = sample_has_support.sum().clamp_min(1.0)
    return loss.sum() / denom


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


def compute_losses(
    outputs: ModelOutputs,
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    *,
    stage1_only: bool = False,
) -> LossBreakdown:
    target_curve = batch["curve"]
    if loss_cfg.prune_dip_bins_from_line:
        # Stage 1 only ever has to fit the smooth baseline; bins a real dip
        # touches are excluded from its loss entirely instead of forcing it
        # to compromise between the baseline and dips it can't represent.
        baseline_mask = (batch["dip_mask"] <= 0.05).float()
        line = _masked_mean((outputs.coarse_line - target_curve).pow(2), baseline_mask)
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
            mask=zero,
            iou=zero,
            dip=zero,
            count=zero,
            geometry_reconstruction=geometry_reconstruction,
            embedding_variance=embedding_variance,
            embedding_covariance=embedding_covariance,
            final=zero,
            physics_aux=zero,
        )

    dip_target = batch["dip_mask"]
    supervision_mask = batch["dip_supervision_mask"] * batch["label_available"].unsqueeze(-1)
    pos_weight = torch.tensor([loss_cfg.mask_positive_weight], device=dip_target.device, dtype=dip_target.dtype)
    mask_loss_raw = F.binary_cross_entropy_with_logits(
        outputs.dip_presence_logits,
        dip_target,
        reduction="none",
        pos_weight=pos_weight,
    )
    mask = _masked_mean(mask_loss_raw, supervision_mask)

    pred_prob = torch.sigmoid(outputs.dip_presence_logits)
    iou = soft_iou_loss(pred_prob, dip_target, supervision_mask)

    anchor_mask = batch["dip_anchor_mask"] * batch["label_available"].unsqueeze(-1)
    offset_loss_raw = (outputs.dip_offset_ghz - batch["dip_offset_ghz"]).pow(2)
    depth_loss_raw = (outputs.dip_depth_db - batch["dip_depth_db"]).pow(2)
    dip = _masked_mean(offset_loss_raw + depth_loss_raw, anchor_mask)

    max_count = outputs.count_logits.shape[-1] - 1
    target_count = batch["dip_count"].round().long().clamp(0, max_count)
    count = F.cross_entropy(outputs.count_logits, target_count)

    final = F.mse_loss(outputs.final_curve, target_curve)

    if loss_cfg.use_physics_aux:
        pred_anchor = outputs.dip_presence_logits.argmax(dim=1)
        pred_freq = outputs.adapter_outputs.mode_frequencies_ghz.mean(dim=1)
        target_freq = batch["dip_index"].clamp_min(0)
        first_mode = outputs.adapter_outputs.mode_frequencies_ghz[:, 0]
        anchor_freq = outputs.adapter_outputs.mode_frequencies_ghz.new_zeros(first_mode.shape)
        valid = batch["label_available"] > 0.5
        anchor_freq[valid] = first_mode[valid]
        physics_aux = ((pred_anchor.float() - target_freq.float()).abs() / outputs.coarse_line.shape[1]).mean()
        physics_aux = physics_aux + (pred_freq.mean() - anchor_freq.mean()).abs() * 0.0
    else:
        physics_aux = line.new_zeros(())

    total = (
        loss_cfg.line_weight * line
        + loss_cfg.mask_weight * mask
        + loss_cfg.iou_weight * iou
        + loss_cfg.dip_weight * dip
        + loss_cfg.count_weight * count
        + loss_cfg.geometry_reconstruction_weight * geometry_reconstruction
        + loss_cfg.embedding_variance_weight * embedding_variance
        + loss_cfg.embedding_covariance_weight * embedding_covariance
        + loss_cfg.final_weight * final
        + loss_cfg.physics_aux_weight * physics_aux
    )
    return LossBreakdown(
        total=total,
        line=line,
        mask=mask,
        iou=iou,
        dip=dip,
        count=count,
        geometry_reconstruction=geometry_reconstruction,
        embedding_variance=embedding_variance,
        embedding_covariance=embedding_covariance,
        final=final,
        physics_aux=physics_aux,
    )
