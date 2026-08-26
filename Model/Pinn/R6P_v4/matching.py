from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    """Per-sample optimal assignment between predicted query slots and ground-truth dips.

    `pred_indices[b]` and `gt_indices[b]` are parallel: pred_indices[b][k] is the
    query slot matched to the k-th true dip (at gt_indices[b][k]) for sample b.
    Both are empty for samples with zero true dips.
    """

    pred_indices: list[torch.Tensor]
    gt_indices: list[torch.Tensor]


@torch.no_grad()
def hungarian_match(
    existence_logits: torch.Tensor,
    pred_location: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_location: list[torch.Tensor],
    gt_depth: list[torch.Tensor],
    *,
    class_weight: float = 1.0,
    location_weight: float = 5.0,
    depth_weight: float = 1.0,
) -> MatchResult:
    """Bipartite-match predicted query slots to ground-truth dips (DETR-style).

    Matching runs with no gradient (it's a discrete assignment, done once per
    forward pass on detached values); the loss computed afterward on the
    resulting index pairs is what actually backpropagates, exactly as in DETR.
    A rectangular (num_queries x num_true_dips) cost matrix means slots beyond
    the true count are simply left unmatched (implicitly "no object") rather
    than needing a fixed one-to-one mapping.
    """
    batch_size = existence_logits.shape[0]
    existence_prob = torch.sigmoid(existence_logits).cpu().numpy()
    location_np = pred_location.detach().cpu().numpy()
    depth_np = pred_depth.detach().cpu().numpy()

    pred_indices: list[torch.Tensor] = []
    gt_indices: list[torch.Tensor] = []
    for b in range(batch_size):
        num_gt = int(gt_location[b].shape[0])
        if num_gt == 0:
            pred_indices.append(torch.empty(0, dtype=torch.long))
            gt_indices.append(torch.empty(0, dtype=torch.long))
            continue
        gt_loc_np = gt_location[b].detach().cpu().numpy()
        gt_depth_np = gt_depth[b].detach().cpu().numpy()
        class_cost = -class_weight * existence_prob[b][:, None]
        location_cost = location_weight * abs(location_np[b][:, None] - gt_loc_np[None, :])
        depth_cost = depth_weight * abs(depth_np[b][:, None] - gt_depth_np[None, :])
        cost = class_cost + location_cost + depth_cost
        row_idx, col_idx = linear_sum_assignment(cost)
        pred_indices.append(torch.as_tensor(row_idx, dtype=torch.long))
        gt_indices.append(torch.as_tensor(col_idx, dtype=torch.long))
    return MatchResult(pred_indices=pred_indices, gt_indices=gt_indices)
