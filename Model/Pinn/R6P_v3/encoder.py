from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DataConfig, ModelConfig

try:
    from torch_geometric.nn import GraphConv as PYGGraphConv

    PYG_AVAILABLE = True
except ModuleNotFoundError:
    PYGGraphConv = None
    PYG_AVAILABLE = False


def build_grid_edge_index(height: int, width: int, connectivity: int = 4) -> torch.Tensor:
    neighbors_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    neighbors_8 = neighbors_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    neighbors = neighbors_4 if connectivity == 4 else neighbors_8
    edges: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            source = row * width + col
            for d_row, d_col in neighbors:
                nbr_row = row + d_row
                nbr_col = col + d_col
                if 0 <= nbr_row < height and 0 <= nbr_col < width:
                    target = nbr_row * width + nbr_col
                    edges.append((source, target))
    if not edges:
        raise ValueError("Grid produced no edges.")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def expand_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    if batch_size == 1:
        return edge_index
    offsets = torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype) * num_nodes
    src = edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)
    dst = edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)
    return torch.stack((src.reshape(-1), dst.reshape(-1)), dim=0)


class FallbackGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(in_channels, out_channels)
        self.neighbor_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        aggregated = x.new_zeros(x.shape)
        degree = x.new_zeros((x.shape[0], 1))
        aggregated.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype))
        degree = degree.clamp_min(1.0)
        mean_neighbors = aggregated / degree
        return self.self_proj(x) + self.neighbor_proj(mean_neighbors)


def _make_graph_conv(in_channels: int, out_channels: int) -> nn.Module:
    if PYG_AVAILABLE:
        return PYGGraphConv(in_channels, out_channels)
    return FallbackGraphConv(in_channels, out_channels)


@dataclass
class EncoderOutputs:
    node_features: torch.Tensor
    pooled_features: torch.Tensor
    geometry_embedding: torch.Tensor
    geometry_reconstruction_logits: torch.Tensor


class GridGraphEncoder(nn.Module):
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.height = int(data_cfg.grid_height)
        self.width = int(data_cfg.grid_width)
        self.num_nodes = self.height * self.width
        self.conv1 = _make_graph_conv(1, model_cfg.encoder_hidden_1)
        self.conv2 = _make_graph_conv(model_cfg.encoder_hidden_1, model_cfg.encoder_hidden_2)
        self.spatial_pool = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_nodes * model_cfg.encoder_hidden_2, model_cfg.geometry_mlp_hidden),
            nn.ReLU(),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(
                model_cfg.encoder_hidden_2 + model_cfg.geometry_mlp_hidden,
                model_cfg.geometry_mlp_hidden,
            ),
            nn.ReLU(),
            nn.Linear(model_cfg.geometry_mlp_hidden, model_cfg.d_model),
        )
        self.geometry_reconstruction_head = nn.Linear(model_cfg.d_model, self.num_nodes)
        edge_index = build_grid_edge_index(self.height, self.width, connectivity=data_cfg.connectivity)
        self.register_buffer("edge_index", edge_index, persistent=False)

    def forward(self, geometry: torch.Tensor) -> EncoderOutputs:
        if geometry.ndim == 2:
            batch_size = geometry.shape[0]
            grid = geometry.reshape(batch_size, self.height, self.width)
        elif geometry.ndim == 3:
            batch_size = geometry.shape[0]
            grid = geometry
        else:
            raise ValueError(f"Expected geometry with 2 or 3 dims, got {tuple(geometry.shape)}.")
        node_values = grid.reshape(batch_size * self.num_nodes, 1)
        batched_edge_index = expand_edge_index(self.edge_index, num_nodes=self.num_nodes, batch_size=batch_size)
        batched_edge_index = batched_edge_index.to(node_values.device)
        x = F.relu(self.conv1(node_values, batched_edge_index))
        x = F.relu(self.conv2(x, batched_edge_index))
        x_nodes = x.reshape(batch_size, self.num_nodes, -1)
        pooled = x_nodes.mean(dim=1)
        spatial = self.spatial_pool(x_nodes)
        z_g = self.post_pool(torch.cat((pooled, spatial), dim=-1))
        geometry_reconstruction_logits = self.geometry_reconstruction_head(z_g)
        return EncoderOutputs(
            node_features=x_nodes,
            pooled_features=pooled,
            geometry_embedding=z_g,
            geometry_reconstruction_logits=geometry_reconstruction_logits,
        )
