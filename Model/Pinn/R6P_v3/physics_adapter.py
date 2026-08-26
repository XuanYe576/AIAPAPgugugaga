from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DataConfig, ModelConfig, PhysicsConfig


C0 = 299_792_458.0


@dataclass
class AdapterOutputs:
    conditioning: torch.Tensor
    film_scale: torch.Tensor
    film_shift: torch.Tensor
    mode_frequencies_ghz: torch.Tensor
    geometry_descriptors: torch.Tensor


class PhysicsConditioningAdapter(nn.Module):
    def __init__(
        self,
        freq_axis_ghz: torch.Tensor,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        physics_cfg: PhysicsConfig,
    ) -> None:
        super().__init__()
        self.height = int(data_cfg.grid_height)
        self.width = int(data_cfg.grid_width)
        self.seq_len = int(freq_axis_ghz.numel())
        self.d_model = int(model_cfg.d_model)
        self.physics_cfg = physics_cfg
        self.register_buffer("freq_axis_ghz", freq_axis_ghz.float().reshape(1, self.seq_len, 1), persistent=False)
        descriptor_dim = 8
        mode_count = len(physics_cfg.physics_modes)
        raw_cond_dim = descriptor_dim + mode_count * 3
        self.conditioning_proj = nn.Sequential(
            nn.Linear(raw_cond_dim, physics_cfg.conditioning_dim),
            nn.GELU(),
            nn.Linear(physics_cfg.conditioning_dim, physics_cfg.conditioning_dim),
        )
        self.film_proj = nn.Sequential(
            nn.Linear(physics_cfg.conditioning_dim + model_cfg.d_model, physics_cfg.film_hidden_dim),
            nn.GELU(),
            nn.Linear(physics_cfg.film_hidden_dim, 2 * model_cfg.d_model),
        )

    def _compute_extents(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_active = (geometry.max(dim=2).values > 0.5).float()
        col_active = (geometry.max(dim=1).values > 0.5).float()
        row_any = row_active.sum(dim=1) > 0.0
        col_any = col_active.sum(dim=1) > 0.0
        first_row = row_active.argmax(dim=1)
        first_col = col_active.argmax(dim=1)
        last_row = self.height - torch.flip(row_active, dims=[1]).argmax(dim=1) - 1
        last_col = self.width - torch.flip(col_active, dims=[1]).argmax(dim=1) - 1
        default_last_row = torch.full_like(last_row, self.height - 1)
        default_last_col = torch.full_like(last_col, self.width - 1)
        first_row = torch.where(row_any, first_row, torch.zeros_like(first_row))
        first_col = torch.where(col_any, first_col, torch.zeros_like(first_col))
        last_row = torch.where(row_any, last_row, default_last_row)
        last_col = torch.where(col_any, last_col, default_last_col)
        length_cells = (last_row - first_row + 1).float().clamp_min(1.0)
        width_cells = (last_col - first_col + 1).float().clamp_min(1.0)
        return length_cells, width_cells

    def _compute_descriptors(self, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length_cells, width_cells = self._compute_extents(geometry)
        fill_factor = geometry.mean(dim=(1, 2))
        horizontal_sym = 1.0 - (geometry - geometry.flip(dims=[1])).abs().mean(dim=(1, 2))
        vertical_sym = 1.0 - (geometry - geometry.flip(dims=[2])).abs().mean(dim=(1, 2))
        area = geometry.sum(dim=(1, 2)).clamp_min(1.0)
        up = F.pad(geometry[:, :-1, :], (0, 0, 1, 0))
        down = F.pad(geometry[:, 1:, :], (0, 0, 0, 1))
        left = F.pad(geometry[:, :, :-1], (1, 0, 0, 0))
        right = F.pad(geometry[:, :, 1:], (0, 1, 0, 0))
        exposed_edges = geometry * (4.0 - up - down - left - right)
        perimeter = exposed_edges.sum(dim=(1, 2)).clamp_min(1.0)
        length_m = length_cells * self.physics_cfg.cell_size_y_m
        width_m = width_cells * self.physics_cfg.cell_size_x_m
        aspect_ratio = width_m / length_m.clamp_min(1.0e-6)
        compactness = area / perimeter
        descriptors = torch.stack(
            [
                fill_factor,
                aspect_ratio,
                horizontal_sym,
                vertical_sym,
                length_m,
                width_m,
                area / float(self.height * self.width),
                compactness,
            ],
            dim=-1,
        )
        return descriptors, torch.stack((length_m, width_m), dim=-1)

    def _compute_mode_frequencies(self, length_m: torch.Tensor, width_m: torch.Tensor) -> torch.Tensor:
        h = torch.tensor(self.physics_cfg.substrate_height_m, device=length_m.device, dtype=length_m.dtype).clamp_min(1.0e-6)
        er = torch.tensor(self.physics_cfg.relative_permittivity, device=length_m.device, dtype=length_m.dtype).clamp_min(1.0)
        eps_eff = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 + 12.0 * h / width_m.clamp_min(1.0e-6)).pow(-0.5)
        delta_l = 0.412 * h * (
            ((eps_eff + 0.3) * ((width_m / h) + 0.264))
            / ((eps_eff - 0.258) * ((width_m / h) + 0.8)).clamp_min(1.0e-6)
        )
        length_eff = length_m + 2.0 * delta_l
        width_eff = width_m + 2.0 * delta_l
        mode_freqs: list[torch.Tensor] = []
        for mode_m, mode_n in self.physics_cfg.physics_modes:
            mode_term = (mode_m / length_eff.clamp_min(1.0e-6)).pow(2)
            mode_term = mode_term + (mode_n / width_eff.clamp_min(1.0e-6)).pow(2)
            freq_hz = 0.5 * C0 * torch.sqrt(mode_term / eps_eff.clamp_min(1.0e-6))
            mode_freqs.append(freq_hz / 1.0e9)
        return torch.stack(mode_freqs, dim=-1)

    def forward(self, geometry: torch.Tensor, z_g: torch.Tensor) -> AdapterOutputs:
        if geometry.ndim != 3:
            raise ValueError(f"Expected geometry with shape (B, H, W), got {tuple(geometry.shape)}.")
        geometry = geometry.float()
        descriptors, physical_sizes = self._compute_descriptors(geometry)
        mode_freqs_ghz = self._compute_mode_frequencies(physical_sizes[:, 0], physical_sizes[:, 1])
        freq_axis = self.freq_axis_ghz.to(geometry.device)
        bandwidth = self.physics_cfg.resonance_bandwidth_scale * (
            freq_axis.max() - freq_axis.min()
        )
        bandwidth = bandwidth.clamp_min(1.0e-3)
        delta = (freq_axis - mode_freqs_ghz.unsqueeze(1)) / bandwidth
        gaussian = torch.exp(-0.5 * delta.pow(2))
        lorentz = 1.0 / (1.0 + delta.pow(2))
        descriptor_tokens = descriptors.unsqueeze(1).expand(-1, self.seq_len, -1)
        raw_conditioning = torch.cat(
            [
                delta.reshape(delta.shape[0], self.seq_len, -1),
                gaussian.reshape(gaussian.shape[0], self.seq_len, -1),
                lorentz.reshape(lorentz.shape[0], self.seq_len, -1),
                descriptor_tokens,
            ],
            dim=-1,
        )
        conditioning = self.conditioning_proj(raw_conditioning)
        film_input = torch.cat([conditioning, z_g.unsqueeze(1).expand(-1, self.seq_len, -1)], dim=-1)
        film = self.film_proj(film_input)
        film_scale, film_shift = film.chunk(2, dim=-1)
        return AdapterOutputs(
            conditioning=conditioning,
            film_scale=film_scale,
            film_shift=film_shift,
            mode_frequencies_ghz=mode_freqs_ghz,
            geometry_descriptors=descriptors,
        )
