from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .decoder import Stage1SpectralDecoder
from .dip_queries import DipQueryDecoder
from .encoder import GridGraphEncoder
from .physics_adapter import AdapterOutputs, PhysicsConditioningAdapter


@dataclass
class ModelOutputs:
    coarse_line: torch.Tensor
    final_curve: torch.Tensor
    stage_features: torch.Tensor
    geometry_embedding: torch.Tensor
    geometry_reconstruction_logits: torch.Tensor
    dip_curve: torch.Tensor
    dip_existence_logits: torch.Tensor
    dip_location: torch.Tensor
    dip_depth: torch.Tensor
    adapter_outputs: AdapterOutputs


class R6P_v4Model(nn.Module):
    def __init__(self, cfg: ExperimentConfig, freq_axis_ghz: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("freq_axis_ghz", freq_axis_ghz.float(), persistent=False)
        self.encoder = GridGraphEncoder(cfg.data, cfg.model)
        self.decoder = Stage1SpectralDecoder(seq_len=int(freq_axis_ghz.numel()), model_cfg=cfg.model)
        self.adapter = PhysicsConditioningAdapter(
            freq_axis_ghz=freq_axis_ghz,
            data_cfg=cfg.data,
            model_cfg=cfg.model,
            physics_cfg=cfg.physics,
        )
        self.dip_queries = DipQueryDecoder(model_cfg=cfg.model, physics_cfg=cfg.physics)

    def forward(
        self,
        geometry: torch.Tensor,
        *,
        stage1_only: bool = False,
        use_adapter: bool = True,
        hard_inference: bool = False,
    ) -> ModelOutputs:
        encoder_outputs = self.encoder(geometry)
        decoder_outputs = self.decoder(encoder_outputs.geometry_embedding)
        adapter_outputs = self.adapter(geometry, encoder_outputs.geometry_embedding)
        if not use_adapter:
            adapter_outputs = AdapterOutputs(
                conditioning=torch.zeros_like(adapter_outputs.conditioning),
                film_scale=torch.zeros_like(adapter_outputs.film_scale),
                film_shift=torch.zeros_like(adapter_outputs.film_shift),
                mode_frequencies_ghz=adapter_outputs.mode_frequencies_ghz,
                geometry_descriptors=adapter_outputs.geometry_descriptors,
            )
        if stage1_only:
            num_queries = self.dip_queries.num_queries
            zeros_q = decoder_outputs.coarse_line.new_zeros(decoder_outputs.coarse_line.shape[0], num_queries)
            return ModelOutputs(
                coarse_line=decoder_outputs.coarse_line,
                final_curve=decoder_outputs.coarse_line,
                stage_features=decoder_outputs.decoder_features,
                geometry_embedding=encoder_outputs.geometry_embedding,
                geometry_reconstruction_logits=encoder_outputs.geometry_reconstruction_logits,
                dip_curve=decoder_outputs.coarse_line,
                dip_existence_logits=zeros_q,
                dip_location=zeros_q,
                dip_depth=zeros_q,
                adapter_outputs=adapter_outputs,
            )
        dip_outputs = self.dip_queries(
            stage_features=decoder_outputs.decoder_features,
            coarse_line=decoder_outputs.coarse_line,
            adapter_outputs=adapter_outputs,
            freq_axis_ghz=self.freq_axis_ghz,
            hard_inference=hard_inference,
        )
        return ModelOutputs(
            coarse_line=decoder_outputs.coarse_line,
            final_curve=dip_outputs.dip_curve,
            stage_features=decoder_outputs.decoder_features,
            geometry_embedding=encoder_outputs.geometry_embedding,
            geometry_reconstruction_logits=encoder_outputs.geometry_reconstruction_logits,
            dip_curve=dip_outputs.dip_curve,
            dip_existence_logits=dip_outputs.existence_logits,
            dip_location=dip_outputs.location,
            dip_depth=dip_outputs.depth,
            adapter_outputs=adapter_outputs,
        )
