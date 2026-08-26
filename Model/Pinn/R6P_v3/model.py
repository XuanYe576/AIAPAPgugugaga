from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .count_adapter import DipCountAdapter
from .decoder import Stage1SpectralDecoder
from .dip_experts import AttentionMoEDipExperts
from .encoder import GridGraphEncoder
from .fusion import SoftDipFusion
from .physics_adapter import AdapterOutputs, PhysicsConditioningAdapter


@dataclass
class ModelOutputs:
    coarse_line: torch.Tensor
    final_curve: torch.Tensor
    stage_features: torch.Tensor
    geometry_embedding: torch.Tensor
    geometry_reconstruction_logits: torch.Tensor
    dip_curve: torch.Tensor
    dip_presence_logits: torch.Tensor
    dip_offset_ghz: torch.Tensor
    dip_depth_db: torch.Tensor
    fusion_gate: torch.Tensor
    count_logits: torch.Tensor
    adapter_outputs: AdapterOutputs


class R6P_v3Model(nn.Module):
    def __init__(self, cfg: ExperimentConfig, freq_axis_ghz: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = GridGraphEncoder(cfg.data, cfg.model)
        self.decoder = Stage1SpectralDecoder(seq_len=int(freq_axis_ghz.numel()), model_cfg=cfg.model)
        self.adapter = PhysicsConditioningAdapter(
            freq_axis_ghz=freq_axis_ghz,
            data_cfg=cfg.data,
            model_cfg=cfg.model,
            physics_cfg=cfg.physics,
        )
        self.dip_experts = AttentionMoEDipExperts(
            seq_len=int(freq_axis_ghz.numel()),
            freq_axis_ghz=freq_axis_ghz,
            model_cfg=cfg.model,
            physics_cfg=cfg.physics,
        )
        self.count_adapter = DipCountAdapter(cfg.model)
        self.fusion = SoftDipFusion(temperature=cfg.loss.gate_temperature)

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
        count_logits = self.count_adapter(encoder_outputs.geometry_embedding)
        if not use_adapter:
            adapter_outputs = AdapterOutputs(
                conditioning=torch.zeros_like(adapter_outputs.conditioning),
                film_scale=torch.zeros_like(adapter_outputs.film_scale),
                film_shift=torch.zeros_like(adapter_outputs.film_shift),
                mode_frequencies_ghz=adapter_outputs.mode_frequencies_ghz,
                geometry_descriptors=adapter_outputs.geometry_descriptors,
            )
        if stage1_only:
            zeros = torch.zeros_like(decoder_outputs.coarse_line)
            return ModelOutputs(
                coarse_line=decoder_outputs.coarse_line,
                final_curve=decoder_outputs.coarse_line,
                stage_features=decoder_outputs.decoder_features,
                geometry_embedding=encoder_outputs.geometry_embedding,
                geometry_reconstruction_logits=encoder_outputs.geometry_reconstruction_logits,
                dip_curve=decoder_outputs.coarse_line,
                dip_presence_logits=zeros,
                dip_offset_ghz=zeros,
                dip_depth_db=zeros,
                fusion_gate=zeros,
                count_logits=count_logits,
                adapter_outputs=adapter_outputs,
            )
        dip_outputs = self.dip_experts(
            stage_features=decoder_outputs.decoder_features,
            coarse_line=decoder_outputs.coarse_line,
            adapter_outputs=adapter_outputs,
        )
        fusion_outputs = self.fusion(
            coarse_line=decoder_outputs.coarse_line,
            dip_curve=dip_outputs.dip_curve,
            dip_presence_logits=dip_outputs.dip_presence_logits,
            count_logits=count_logits,
            hard_inference=hard_inference,
        )
        return ModelOutputs(
            coarse_line=decoder_outputs.coarse_line,
            final_curve=fusion_outputs.final_curve,
            stage_features=decoder_outputs.decoder_features,
            geometry_embedding=encoder_outputs.geometry_embedding,
            geometry_reconstruction_logits=encoder_outputs.geometry_reconstruction_logits,
            dip_curve=dip_outputs.dip_curve,
            dip_presence_logits=dip_outputs.dip_presence_logits,
            dip_offset_ghz=dip_outputs.dip_offset_ghz,
            dip_depth_db=dip_outputs.dip_depth_db,
            fusion_gate=fusion_outputs.gate,
            count_logits=count_logits,
            adapter_outputs=adapter_outputs,
        )
