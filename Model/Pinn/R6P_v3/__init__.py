"""R6P_v3: two-stage, physics-conditioned surrogate for antenna S11 prediction."""

from .config import ExperimentConfig
from .model import ModelOutputs, R6P_v3Model

__all__ = ["ExperimentConfig", "ModelOutputs", "R6P_v3Model"]
