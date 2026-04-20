from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelInputBatch:
    sample: torch.Tensor
    condition: torch.Tensor
    condition_type_ids: torch.Tensor
    timestep: torch.Tensor
    target: torch.Tensor


@dataclass
class TrainStepOutput:
    loss_total: torch.Tensor
    loss_dict: dict[str, torch.Tensor]
    pred: torch.Tensor | None
    target: torch.Tensor | None
    sample: torch.Tensor | None = None
    condition: torch.Tensor | None = None