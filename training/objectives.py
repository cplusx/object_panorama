from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .types import ModelInputBatch


def build_paired_supervised_batch(batch: dict[str, Any], fixed_timestep: float) -> ModelInputBatch:
    sample = batch["input"]
    batch_size = sample.shape[0]
    timestep = torch.full((batch_size,), float(fixed_timestep), device=sample.device, dtype=sample.dtype)
    return ModelInputBatch(
        sample=sample,
        condition=batch["condition"],
        condition_type_ids=batch["condition_type_ids"],
        timestep=timestep,
        target=batch["target"],
    )


def build_x0_prediction_linear_bridge_batch(
    batch: dict[str, Any],
    t_min: float,
    t_max: float,
    concat_input_to_condition: bool,
) -> ModelInputBatch:
    x0 = batch["target"]
    noise = torch.randn_like(x0)
    timestep = torch.empty(x0.shape[0], device=x0.device, dtype=x0.dtype).uniform_(float(t_min), float(t_max))
    timestep_view = timestep.view(-1, 1, 1, 1)
    x_t = (1.0 - timestep_view) * x0 + timestep_view * noise

    condition = batch["condition"]
    if concat_input_to_condition:
        condition = torch.cat([batch["input"], condition], dim=1)

    return ModelInputBatch(
        sample=x_t,
        condition=condition,
        condition_type_ids=batch["condition_type_ids"],
        timestep=timestep,
        target=x0,
    )


def compute_prediction_losses(pred: torch.Tensor, target: torch.Tensor, loss_cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    mse_weight = float(loss_cfg.get("mse_weight", 1.0))
    l1_weight = float(loss_cfg.get("l1_weight", 0.0))

    loss_mse = F.mse_loss(pred, target)
    loss_l1 = F.l1_loss(pred, target)
    loss_total = mse_weight * loss_mse + l1_weight * loss_l1
    return {
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "loss_total": loss_total,
    }